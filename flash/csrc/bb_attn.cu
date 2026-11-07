// Breeze-TTS-2-flash: GQA flash-decoding attention for the backbone single-token decode step.
//
// Semantics reproduce transformers eager_attention_forward (Qwen3) for q_len == 1:
//   s   = bf16(q @ k^T)            (fp32 accumulation)
//   s   = bf16(s * float(scaling))
//   s   = s + mask                 (bf16 add if mask is bf16, fp32 add if mask is fp32)
//   p   = bf16(softmax_fp32(s))    (same max / sum order as PyTorch softmax_warp_forward<.., 11>)
//   out = bf16(p @ v)              (fp32 accumulation)
// KV positions whose additive mask is <= -1e30 (finfo.min / -inf) contribute exactly 0 in eager, so they are
// skipped: whole 64-position chunks are skipped without touching K/V, which makes the cost proportional to the
// number of valid positions. All decisions are made on device -> CUDA-graph capturable with a fixed grid.
//
// Two launches per call:
//   phase 1 (grid nch x NKV x B): mask scan, [fused: q_norm/k_norm + RoPE + KV-cache write], QK^T for one chunk
//                                and both q heads of the GQA group, chunk max.
//   phase 2 (grid nch x NKV x B): exact softmax statistics (global max + PyTorch-order sum), bf16 probabilities,
//                                P@V for the chunk, last-arriving CTA of each (b, kv-head) combines the partials.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

typedef __nv_bfloat16 bf16;
constexpr int D = 128;   // head_dim
constexpr int CH = 64;   // KV positions per chunk
constexpr int NT = 128;  // threads per CTA
constexpr int G = 2;     // q heads per kv head (16 / 8)
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ uint4 ld_cg(const void* p) {
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_nc(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void cvt8(const uint4& v, float* f) {
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }

template <typename MT> __device__ __forceinline__ float mload(const MT* m, long i);
template <> __device__ __forceinline__ float mload<bf16>(const bf16* m, long i) { return __bfloat162float(m[i]); }
template <> __device__ __forceinline__ float mload<float>(const float* m, long i) { return m[i]; }

struct AttnArgs {
  const bf16* q; long q_sb, q_sh;        // non-fused query [B, NH, 1, D] (element strides), d contiguous
  const bf16* qkv; long qkv_sb;          // fused: raw projections [B, (NH + 2 NKV) * D]
  const bf16* qn_w; const bf16* kn_w; float eps;
  const bf16* cos; const bf16* sin; long cs_sb;  // [B, 1, D]
  bf16* kc; bf16* vc; long c_sb, c_sh;   // StaticCache layer tensors [B, NKV, S, D] (pos stride D)
  const void* mask; long m_sb;           // [B, 1, 1, S] additive mask, last dim contiguous
  const long* cache_pos;                 // fused: device scalar (slot to write)
  float scaling;
  float* scores;                         // [B, NH, S] fp32 workspace
  float* cmax;                           // [B, NH, nch]
  float* part;                           // [B, NH, nch, D]
  int* counters;                         // [B, NKV] (zero-initialised, self-resetting)
  bf16* out;                             // [B, 1, NH, D]
  float* dbg;                            // optional [B, NH, S] probabilities (bf16-rounded) for verification
  int S, NH, NKV, nch;
};

// Qwen3RMSNorm over head_dim followed by RoPE (rotate_half), eager rounding points:
//   y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps)));  out = bf16(bf16(y * cos) + bf16(rotate_half(y) * sin))
// One warp per head; lane owns dims [4*lane, 4*lane+4).
__device__ __forceinline__ void norm_rope_warp(const bf16* x, const bf16* w, float eps, const bf16* cs,
                                               const bf16* sn, int lane, float out[4]) {
  const int d0 = lane * 4;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = b2f(x[d0 + j]);
  float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
  const float r = rsqrtf(ss * (1.0f / D) + eps);
  float y[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) y[j] = rbf(b2f(w[d0 + j]) * rbf(v[j] * r));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float partner = __shfl_xor_sync(FULL, y[j], 16);  // dim d0 + j +/- 64
    const float rot = (lane < 16) ? -partner : partner;
    out[j] = rbf(rbf(y[j] * b2f(cs[d0 + j])) + rbf(rot * b2f(sn[d0 + j])));
  }
}

template <typename MT, bool FUSED>
__global__ void __launch_bounds__(NT) attn_qk_kernel(const AttnArgs a) {
  const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int p0 = c * CH;
  __shared__ float qs[G][D];
  __shared__ float ms[CH];
  __shared__ float red[NT / 16][G];

  bool valid = false;
  if (tid < CH) {
    const int pos = p0 + tid;
    const float m = (pos < a.S) ? mload<MT>(reinterpret_cast<const MT*>(a.mask), (long)b * a.m_sb + pos) : -INFINITY;
    valid = m > -1e30f;
    ms[tid] = m;
  }
  // q heads of this GQA group (warps 0..G-1): RMSNorm + RoPE (fused) or plain load, overlapped with the mask load
  if (warp < G) {
    const int hq = h * G + warp;
    float qv[4];
    if (FUSED) {
      norm_rope_warp(a.qkv + (long)b * a.qkv_sb + (long)hq * D, a.qn_w, a.eps, a.cos + (long)b * a.cs_sb,
                     a.sin + (long)b * a.cs_sb, lane, qv);
    } else {
      const bf16* qp = a.q + (long)b * a.q_sb + (long)hq * a.q_sh + lane * 4;
#pragma unroll
      for (int j = 0; j < 4; ++j) qv[j] = b2f(qp[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) qs[warp][lane * 4 + j] = qv[j];
  }
  // fused: the CTA owning slot cache_pos writes the new k (k_norm + RoPE) and v (warps 2, 3)
  if (FUSED) {
    const int pos = (int)a.cache_pos[0];
    if (pos >= p0 && pos < p0 + CH && pos < a.S) {
      const long off = (long)b * a.c_sb + (long)h * a.c_sh + (long)pos * D;
      if (warp == 2) {
        float kv[4];
        norm_rope_warp(a.qkv + (long)b * a.qkv_sb + (long)(a.NH + h) * D, a.kn_w, a.eps, a.cos + (long)b * a.cs_sb,
                       a.sin + (long)b * a.cs_sb, lane, kv);
#pragma unroll
        for (int j = 0; j < 4; ++j) a.kc[off + lane * 4 + j] = __float2bfloat16(kv[j]);
      } else if (warp == 3 && lane < 16) {
        const uint4 vv = *reinterpret_cast<const uint4*>(a.qkv + (long)b * a.qkv_sb + (long)(a.NH + a.NKV + h) * D + lane * 8);
        *reinterpret_cast<uint4*>(a.vc + off + lane * 8) = vv;
      }
    }
  }
  // barrier: publishes qs and makes the cache write above visible to the (L2) K loads below
  if (!__syncthreads_or(valid)) {
    if (tid < G) a.cmax[((long)b * a.NH + h * G + tid) * a.nch + c] = -INFINITY;
    return;
  }

  // K loads for this chunk: thread -> position group pg (positions pg + 8*it), dims [8*l16, 8*l16+8)
  const int pg = tid >> 4, l16 = tid & 15;
  const bf16* kbase = a.kc + (long)b * a.c_sb + (long)h * a.c_sh + (long)p0 * D + l16 * 8;
  uint4 kr[CH / 8];
#pragma unroll
  for (int it = 0; it < CH / 8; ++it) {
    const int pi = pg + 8 * it;
    kr[it] = (p0 + pi < a.S) ? ld_cg(kbase + (long)pi * D) : make_uint4(0, 0, 0, 0);
  }

  float qf[G][8];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int j = 0; j < 8; ++j) qf[g][j] = qs[g][l16 * 8 + j];

  float smax[G];
#pragma unroll
  for (int g = 0; g < G; ++g) smax[g] = -INFINITY;
#pragma unroll
  for (int it = 0; it < CH / 8; ++it) {
    const int pi = pg + 8 * it;
    float kf[8];
    cvt8(kr[it], kf);
    float dsum[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) s = fmaf(qf[g][j], kf[j], s);
      dsum[g] = s;
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) dsum[g] += __shfl_xor_sync(FULL, dsum[g], o);
    if (l16 == 0) {
      const float m = ms[pi];
      const bool ok = (p0 + pi < a.S) && (m > -1e30f);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        float s = rbf(dsum[g]);
        s = rbf(s * a.scaling);
        s = (sizeof(MT) == 2) ? rbf(s + m) : (s + m);
        s = ok ? s : -INFINITY;
        if (p0 + pi < a.S) a.scores[((long)b * a.NH + h * G + g) * a.S + p0 + pi] = s;
        smax[g] = fmaxf(smax[g], s);
      }
    }
  }
  if (l16 == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) red[pg][g] = smax[g];
  }
  __syncthreads();
  if (tid < G) {
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NT / 16; ++i) m = fmaxf(m, red[i][tid]);
    a.cmax[((long)b * a.NH + h * G + tid) * a.nch + c] = m;
  }
}

constexpr int MAXIT = 64;  // S <= 2048: softmax elements per lane (PyTorch warp softmax, log2_elements = 11)
constexpr int MAXCH = 32;  // S <= 2048: chunks per row

__global__ void __launch_bounds__(NT) attn_pv_kernel(const AttnArgs a) {
  const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int p0 = c * CH;
  __shared__ unsigned long long s_am;
  __shared__ float sM[G], sL[G];
  __shared__ float ps[G][CH];
  __shared__ float red[NT / 16][G][D];
  __shared__ int s_last;

  // prefetch this thread's own-chunk score (used only if the chunk is active)
  const int my_g = tid / CH, my_pi = tid % CH;  // G * CH == NT
  const int my_pos = p0 + my_pi;
  const float* my_sc = a.scores + ((long)b * a.NH + h * G + my_g) * a.S;
  const float my_s = (my_pos < a.S) ? __ldcg(my_sc + my_pos) : -INFINITY;
  const float* cm0 = a.cmax + ((long)b * a.NH + h * G) * a.nch;
  const bool active = cm0[c] > -INFINITY;  // validity is per (b, chunk); identical for both heads

  const int pg = tid >> 4, l16 = tid & 15;
  uint4 vr[CH / 8];
  if (active) {
    const bf16* vbase = a.vc + (long)b * a.c_sb + (long)h * a.c_sh + (long)p0 * D + l16 * 8;
#pragma unroll
    for (int it = 0; it < CH / 8; ++it) {
      const int pi = pg + 8 * it;
      vr[it] = (p0 + pi < a.S) ? ld_nc(vbase + (long)pi * D) : make_uint4(0, 0, 0, 0);
    }
  }
  if (warp < G) {
    const int hq = h * G + warp;
    const float* cm = a.cmax + ((long)b * a.NH + hq) * a.nch;
    const float c0 = lane < a.nch ? cm[lane] : -INFINITY;
    const float c1 = lane + 32 < a.nch ? cm[lane + 32] : -INFINITY;
    const unsigned lo = __ballot_sync(FULL, c0 > -INFINITY), hi = __ballot_sync(FULL, c1 > -INFINITY);
    const unsigned long long am = ((unsigned long long)hi << 32) | lo;
    if (warp == 0 && lane == 0) s_am = am;
    if (active) {
      // exact replica of softmax_warp_forward<float,float,float,log2=11>: lane l owns elements l + 32*it,
      // sequential fp32 sum of exp(x - max) over it, then xor-butterfly 16..1. Skipped (masked) elements
      // are -inf -> exp() == 0 exactly, so adding them (or not) leaves the sum bit-identical.
      float m = fmaxf(c0, c1);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
      const float* sc = a.scores + ((long)b * a.NH + hq) * a.S;
      float x[MAXIT];
#pragma unroll
      for (int it = 0; it < MAXIT; ++it) {
        const int pos = lane + 32 * it;
        x[it] = (((am >> ((it * 32) / CH)) & 1ull) && pos < a.S) ? __ldcg(sc + pos) : -INFINITY;
      }
      float sum = 0.f;
#pragma unroll
      for (int it = 0; it < MAXIT; ++it) sum += expf(x[it] - m);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL, sum, o);
      if (lane == 0) { sM[warp] = m; sL[warp] = sum; }
    }
  }
  __syncthreads();
  const unsigned long long am = s_am;

  if (active) {
    {
      float pr = 0.f;
      if (my_pos < a.S) {
        pr = rbf(expf(my_s - sM[my_g]) / sL[my_g]);
        if (a.dbg) a.dbg[((long)b * a.NH + h * G + my_g) * a.S + my_pos] = pr;
      }
      ps[my_g][my_pi] = pr;
    }
    __syncthreads();
    float acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[g][j] = 0.f;
#pragma unroll
    for (int it = 0; it < CH / 8; ++it) {
      const int pi = pg + 8 * it;
      float vf[8];
      cvt8(vr[it], vf);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float p = ps[g][pi];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[g][j] = fmaf(p, vf[j], acc[g][j]);
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < 8; ++j) red[pg][g][l16 * 8 + j] = acc[g][j];
    __syncthreads();
    for (int o = tid; o < G * D; o += NT) {
      const int g = o / D, d = o % D;
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < NT / 16; ++i) s += red[i][g][d];
      a.part[(((long)b * a.NH + h * G + g) * a.nch + c) * D + d] = s;
    }
    __threadfence();
  } else if (a.dbg) {
    if (my_pos < a.S) a.dbg[((long)b * a.NH + h * G + my_g) * a.S + my_pos] = 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(&a.counters[b * a.NKV + h], 1);
    s_last = (prev == a.nch - 1);
  }
  __syncthreads();
  if (s_last) {
    __threadfence();
    // all partial loads issued at once (predicated on chunk activity), then summed in chunk order
    for (int o = tid; o < G * D; o += NT) {
      const int g = o / D, d = o % D, hq = h * G + g;
      const float* pp = a.part + ((long)b * a.NH + hq) * a.nch * D + d;
      float v[MAXCH];
#pragma unroll
      for (int ci = 0; ci < MAXCH; ++ci) v[ci] = (ci < a.nch && ((am >> ci) & 1ull)) ? __ldcg(pp + (long)ci * D) : 0.f;
      float s = 0.f;
#pragma unroll
      for (int ci = 0; ci < MAXCH; ++ci) s += v[ci];
      a.out[((long)b * a.NH + hq) * D + d] = __float2bfloat16(s);
    }
    if (tid == 0) a.counters[b * a.NKV + h] = 0;
  }
}


void check_ws(const at::Tensor& scores, const at::Tensor& cmax, const at::Tensor& part, const at::Tensor& counters,
              int B, int NH, int NKV, int S, int nch) {
  TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.is_contiguous() && scores.numel() >= (long)B * NH * S);
  TORCH_CHECK(cmax.scalar_type() == at::kFloat && cmax.is_contiguous() && cmax.numel() >= (long)B * NH * nch);
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() >= (long)B * NH * nch * D);
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.is_contiguous() && counters.numel() >= (long)B * NKV);
}

void launch(const AttnArgs& a, int B, bool fused, at::ScalarType mdt) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(a.nch, a.NKV, B);
  if (mdt == at::kBFloat16) {
    if (fused) attn_qk_kernel<bf16, true><<<grid, NT, 0, stream>>>(a);
    else attn_qk_kernel<bf16, false><<<grid, NT, 0, stream>>>(a);
  } else {
    if (fused) attn_qk_kernel<float, true><<<grid, NT, 0, stream>>>(a);
    else attn_qk_kernel<float, false><<<grid, NT, 0, stream>>>(a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  attn_pv_kernel<<<grid, NT, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_common(AttnArgs& a, const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& mask, double scaling,
                 at::Tensor& scores, at::Tensor& cmax, at::Tensor& part, at::Tensor& counters, at::Tensor& out,
                 const c10::optional<at::Tensor>& dbg, int B, int NH) {
  TORCH_CHECK(kc.dim() == 4 && vc.dim() == 4 && kc.sizes() == vc.sizes() && kc.strides() == vc.strides());
  TORCH_CHECK(kc.scalar_type() == at::kBFloat16 && vc.scalar_type() == at::kBFloat16);
  TORCH_CHECK(kc.size(0) == B && kc.size(3) == D && kc.stride(3) == 1 && kc.stride(2) == D);
  const int NKV = kc.size(1), S = kc.size(2);
  TORCH_CHECK(NH == NKV * G, "only GQA group size 2 supported");
  const int nch = (S + CH - 1) / CH;
  TORCH_CHECK(nch <= MAXCH && S <= 32 * MAXIT, "S too large (max 2048)");
  TORCH_CHECK(mask.dim() == 4 && mask.size(-1) >= S && mask.stride(-1) == 1 && mask.size(2) == 1);
  TORCH_CHECK(mask.size(0) == B || mask.size(0) == 1);
  TORCH_CHECK(mask.scalar_type() == at::kBFloat16 || mask.scalar_type() == at::kFloat);
  check_ws(scores, cmax, part, counters, B, NH, NKV, S, nch);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == (long)B * NH * D);
  a.kc = reinterpret_cast<bf16*>(kc.data_ptr());
  a.vc = reinterpret_cast<bf16*>(vc.data_ptr());
  a.c_sb = kc.stride(0);
  a.c_sh = kc.stride(1);
  a.mask = mask.data_ptr();
  a.m_sb = mask.size(0) == 1 ? 0 : mask.stride(0);
  a.scaling = (float)scaling;
  a.scores = scores.data_ptr<float>();
  a.cmax = cmax.data_ptr<float>();
  a.part = part.data_ptr<float>();
  a.counters = counters.data_ptr<int>();
  a.out = reinterpret_cast<bf16*>(out.data_ptr());
  a.dbg = nullptr;
  if (dbg.has_value() && dbg->defined()) {
    TORCH_CHECK(dbg->scalar_type() == at::kFloat && dbg->is_contiguous() && dbg->numel() >= (long)B * NH * S);
    a.dbg = dbg->data_ptr<float>();
  }
  a.S = S;
  a.NH = NH;
  a.NKV = NKV;
  a.nch = nch;
}

}  // namespace

// query: [B, NH, 1, D] bf16 (any batch/head strides, d contiguous); key/value: StaticCache [B, NKV, S, D].
// out: [B, 1, NH, D] bf16 contiguous (== eager attn_output.transpose(1, 2).contiguous()).
void attn_decode(const at::Tensor& query, const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& mask,
                 double scaling, at::Tensor scores, at::Tensor cmax, at::Tensor part, at::Tensor counters,
                 at::Tensor out, c10::optional<at::Tensor> dbg) {
  const c10::cuda::CUDAGuard guard(query.device());
  TORCH_CHECK(query.dim() == 4 && query.size(2) == 1 && query.size(3) == D && query.stride(3) == 1);
  TORCH_CHECK(query.scalar_type() == at::kBFloat16);
  const int B = query.size(0), NH = query.size(1);
  AttnArgs a{};
  fill_common(a, kc, vc, mask, scaling, scores, cmax, part, counters, out, dbg, B, NH);
  a.q = reinterpret_cast<const bf16*>(query.data_ptr());
  a.q_sb = query.stride(0);
  a.q_sh = query.stride(1);
  launch(a, B, false, mask.scalar_type());
}

// Fused: qkv = raw [B, (NH + 2 NKV) * D] projections (q | k | v). Applies q_norm / k_norm + RoPE, writes the new
// k (normed + roped) and v into the cache at slot cache_pos[0], then attends.
void attn_decode_fused(const at::Tensor& qkv, const at::Tensor& qn_w, const at::Tensor& kn_w, double eps,
                       const at::Tensor& cos, const at::Tensor& sin, at::Tensor kc, at::Tensor vc,
                       const at::Tensor& mask, const at::Tensor& cache_pos, double scaling, at::Tensor scores,
                       at::Tensor cmax, at::Tensor part, at::Tensor counters, at::Tensor out,
                       c10::optional<at::Tensor> dbg) {
  const c10::cuda::CUDAGuard guard(qkv.device());
  TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.scalar_type() == at::kBFloat16);
  const int B = qkv.size(0);
  const int NKV = kc.size(1), NH = NKV * G;
  TORCH_CHECK(qkv.size(1) == (NH + 2 * NKV) * D);
  TORCH_CHECK(qn_w.numel() == D && kn_w.numel() == D && qn_w.scalar_type() == at::kBFloat16 &&
              kn_w.scalar_type() == at::kBFloat16 && qn_w.is_contiguous() && kn_w.is_contiguous());
  TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16 && cos.is_contiguous() &&
              sin.is_contiguous() && cos.numel() == (long)B * D && sin.numel() == (long)B * D);
  TORCH_CHECK(cache_pos.scalar_type() == at::kLong && cache_pos.numel() >= 1);
  AttnArgs a{};
  fill_common(a, kc, vc, mask, scaling, scores, cmax, part, counters, out, dbg, B, NH);
  a.qkv = reinterpret_cast<const bf16*>(qkv.data_ptr());
  a.qkv_sb = qkv.stride(0);
  a.qn_w = reinterpret_cast<const bf16*>(qn_w.data_ptr());
  a.kn_w = reinterpret_cast<const bf16*>(kn_w.data_ptr());
  a.eps = (float)eps;
  a.cos = reinterpret_cast<const bf16*>(cos.data_ptr());
  a.sin = reinterpret_cast<const bf16*>(sin.data_ptr());
  a.cs_sb = D;
  a.cache_pos = cache_pos.data_ptr<long>();
  launch(a, B, true, mask.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_decode", &attn_decode, "GQA flash-decoding attention (q_len=1)", py::arg("query"), py::arg("kc"),
        py::arg("vc"), py::arg("mask"), py::arg("scaling"), py::arg("scores"), py::arg("cmax"), py::arg("part"),
        py::arg("counters"), py::arg("out"), py::arg("dbg") = py::none());
  m.def("attn_decode_fused", &attn_decode_fused, "q/k norm + RoPE + KV write + GQA flash-decoding attention",
        py::arg("qkv"), py::arg("qn_w"), py::arg("kn_w"), py::arg("eps"), py::arg("cos"), py::arg("sin"),
        py::arg("kc"), py::arg("vc"), py::arg("mask"), py::arg("cache_pos"), py::arg("scaling"), py::arg("scores"),
        py::arg("cmax"), py::arg("part"), py::arg("counters"), py::arg("out"), py::arg("dbg") = py::none());
}
