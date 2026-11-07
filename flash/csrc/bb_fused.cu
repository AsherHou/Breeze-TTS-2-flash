// Breeze-TTS-2-flash: fused skinny BF16 GEMV kernels for the backbone decode step (M = batch rows 1..4).
//   y[m, n] = sum_k xin[m, k] * W[n, k]        (nn.Linear layout, W [N, K] row-major, fp32 accumulation)
// Prologue  PRO: 0 = x as is, 1 = Qwen3RMSNorm(x) computed per CTA (eager rounding: bf16(w * bf16(x * rsqrt(..))))
// Epilogue  EPI: 0 = bf16(acc)
//                1 = bf16(res + bf16(acc))                        (residual add, eager order)
//                2 = bf16(bf16(silu(bf16(acc_gate))) * bf16(acc_up))   (Qwen3MLP act, W0 = gate, W1 = up)
// Up to three weight segments (q|k|v) are addressed without concatenating weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ uint4 ld_stream(const void* p) {
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
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16(a));
  const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16(b));
  return lo | (hi << 16);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

struct GemvArgs {
  const bf16* x; long ldx;
  const bf16* nw; float eps;
  const bf16* W0; const bf16* W1; const bf16* W2;
  int N0, N1;       // segment boundaries (rows) for EPI 0/1
  int K, Ntot;      // Ntot: output rows (EPI 2: rows of gate == rows of up)
  bf16* y; long ldy;
  const bf16* res; long ldr;
};

template <int M, int R, int U, int PRO, int EPI>
__global__ void __launch_bounds__(256) gemv_fused_kernel(const GemvArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* xs = reinterpret_cast<bf16*>(smem_raw);
  __shared__ float red[8][M];
  constexpr int NR = (EPI == 2) ? 2 * R : R;
  const int K = a.K;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
  const int row0 = (blockIdx.x * wpb + warp) * R;
  const bool live = row0 < a.Ntot;

  const bf16* wrow[NR];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int n = min(row0 + r, a.Ntot - 1);
    if (EPI == 2) {
      wrow[r] = a.W0 + (size_t)n * K;
      wrow[R + r] = a.W1 + (size_t)n * K;
    } else {
      wrow[r] = (n < a.N0) ? a.W0 + (size_t)n * K
              : (n < a.N0 + a.N1) ? a.W1 + (size_t)(n - a.N0) * K : a.W2 + (size_t)(n - a.N0 - a.N1) * K;
    }
  }
  // issue the first weight chunk before the x prologue (independent of the previous kernel's output)
  uint4 wv[U][NR];
  int k0 = lane * 8;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int k = k0 + u * 256;
      wv[u][r] = (live && k < K) ? ld_stream(wrow[r] + k) : make_uint4(0, 0, 0, 0);
    }

  // ---- prologue: x -> smem (optionally RMSNorm'ed)
  const int kv = K / 8;  // 16-byte vectors per row
  if (PRO == 0) {
    for (int i = threadIdx.x; i < M * kv; i += blockDim.x) {
      const int m = i / kv, j = i - m * kv;
      reinterpret_cast<uint4*>(xs)[i] = *reinterpret_cast<const uint4*>(a.x + m * a.ldx + j * 8);
    }
  } else {
    float ss[M];
#pragma unroll
    for (int m = 0; m < M; ++m) ss[m] = 0.f;
    for (int i = threadIdx.x; i < M * kv; i += blockDim.x) {
      const int m = i / kv, j = i - m * kv;
      const uint4 v = *reinterpret_cast<const uint4*>(a.x + m * a.ldx + j * 8);
      reinterpret_cast<uint4*>(xs)[i] = v;
      float f[8];
      cvt8(v, f);
      float s = 0.f;
#pragma unroll
      for (int t = 0; t < 8; ++t) s = fmaf(f[t], f[t], s);
#pragma unroll
      for (int mm = 0; mm < M; ++mm)
        if (mm == m) ss[mm] += s;
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float s = warp_sum(ss[m]);
      if (lane == 0) red[warp][m] = s;
    }
    __syncthreads();
    float rs[M];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float s = 0.f;
      for (int w = 0; w < wpb; ++w) s += red[w][m];
      rs[m] = rsqrtf(s * (1.0f / (float)K) + a.eps);
    }
    for (int i = threadIdx.x; i < M * kv; i += blockDim.x) {
      const int m = i / kv, j = i - m * kv;
      float f[8], w[8];
      cvt8(reinterpret_cast<const uint4*>(xs)[i], f);
      cvt8(*reinterpret_cast<const uint4*>(a.nw + j * 8), w);
      float r = 0.f;
#pragma unroll
      for (int mm = 0; mm < M; ++mm)
        if (mm == m) r = rs[mm];
      uint4 o;
      o.x = pack2(w[0] * rbf(f[0] * r), w[1] * rbf(f[1] * r));
      o.y = pack2(w[2] * rbf(f[2] * r), w[3] * rbf(f[3] * r));
      o.z = pack2(w[4] * rbf(f[4] * r), w[5] * rbf(f[5] * r));
      o.w = pack2(w[6] * rbf(f[6] * r), w[7] * rbf(f[7] * r));
      reinterpret_cast<uint4*>(xs)[i] = o;
    }
  }
  __syncthreads();
  if (!live) return;

  float acc[NR][M];
#pragma unroll
  for (int r = 0; r < NR; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  while (true) {
    uint4 cur[U][NR];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < NR; ++r) cur[u][r] = wv[u][r];
    const int kn = k0 + 256 * U;
    if (kn < K) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int r = 0; r < NR; ++r) {
          const int k = kn + u * 256;
          wv[u][r] = (k < K) ? ld_stream(wrow[r] + k) : make_uint4(0, 0, 0, 0);
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = k0 + u * 256;
      if (k < K) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float xf[8];
          cvt8(*reinterpret_cast<const uint4*>(xs + m * K + k), xf);
#pragma unroll
          for (int r = 0; r < NR; ++r) {
            float wf[8];
            cvt8(cur[u][r], wf);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[r][m] = fmaf(wf[j], xf[j], acc[r][m]);
          }
        }
      }
    }
    k0 = kn;
    if (k0 >= K) break;
  }

#pragma unroll
  for (int r = 0; r < NR; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = warp_sum(acc[r][m]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = row0 + r;
      if (n < a.Ntot) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          float v;
          if (EPI == 0) {
            v = acc[r][m];
          } else if (EPI == 1) {
            v = __bfloat162float(a.res[m * a.ldr + n]) + rbf(acc[r][m]);
          } else {
            const float g = rbf(acc[r][m]);
            const float u = rbf(acc[R + r][m]);
            v = rbf(g / (1.0f + expf(-g))) * u;
          }
          a.y[m * a.ldy + n] = __float2bfloat16(v);
        }
      }
    }
  }
}

// Standalone Qwen3RMSNorm over rows: y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))). One CTA per row.
__global__ void __launch_bounds__(256) rmsnorm_kernel(const bf16* __restrict__ x, long ldx, const bf16* __restrict__ w,
                                                      float eps, bf16* __restrict__ y, long ldy, int K) {
  const int m = blockIdx.x;
  __shared__ float red[8];
  const bf16* xr = x + m * ldx;
  float s = 0.f;
  for (int j = threadIdx.x * 8; j < K; j += blockDim.x * 8) {
    float f[8];
    cvt8(*reinterpret_cast<const uint4*>(xr + j), f);
#pragma unroll
    for (int t = 0; t < 8; ++t) s = fmaf(f[t], f[t], s);
  }
  s = warp_sum(s);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) tot += red[i];
  const float r = rsqrtf(tot * (1.0f / (float)K) + eps);
  for (int j = threadIdx.x * 8; j < K; j += blockDim.x * 8) {
    float f[8], ww[8];
    cvt8(*reinterpret_cast<const uint4*>(xr + j), f);
    cvt8(*reinterpret_cast<const uint4*>(w + j), ww);
    uint4 o;
    o.x = pack2(ww[0] * rbf(f[0] * r), ww[1] * rbf(f[1] * r));
    o.y = pack2(ww[2] * rbf(f[2] * r), ww[3] * rbf(f[3] * r));
    o.z = pack2(ww[4] * rbf(f[4] * r), ww[5] * rbf(f[5] * r));
    o.w = pack2(ww[6] * rbf(f[6] * r), ww[7] * rbf(f[7] * r));
    *reinterpret_cast<uint4*>(y + m * ldy + j) = o;
  }
}

template <int M, int R, int U, int PRO, int EPI>
void launch_t(const GemvArgs& a, int warps) {
  const int rows_per_block = warps * R;
  const int grid = (a.Ntot + rows_per_block - 1) / rows_per_block;
  const size_t smem = (size_t)M * a.K * 2;
  auto kern = gemv_fused_kernel<M, R, U, PRO, EPI>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kern<<<grid, warps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int M, int PRO, int EPI>
void dispatch_ru(const GemvArgs& a, int R, int U, int warps) {
  if (R == 1 && U == 1) launch_t<M, 1, 1, PRO, EPI>(a, warps);
  else if (R == 1 && U == 2) launch_t<M, 1, 2, PRO, EPI>(a, warps);
  else if (R == 1 && U == 4) launch_t<M, 1, 4, PRO, EPI>(a, warps);
  else if (R == 2 && U == 1) launch_t<M, 2, 1, PRO, EPI>(a, warps);
  else if (R == 2 && U == 2) launch_t<M, 2, 2, PRO, EPI>(a, warps);
  else if (R == 4 && U == 1) launch_t<M, 4, 1, PRO, EPI>(a, warps);
  else TORCH_CHECK(false, "unsupported R/U");
}

template <int PRO, int EPI>
void dispatch_m(const GemvArgs& a, int M, int R, int U, int warps) {
  if (M == 1) dispatch_ru<1, PRO, EPI>(a, R, U, warps);
  else if (M == 2) dispatch_ru<2, PRO, EPI>(a, R, U, warps);
  else if (M == 4) dispatch_ru<4, PRO, EPI>(a, R, U, warps);
  else TORCH_CHECK(false, "M must be 1, 2 or 4");
}

const bf16* bp(const at::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr()); }

void check_w(const at::Tensor& W, int K) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == at::kBFloat16 && W.is_contiguous() && W.dim() == 2 && W.size(1) == K);
}
void check_x(const at::Tensor& x, int K) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K && x.stride(1) == 1);
  TORCH_CHECK(x.stride(0) % 8 == 0 && K % 8 == 0);
}

}  // namespace

// y[:, 0:N0+N1+N2] = norm?(x) @ [W0; W1; W2]^T     (W1 / W2 may be empty tensors)
void gemv_seg(const at::Tensor& x, c10::optional<at::Tensor> nw, double eps, const at::Tensor& W0,
              const at::Tensor& W1, const at::Tensor& W2, at::Tensor y, int64_t R, int64_t U, int64_t warps) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int K = x.size(1), M = x.size(0);
  check_x(x, K);
  check_w(W0, K);
  const int N0 = W0.size(0), N1 = W1.numel() ? W1.size(0) : 0, N2 = W2.numel() ? W2.size(0) : 0;
  if (N1) check_w(W1, K);
  if (N2) check_w(W2, K);
  TORCH_CHECK(N0 % R == 0 && N1 % R == 0, "segment sizes must be multiples of R");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == M && y.size(1) == N0 + N1 + N2 && y.stride(1) == 1);
  GemvArgs a{};
  a.x = bp(x); a.ldx = x.stride(0);
  a.W0 = bp(W0); a.W1 = N1 ? bp(W1) : a.W0; a.W2 = N2 ? bp(W2) : a.W0;
  a.N0 = N0; a.N1 = N1; a.K = K; a.Ntot = N0 + N1 + N2;
  a.y = reinterpret_cast<bf16*>(y.data_ptr()); a.ldy = y.stride(0);
  if (nw.has_value() && nw->defined()) {
    TORCH_CHECK(nw->scalar_type() == at::kBFloat16 && nw->is_contiguous() && nw->numel() == K);
    a.nw = bp(*nw); a.eps = (float)eps;
    dispatch_m<1, 0>(a, M, R, U, warps);
  } else {
    dispatch_m<0, 0>(a, M, R, U, warps);
  }
}

// y = res + x @ W^T   (bf16 rounding of the GEMV result before the residual add, like eager)
void gemv_resid(const at::Tensor& x, const at::Tensor& W, const at::Tensor& res, at::Tensor y, int64_t R, int64_t U,
                int64_t warps) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int K = x.size(1), M = x.size(0);
  check_x(x, K);
  check_w(W, K);
  const int N = W.size(0);
  TORCH_CHECK(res.scalar_type() == at::kBFloat16 && res.size(0) == M && res.size(1) == N && res.stride(1) == 1);
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == M && y.size(1) == N && y.stride(1) == 1);
  GemvArgs a{};
  a.x = bp(x); a.ldx = x.stride(0);
  a.W0 = a.W1 = a.W2 = bp(W);
  a.N0 = N; a.N1 = 0; a.K = K; a.Ntot = N;
  a.y = reinterpret_cast<bf16*>(y.data_ptr()); a.ldy = y.stride(0);
  a.res = bp(res); a.ldr = res.stride(0);
  dispatch_m<0, 1>(a, M, R, U, warps);
}

// act = silu(norm(x) @ Wg^T) * (norm(x) @ Wu^T)
void gemv_gateup(const at::Tensor& x, const at::Tensor& nw, double eps, const at::Tensor& Wg, const at::Tensor& Wu,
                 at::Tensor y, int64_t R, int64_t U, int64_t warps) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int K = x.size(1), M = x.size(0);
  check_x(x, K);
  check_w(Wg, K);
  check_w(Wu, K);
  TORCH_CHECK(Wg.size(0) == Wu.size(0));
  const int N = Wg.size(0);
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == M && y.size(1) == N && y.stride(1) == 1);
  TORCH_CHECK(nw.scalar_type() == at::kBFloat16 && nw.is_contiguous() && nw.numel() == K);
  GemvArgs a{};
  a.x = bp(x); a.ldx = x.stride(0);
  a.nw = bp(nw); a.eps = (float)eps;
  a.W0 = bp(Wg); a.W1 = bp(Wu); a.W2 = a.W0;
  a.N0 = N; a.N1 = 0; a.K = K; a.Ntot = N;
  a.y = reinterpret_cast<bf16*>(y.data_ptr()); a.ldy = y.stride(0);
  dispatch_m<1, 2>(a, M, R, U, warps);
}

void rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, at::Tensor y) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int K = x.size(1), M = x.size(0);
  check_x(x, K);
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == K);
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.size(0) == M && y.size(1) == K && y.stride(1) == 1 &&
              y.stride(0) % 8 == 0);
  rmsnorm_kernel<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(bp(x), x.stride(0), bp(w), (float)eps,
                                                                 reinterpret_cast<bf16*>(y.data_ptr()), y.stride(0), K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_seg", &gemv_seg, py::arg("x"), py::arg("nw"), py::arg("eps"), py::arg("W0"), py::arg("W1"),
        py::arg("W2"), py::arg("y"), py::arg("R") = 2, py::arg("U") = 1, py::arg("warps") = 8);
  m.def("gemv_resid", &gemv_resid, py::arg("x"), py::arg("W"), py::arg("res"), py::arg("y"), py::arg("R") = 2,
        py::arg("U") = 1, py::arg("warps") = 8);
  m.def("gemv_gateup", &gemv_gateup, py::arg("x"), py::arg("nw"), py::arg("eps"), py::arg("Wg"), py::arg("Wu"),
        py::arg("y"), py::arg("R") = 1, py::arg("U") = 1, py::arg("warps") = 8);
  m.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("w"), py::arg("eps"), py::arg("y"));
}
