// Breeze-TTS-2-flash: depth-decoder FRAME megakernel with LN12 compressed weights.
//
// ONE cooperative launch per audio frame runs all 15 depth-decoder steps AND the official sampling rule:
//   for s in 0..14:  forward(step s) -> logits[B,V] -> [barrier] -> fused CFG/temperature/top-k/top-p/
//                    softmax/argmax(p/q) sampler -> token -> [barrier] -> next step embeds the token
// The random numbers q ~ Exp(1) are generated OUTSIDE by 15 torch exponential_() calls with the same sizes and
// order as the official 15 torch.multinomial() calls (multinomial(p,1) == argmax(p / Exp(1)) in PyTorch 2.9,
// verified bit-exact, also under CUDA graphs) -> identical RNG consumption and identical sampling rule.
//
// Weights: every matmul stage reads LN12 compressed weights (about 12.1 bits per weight instead of 16, lossless:
// the decoded values are the original BF16 bits, so results are bitwise identical to the BF16 kernel).
// Format: see flash/ln12_format.py and ln12.cuh.
//
// The K/V cache of a layer is staged into shared memory before the attention part of stage S2 (one L2 round trip).
// Optional build flags (results are identical with and without them; only memory scheduling changes):
//   -DDD_PF_EVICT_LAST  weight prefetches use prefetch.global.L2::evict_last; dd2_init sets the process's
//                       persisting-L2 set-aside to 0 (required for evict_last to help) and checks it.
//   -DDD_STEP_TABLE     steps 1..14 take the projector output and the layer-0 q|k|v pre-activations from a
//                       per-token table built once by dd2_tab_build() with the same device functions and the same
//                       accumulation order (bit-exact), instead of recomputing them every frame.
// Tuning constants (DD_PF_*, DD_S3_EARLY_PF, DD_FAST_TOPK, DD_NT) are L2 prefetch sizes in KiB / launch shape
// chosen for an RTX 4080 SUPER (80 SMs, 64 MiB L2).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "depth_common.cuh"
#include "ln12.cuh"
#define PFR(CWX, BFP, R0, R1, KK, MB) prefetch_rows_c(CWX, R0, R1, KK, MB)


struct P2 {
  LayerW L[NL];
  const bf16 *embed, *proj, *normf, *head, *cos_tab, *sin_tab;
  bf16 *kc, *vc, *xres, *qb, *act, *ob;
  const int64_t* forced;    // optional [NCB][B] teacher-forced tokens (verification)
  float* logits;      // [B][V] scratch
  float* dbg;         // optional [NCB][B][V]
  unsigned* bar;
  const int64_t* first_cb;  // [B]
  const bf16* bh;           // [B][AE]
  int64_t* tokbuf;          // [B]   (official _tok_buf)
  int64_t* outtok;          // [B][NCB] (official output_tokens)
  const float* temp;        // [half]
  const int64_t* topk;      // [half]
  const float* topp;        // [half]
  const int64_t* dosample;  // [half]
  const float* gs;          // [half]
  const float* q;           // [NCB][half][V]
#ifdef DD_STEP_TABLE
  const bf16* ptab;  // [TAB_ROWS][H]    = bf16(<proj row n, embed[V + t]>)
  const bf16* ttab;  // [TAB_ROWS][1536] = pre-RoPE layer-0 q|k|v of RMSNorm(ptab row t)
#endif
  float eps, scaling;
};

// shared-memory staging area sU: rows of FF bf16 that fit (B=1: 2 rows = 32 KB, B>=2: 4 rows = 64 KB)
template <int B>
struct SU { static constexpr int ROWS = B == 1 ? 2 : 4; static constexpr int ELEMS = ROWS * 8192; };

// ------------------------------------------------------------------ one forward step (ends after head)
// Every stage's first weight unit is preloaded into registers before the preceding barrier (dot_rows_pre).
__device__ __forceinline__ void qkv_unit(const LayerW& L, int u, const bf16*& W, int& kind, int& hh, int& d) {
  if (u < 512) { kind = 0; hh = u >> 6; d = u & 63; W = L.wq; }
  else if (u < 640) { kind = 1; hh = (u - 512) >> 6; d = (u - 512) & 63; W = L.wk; }
  else { kind = 2; hh = (u - 640) >> 6; d = (u - 640) & 63; W = L.wv; }
}

#ifdef DD_STEP_TABLE
constexpr int TAB_ROWS = (NCB - 1) * V;  // steps 1..14
__device__ __forceinline__ int tab_idx(const int64_t* tok, int b, int step) {
  long long id = (long long)__ldcg(reinterpret_cast<const long long*>(tok) + b) + (long long)step * V;
  id = id < 0 ? 0 : (id > 16LL * V - 1 ? 16LL * V - 1 : id);  // identical to the embedding gather
  long long t = id - V;                                          // step >= 1 and tok in [0, V-1] => t in [0, TAB_ROWS)
  t = t < 0 ? 0 : (t > TAB_ROWS - 1 ? TAB_ROWS - 1 : t);
  return (int)t;
}
#endif
template <int B, int T>
__device__ __forceinline__ void forward_step(const P2& p, const int step, unsigned& epoch, bf16* sU, bf16* sX) {
  constexpr int M = B * T;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int G = gridDim.x, bid = blockIdx.x;
  const int rH0 = H * bid / G, rH1 = H * (bid + 1) / G;
  const int rF0 = FF * bid / G, rF1 = FF * (bid + 1) / G;
  const int rQ0 = 768 * bid / G, rQ1 = 768 * (bid + 1) / G;
  const int rV0 = V * bid / G, rV1 = V * (bid + 1) / G;
  const int64_t* tok = step == 0 ? p.first_cb : p.tokbuf;
  const int uQ = rQ0 + warp, nH = rH0 + warp, nF = rF0 + warp, nV = rV0 + warp;

#ifdef DD_STEP_TABLE
  if (step > 0) {
    const int nr = rH1 - rH0;
    for (int i = threadIdx.x; i < M * nr; i += NT) {
      const int m = i / nr, n = rH0 + i % nr, b = m / T;
      __stcg(p.xres + m * H + n, p.ptab[(size_t)tab_idx(tok, b, step) * H + n]);
    }
  } else {
#endif
  // E: embedding gather + projector
  for (int m = warp; m < M; m += NW) {
    const int b = m / T, t = m % T;
    const uint4* src;
    if (step == 0 && t == 0) {
      src = reinterpret_cast<const uint4*>(p.bh + (size_t)b * AE);
    } else {
      long long id = (long long)__ldcg(reinterpret_cast<const long long*>(tok) + b) + (step == 0 ? 0 : (long long)step * V);
      id = id < 0 ? 0 : (id > 16LL * V - 1 ? 16LL * V - 1 : id);
      src = reinterpret_cast<const uint4*>(p.embed + (size_t)id * AE);
    }
#pragma unroll
    for (int i = 0; i < AE / 256; ++i) reinterpret_cast<uint4*>(sU + m * AE)[i * 32 + lane] = __ldcg(src + i * 32 + lane);
  }
  __syncthreads();
  for (int n = nH; n < rH1; n += NW) {
    float acc[1][M] = {};

    { const CW* cw[1] = {&c_cw[NL * 7]}; const int nr[1] = {n}; dot_rows_c<M, 1, AE>(cw, nr, sU, acc); }

    if (lane < M) __stcg(p.xres + lane * H + n, __float2bfloat16_rn(acc[0][lane]));
  }
  grid_sync(p.bar, epoch);
#ifdef DD_STEP_TABLE
  }
#endif

#pragma unroll 1
  for (int l = 0; l < NL; ++l) {
    const LayerW& L = p.L[l];
    // S1: RMSNorm + QKV + RoPE + KV write
#ifndef DD_PF_O_AT_S1
#define DD_PF_O_AT_S1 32
#endif
#ifndef DD_PF_S4_AT_S3
#define DD_PF_S4_AT_S3 64
#endif
#ifndef DD_PF_S1_AT_S4
#define DD_PF_S1_AT_S4 48
#endif
    if (DD_PF_O_AT_S1 > 0) PFR(cwl(l, CO), L.wo, rH0, rH1, H, DD_PF_O_AT_S1 * 1024);
#ifdef DD_STEP_TABLE
    const bool tabq = (step > 0 && l == 0);
#else
    constexpr bool tabq = false;
#endif
    if (!tabq) load_norm<M>(p.xres, L.ln1, sX, p.eps);
    for (int u = uQ; u < rQ1; u += NW) {
      const bf16* W; int kind, hh, d; qkv_unit(L, u, W, kind, hh, d);
      const int row = hh * HD + d;
      float acc[2][M] = {};

#ifdef DD_STEP_TABLE
      if (tabq) {
        const int base = kind == 0 ? 0 : (kind == 1 ? QD : QD + NKV * HD);
#pragma unroll
        for (int m = 0; m < M; ++m)
          if (lane == m) {
            const size_t ti = (size_t)tab_idx(tok, m / T, step) * 1536 + base + row;
            acc[0][m] = bf2f(p.ttab[ti]);
            acc[1][m] = bf2f(p.ttab[ti + 64]);
          }
      } else
#endif
      {
      { const CW* cw[2] = {&cwl(l, kind), &cwl(l, kind)}; const int nr[2] = {row, row + 64}; dot_rows_c<M, 2, H>(cw, nr, sX, acc); }
      }

      if (lane < M) {
        const int m = lane, b = m / T, t = m % T;
        const int pos = step == 0 ? t : step + 1;
        float y0 = rbf(acc[0][m]), y1 = rbf(acc[1][m]);
        if (kind < 2) {
          const float c0 = bf2f(p.cos_tab[pos * HD + d]), s0 = bf2f(p.sin_tab[pos * HD + d]);
          const float c1 = bf2f(p.cos_tab[pos * HD + d + 64]), s1 = bf2f(p.sin_tab[pos * HD + d + 64]);
          const float o0 = rbf(y0 * c0 - y1 * s0), o1 = rbf(y1 * c1 + y0 * s1);
          y0 = o0; y1 = o1;
        }
        if (kind == 0) {
          __stcg(p.qb + m * QD + row, __float2bfloat16_rn(y0));
          __stcg(p.qb + m * QD + row + 64, __float2bfloat16_rn(y1));
        } else {
          bf16* cache = kind == 1 ? p.kc : p.vc;
          bf16* dst = cache + ((((size_t)l * B + b) * NKV + hh) * MAXP + pos) * HD;
          __stcg(dst + d, __float2bfloat16_rn(y0));
          __stcg(dst + d + 64, __float2bfloat16_rn(y1));
        }
      }
    }
    grid_sync(p.bar, epoch);

    // S2: GQA attention (redundant per block) + O-proj.
#ifndef DD_S3_EARLY_PF
#define DD_S3_EARLY_PF 128
#endif
    // the attention part is latency-bound and leaves DRAM idle: start pulling this block's S3 (gate/up) rows into L2
    // Stage this layer's K/V cache [B][NKV][0..maxpos][HD] into shared memory (sU is free here) in ONE L2 round
    // trip, then every warp computes one (row, q-head) pair from shared memory.
    {
      const int maxpos = step == 0 ? 1 : step + 1;
      const int npos = maxpos + 1;
      bf16* sK = sU;                              // [B][NKV][MAXP][HD]
      bf16* sV = sU + B * NKV * MAXP * HD;
      const int vec_per_bk = npos * HD / 8;       // uint4 per (b,kvh)
      const int total = B * NKV * vec_per_bk;
      const bf16* kl = p.kc + (size_t)l * B * NKV * MAXP * HD;
      const bf16* vl = p.vc + (size_t)l * B * NKV * MAXP * HD;
      for (int i = threadIdx.x; i < total; i += NT) {
        const int bk = i / vec_per_bk, r = i % vec_per_bk;
        const size_t off = (size_t)bk * MAXP * HD + (size_t)r * 8;
        *reinterpret_cast<uint4*>(sK + off) = __ldcg(reinterpret_cast<const uint4*>(kl + off));
        *reinterpret_cast<uint4*>(sV + off) = __ldcg(reinterpret_cast<const uint4*>(vl + off));
      }
      __syncthreads();
    if (DD_S3_EARLY_PF > 0) {
      PFR(cwl(l, CG), L.wg, rF0, rF1, H, DD_S3_EARLY_PF * 1024);
      PFR(cwl(l, CU), L.wu, rF0, rF1, H, DD_S3_EARLY_PF * 1024);
    }
      for (int pr = warp; pr < M * NH; pr += NW) {
        const int m = pr / NH, hq = pr % NH, b = m / T, t = m % T;
        const int pos = step == 0 ? t : step + 1;
        const int kvh = hq / (NH / NKV);
        const bf16* kb = sK + ((size_t)(b * NKV + kvh) * MAXP) * HD + lane * 4;
        const bf16* vb = sV + ((size_t)(b * NKV + kvh) * MAXP) * HD + lane * 4;
        const uint2 q2 = __ldcg(reinterpret_cast<const uint2*>(p.qb + m * QD + hq * HD + lane * 4));
        const float qf0 = __uint_as_float(q2.x << 16), qf1 = __uint_as_float(q2.x & 0xffff0000u);
        const float qf2 = __uint_as_float(q2.y << 16), qf3 = __uint_as_float(q2.y & 0xffff0000u);
        float s[MAXP];
#pragma unroll
        for (int j = 0; j < MAXP; ++j) {
          if (j <= pos) {
            const uint2 k2 = *reinterpret_cast<const uint2*>(kb + j * HD);
            s[j] = qf0 * __uint_as_float(k2.x << 16) + qf1 * __uint_as_float(k2.x & 0xffff0000u) +
                   qf2 * __uint_as_float(k2.y << 16) + qf3 * __uint_as_float(k2.y & 0xffff0000u);
          } else {
            s[j] = 0.f;
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
          for (int j = 0; j < MAXP; ++j) s[j] += __shfl_xor_sync(0xffffffffu, s[j], o);
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < MAXP; ++j) {
          s[j] = rbf(s[j]) * p.scaling;
          if (j <= pos) mx = fmaxf(mx, s[j]);
        }
        float den = 0.f;
#pragma unroll
        for (int j = 0; j < MAXP; ++j) {
          s[j] = j <= pos ? expf(s[j] - mx) : 0.f;
          den += s[j];
        }
        float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < MAXP; ++j) {
          if (j <= pos) {
            const float pj = rbf(s[j] / den);
            const uint2 vpair = *reinterpret_cast<const uint2*>(vb + j * HD);
            o[0] = fmaf(pj, __uint_as_float(vpair.x << 16), o[0]);
            o[1] = fmaf(pj, __uint_as_float(vpair.x & 0xffff0000u), o[1]);
            o[2] = fmaf(pj, __uint_as_float(vpair.y << 16), o[2]);
            o[3] = fmaf(pj, __uint_as_float(vpair.y & 0xffff0000u), o[3]);
          }
        }
        __align__(8) bf16 ob[4] = {__float2bfloat16_rn(o[0]), __float2bfloat16_rn(o[1]), __float2bfloat16_rn(o[2]),
                                   __float2bfloat16_rn(o[3])};
        *reinterpret_cast<uint2*>(sX + m * H + hq * HD + lane * 4) = *reinterpret_cast<uint2*>(ob);
      }
    }
    __syncthreads();
    for (int n = nH; n < rH1; n += NW) {
      float acc[1][M] = {};

      { const CW* cw[1] = {&cwl(l, CO)}; const int nr[1] = {n}; dot_rows_c<M, 1, H>(cw, nr, sX, acc); }

      if (lane < M) {
        __stcg(p.ob + lane * H + n, __float2bfloat16_rn(acc[0][lane]));
      }
    }
    {  // continue prefetching S3 rows beyond the early window while waiting at the barrier
      const int skip = DD_S3_EARLY_PF * 1024 / (H * 3 / 2);   // compressed row = 1.5 KB
      PFR(cwl(l, CG), L.wg, rF0 + skip < rF1 ? rF0 + skip : rF1, rF1, H, 32 * 1024);
      PFR(cwl(l, CU), L.wu, rF0 + skip < rF1 ? rF0 + skip : rF1, rF1, H, 32 * 1024);
    }
    grid_sync(p.bar, epoch);

    // S3: RMSNorm + gate/up + SiLU*up
    if (DD_PF_S4_AT_S3 > 0) PFR(cwl(l, CD), L.wd, rH0, rH1, FF, DD_PF_S4_AT_S3 * 1024);
    load_norm_sum<M>(p.xres, p.ob, L.ln2, sX, p.eps);
    for (int n = nF; n < rF1; n += NW) {
      float acc[2][M] = {};

      { const CW* cw[2] = {&cwl(l, CG), &cwl(l, CU)}; const int nr[2] = {n, n}; dot_rows_c<M, 2, H>(cw, nr, sX, acc); }

      if (lane < M) {
        const float g = rbf(acc[0][lane]), u = rbf(acc[1][lane]);
        const float sg = g / (1.f + expf(-g));
        const float a = sg * u;
        __stcg(p.act + lane * FF + n, __float2bfloat16_rn(a));
      }
    }
    PFR(cwl(l, CD), L.wd, rH0, rH1, FF, 64 * 1024);
    grid_sync(p.bar, epoch);

    // S4: down-proj + residual
    if (DD_PF_S1_AT_S4 > 0 && l + 1 < NL) {
      const int q0 = (rQ0 >> 6) * HD, q1 = ((rQ1 >> 6) + 1) * HD < QD ? ((rQ1 >> 6) + 1) * HD : QD;
      PFR(cwl(l + 1, CQ), p.L[l + 1].wq, q0, q1, H, DD_PF_S1_AT_S4 * 1024);
    }
    if constexpr (M <= SU<B>::ROWS) {
    for (int i = threadIdx.x; i < M * FF / 8; i += NT)
      reinterpret_cast<uint4*>(sU)[i] = __ldcg(reinterpret_cast<const uint4*>(p.act) + i);
    __syncthreads();
    }
    // when the activations do not fit (B=4 prefill: 8 rows), stream them through sU in two K halves; the per-row
    // accumulation order is unchanged, so results are bitwise identical to the one-pass path (batch invariance)
    constexpr int KH = FF / 2;
    for (int it = 0; it < (rH1 - rH0 + NW - 1) / NW; ++it) {
      const int n = nH + it * NW;
      const bool valid = n < rH1;
      float acc[1][M] = {};
      if constexpr (M <= SU<B>::ROWS) {

        if (valid) { const CW* cw[1] = {&cwl(l, CD)}; const int nr[1] = {n}; dot_rows_c<M, 1, FF>(cw, nr, sU, acc); }

      } else {
        const CW* cw4[1] = {&cwl(l, CD)};
        const int nr4[1] = {valid ? n : rH0};
        RowC st4[1];
        rowc_init(st4[0], *cw4[0], nr4[0]);
        rowc_start(st4[0]);
#pragma unroll 1
        for (int half = 0; half < 2; ++half) {
          __syncthreads();
          for (int i = threadIdx.x; i < M * KH / 8; i += NT) {
            const int m = i / (KH / 8), c = i % (KH / 8);
            reinterpret_cast<uint4*>(sU)[i] = __ldcg(reinterpret_cast<const uint4*>(p.act + (size_t)m * FF + half * KH) + c);
          }
          __syncthreads();

          if (valid) dot_rows_part_c<M, 1, FF, KH>(cw4, nr4, st4, sU, acc, half * KH);

        }
        if (valid) reduce_acc<M, 1>(acc);
      }
      if (valid && lane < M) {
        const float x = bf2f(__ldcg(p.xres + lane * H + n));
        const float o = bf2f(__ldcg(p.ob + lane * H + n));
        __stcg(p.xres + lane * H + n, __float2bfloat16_rn((x + o) + rbf(acc[0][lane])));
      }
    }
    if (l + 1 < NL) {
      // the next layer's first q|k|v rows were already prefetched into L2 during S4
    } else {
      PFR(c_cw[NL * 7 + 1 + step], p.head + (size_t)step * V * H, rV0 + NW, rV1, H, 64 * 1024);
    }
    grid_sync(p.bar, epoch);
  }

  // S5: final norm + head
  load_norm<M>(p.xres, p.normf, sX, p.eps);
  for (int n = nV; n < rV1; n += NW) {
    float acc[1][M] = {};

    { const CW* cw[1] = {&c_cw[NL * 7 + 1 + step]}; const int nr[1] = {n}; dot_rows_c<M, 1, H>(cw, nr, sX, acc); }

    if (lane < B) {
      const float v = acc[0][lane * T + (T - 1)];
      __stcg(p.logits + lane * V + n, v);
      if (p.dbg) p.dbg[((size_t)step * B + lane) * V + n] = v;
    }
  }
}

// ------------------------------------------------------------------ fused official sampler (one block per half-row)
// Mirrors DepthDecoderGraph._cfg_sample: CFG -> mask codec-invalid ids -> /temperature -> top-k (k<=1024, 0->1024)
// -> top-p (softmax of sorted, cumsum, shift-right remove) -> softmax -> argmax(probs / q) ; greedy = argmax(cfg)
template <int B>
__device__ __forceinline__ void sample_row(const P2& p, const int step, const int h, unsigned char* smem) {
  constexpr int HALF = B >= 2 ? B / 2 : 1;
  constexpr int NS = 2048;  // sort the first 2048 ids; ids >= 2048 are always masked to -inf
  float* key = reinterpret_cast<float*>(smem);
  int* idx = reinterpret_cast<int*>(smem + NS * 4);
  __shared__ float s_red[NW];
  __shared__ int s_redi[NW];
  __shared__ float s_bc[4];
  __shared__ int s_bci[4];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float* lc = p.logits + (size_t)h * V;
  const float* lu = p.logits + (size_t)(h + HALF) * V;
  const float gs = p.gs[h], temp = p.temp[h];
  // 1) cfg, scaled keys, greedy argmax(cfg) (first index on ties)
  float gbest = -INFINITY;
  int gidx = 0x7fffffff;
  for (int i = tid; i < NS; i += NT) {
    const float c = __ldcg(lc + i);
    float cfg;
    if (B >= 2) {
      const float u = __ldcg(lu + i);
      cfg = __fadd_rn(u, __fmul_rn(gs, __fsub_rn(c, u)));
    } else {
      cfg = c;
    }
    key[i] = __fdiv_rn(cfg, temp);
    idx[i] = i;
    if (cfg > gbest || (cfg == gbest && i < gidx)) { gbest = cfg; gidx = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, gbest, o);
    const int oi = __shfl_xor_sync(0xffffffffu, gidx, o);
    if (ov > gbest || (ov == gbest && oi < gidx)) { gbest = ov; gidx = oi; }
  }
  if (lane == 0) { s_red[warp] = gbest; s_redi[warp] = gidx; }
  __syncthreads();
  if (tid == 0) {
    float bv = s_red[0]; int bi = s_redi[0];
    for (int w = 1; w < NW; ++w) if (s_red[w] > bv || (s_red[w] == bv && s_redi[w] < bi)) { bv = s_red[w]; bi = s_redi[w]; }
    s_bci[0] = bi;
  }
  const int64_t tk0 = p.topk[h];
  const int keff0 = tk0 > 0 ? (tk0 < 1024 ? (int)tk0 : 1024) : 1024;
#ifndef DD_FAST_TOPK
#define DD_FAST_TOPK 1
#endif
  if (DD_FAST_TOPK && keff0 <= 64) {
    // ---- warp tournament: total order (key desc, idx asc) identical to the bitonic path ----
    __syncthreads();  // key/idx written
    float* lk = reinterpret_cast<float*>(smem + NS * 8);          // [16][64] keys
    int* li = reinterpret_cast<int*>(smem + NS * 8 + 16 * 64 * 4);  // [16][64] idx
    auto before = [](float ka, int ia, float kb, int ib) { return ka > kb || (ka == kb && ia < ib); };
    for (int wv = warp; wv < NS / 128; wv += NW) {
      float kv[4]; int iv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) { kv[i] = key[wv * 128 + lane * 4 + i]; iv[i] = idx[wv * 128 + lane * 4 + i]; }
#pragma unroll
      for (int k = 2; k <= 128; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
          float nk[4]; int ni[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int e = lane * 4 + i;
            float pk; int pi;
            if (j >= 4) { pk = __shfl_xor_sync(0xffffffffu, kv[i], j >> 2); pi = __shfl_xor_sync(0xffffffffu, iv[i], j >> 2); }
            else { pk = kv[i ^ j]; pi = iv[i ^ j]; }
            const bool normal = (e & k) == 0, lower = (e & j) == 0;
            const bool mine_first = before(kv[i], iv[i], pk, pi);
            const bool take_first = (lower == normal);
            const bool keep_mine = take_first ? mine_first : !mine_first;
            nk[i] = keep_mine ? kv[i] : pk; ni[i] = keep_mine ? iv[i] : pi;
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) { kv[i] = nk[i]; iv[i] = ni[i]; }
        }
      }
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { lk[wv * 64 + lane * 4 + i] = kv[i]; li[wv * 64 + lane * 4 + i] = iv[i]; }
      }
    }
    // merge rounds: lists 2w, 2w+1 -> w  (top-64 of the union, sorted)
    for (int nl = NS / 128; nl > 1; nl >>= 1) {
      __syncthreads();
      float mk2[2]; int mi2[2];
      const int w = warp;
      if (w < nl / 2) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int e = lane * 2 + i;
          const float ka = lk[(2 * w) * 64 + e], kb = lk[(2 * w + 1) * 64 + 63 - e];
          const int ia = li[(2 * w) * 64 + e], ib = li[(2 * w + 1) * 64 + 63 - e];
          const bool af = before(ka, ia, kb, ib);
          mk2[i] = af ? ka : kb; mi2[i] = af ? ia : ib;
        }
        // bitonic merge of 64 (descending)
#pragma unroll
        for (int j = 32; j > 0; j >>= 1) {
          float nk[2]; int ni[2];
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int e = lane * 2 + i;
            float pk; int pi;
            if (j >= 2) { pk = __shfl_xor_sync(0xffffffffu, mk2[i], j >> 1); pi = __shfl_xor_sync(0xffffffffu, mi2[i], j >> 1); }
            else { pk = mk2[i ^ 1]; pi = mi2[i ^ 1]; }
            const bool lower = (e & j) == 0;
            const bool mine_first = before(mk2[i], mi2[i], pk, pi);
            const bool keep_mine = lower ? mine_first : !mine_first;
            nk[i] = keep_mine ? mk2[i] : pk; ni[i] = keep_mine ? mi2[i] : pi;
          }
#pragma unroll
          for (int i = 0; i < 2; ++i) { mk2[i] = nk[i]; mi2[i] = ni[i]; }
        }
      }
      __syncthreads();
      if (w < nl / 2) {
#pragma unroll
        for (int i = 0; i < 2; ++i) { lk[w * 64 + lane * 2 + i] = mk2[i]; li[w * 64 + lane * 2 + i] = mi2[i]; }
      }
    }
    __syncthreads();
    // ---- sampling math on the sorted top-64 (warp 0); same pairing/tree order as the block path ----
    if (warp == 0) {
      const float topp = p.topp[h];
      const float mx = lk[0];
      const int r0 = 2 * lane, r1 = 2 * lane + 1;
      const float e0 = r0 < keff0 ? expf(lk[r0] - mx) : 0.f;
      const float e1 = r1 < keff0 ? expf(lk[r1] - mx) : 0.f;
      const float S = wsum(e0 + e1);
      const float p0 = r0 < keff0 ? __fdiv_rn(e0, S) : 0.f, p1 = r1 < keff0 ? __fdiv_rn(e1, S) : 0.f;
      const float tsum = p0 + p1;
      float incl = tsum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      float run = incl - tsum;
      const bool k0 = r0 < keff0 && (r0 == 0 || !(run > topp));
      run += p0;
      const bool k1 = r1 < keff0 && !(run > topp);
      const float S2 = wsum((k0 ? e0 : 0.f) + (k1 ? e1 : 0.f));
      const float* qrow = p.q + ((size_t)step * HALF + h) * V;
      float best = -1.f; int bidx = 0x7fffffff;
      if (k0) { const int id = li[r0]; const float sc = __fdiv_rn(__fdiv_rn(e0, S2), qrow[id]); if (sc > best || (sc == best && id < bidx)) { best = sc; bidx = id; } }
      if (k1) { const int id = li[r1]; const float sc = __fdiv_rn(__fdiv_rn(e1, S2), qrow[id]); if (sc > best || (sc == best && id < bidx)) { best = sc; bidx = id; } }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
      }
      if (lane == 0) {
        int tokv = p.dosample[h] ? bidx : s_bci[0];
        if (p.forced) tokv = (int)p.forced[(size_t)step * B + h];
        tokv = tokv < 0 ? 0 : (tokv > V - 1 ? V - 1 : tokv);
        __stcg(reinterpret_cast<long long*>(p.tokbuf) + h, (long long)tokv);
        p.outtok[(size_t)h * NCB + step] = tokv;
        if (B >= 2) {
          __stcg(reinterpret_cast<long long*>(p.tokbuf) + h + HALF, (long long)tokv);
          p.outtok[(size_t)(h + HALF) * NCB + step] = tokv;
        }
      }
    }
    return;
  }
  // 2) bitonic sort, descending by key, ascending by idx on ties
  for (int k = 2; k <= NS; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      for (int t = tid; t < NS / 2; t += NT) {
        const int i = 2 * t - (t & (j - 1));
        const int l2 = i + j;
        const float ki = key[i], kl = key[l2];
        const int ii = idx[i], il = idx[l2];
        const bool i_first = (ki > kl) || (ki == kl && ii < il);
        const bool up = (i & k) == 0;
        if (up != i_first) { key[i] = kl; key[l2] = ki; idx[i] = il; idx[l2] = ii; }
      }
    }
  }
  __syncthreads();
  const int64_t tk = p.topk[h];
  const int keff = tk > 0 ? (tk < 1024 ? (int)tk : 1024) : 1024;
  const float topp = p.topp[h];
  const float mx = key[0];
  // 3) top-p on softmax(sorted top-k): inclusive cumsum of probs over ranks < keff.
  //    thread t owns ranks [t*RPT, t*RPT+RPT), RPT*NT >= 1024 >= keff
  constexpr int RPT = (1024 + NT - 1) / NT;
  float e[RPT];
  float sv = 0.f;
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int r = tid * RPT + i;
    e[i] = r < keff ? expf(key[r] - mx) : 0.f;
    sv += e[i];
  }
  sv = wsum(sv);
  if (lane == 0) s_red[warp] = sv;
  __syncthreads();
  if (tid == 0) { float S = 0.f; for (int w = 0; w < NW; ++w) S += s_red[w]; s_bc[0] = S; }
  __syncthreads();
  const float S = s_bc[0];
  float pr[RPT];
  float tsum = 0.f;
#pragma unroll
  for (int i = 0; i < RPT; ++i) { pr[i] = (tid * RPT + i) < keff ? __fdiv_rn(e[i], S) : 0.f; tsum += pr[i]; }
  float incl = tsum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  __syncthreads();
  if (lane == 31) s_red[warp] = incl;
  __syncthreads();
  float woff = 0.f;
  for (int w = 0; w < warp; ++w) woff += s_red[w];
  float run = woff + incl - tsum;  // cumsum through rank tid*RPT - 1
  // 4) final softmax over the kept set (rank 0 always kept), argmax(prob / q)
  bool keep[RPT];
  float f2 = 0.f;
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const int r = tid * RPT + i;
    keep[i] = r < keff && (r == 0 || !(run > topp));  // remove[r] = cum[r-1] > top_p
    run += pr[i];
    if (keep[i]) f2 += e[i];
  }
  float s2 = wsum(f2);
  __syncthreads();
  if (lane == 0) s_red[warp] = s2;
  __syncthreads();
  if (tid == 0) { float S2 = 0.f; for (int w = 0; w < NW; ++w) S2 += s_red[w]; s_bc[1] = S2; }
  __syncthreads();
  const float S2 = s_bc[1];
  const float* qrow = p.q + ((size_t)step * HALF + h) * V;
  float best = -1.f;
  int bidx = 0x7fffffff;
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    if (keep[i]) {
      const int r = tid * RPT + i;
      const int id = idx[r];
      const float sc = __fdiv_rn(__fdiv_rn(e[i], S2), qrow[id]);
      if (sc > best || (sc == best && id < bidx)) { best = sc; bidx = id; }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
    if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
  }
  __syncthreads();
  if (lane == 0) { s_red[warp] = best; s_redi[warp] = bidx; }
  __syncthreads();
  if (tid == 0) {
    float bv = s_red[0]; int bi = s_redi[0];
    for (int w = 1; w < NW; ++w) if (s_red[w] > bv || (s_red[w] == bv && s_redi[w] < bi)) { bv = s_red[w]; bi = s_redi[w]; }
    int tokv = p.dosample[h] ? bi : s_bci[0];
    if (p.forced) tokv = (int)p.forced[(size_t)step * B + h];
    tokv = tokv < 0 ? 0 : (tokv > V - 1 ? V - 1 : tokv);
    __stcg(reinterpret_cast<long long*>(p.tokbuf) + h, (long long)tokv);
    p.outtok[(size_t)h * NCB + step] = tokv;
    if (B >= 2) {
      __stcg(reinterpret_cast<long long*>(p.tokbuf) + h + HALF, (long long)tokv);
      p.outtok[(size_t)(h + HALF) * NCB + step] = tokv;
    }
  }
}

template <int B>
__global__ void __launch_bounds__(NT, 1) frame_kernel(const P2 p) {
  constexpr int HALF = B >= 2 ? B / 2 : 1;
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* sU = reinterpret_cast<bf16*>(smem);
  bf16* sX = sU + SU<B>::ELEMS;
  unsigned epoch = 0;
  const int G = gridDim.x, bid = blockIdx.x;
  const int rH0 = H * bid / G, rH1 = H * (bid + 1) / G;
 
#pragma unroll 1
  for (int s = 0; s < NCB; ++s) {
    if (s == 0) forward_step<B, 2>(p, 0, epoch, sU, sX);
    else forward_step<B, 1>(p, s, epoch, sU, sX);
    grid_sync(p.bar, epoch);
    if (bid < HALF) sample_row<B>(p, s, bid, smem);
    else if (s + 1 < NCB) {
#ifndef DD_STEP_TABLE
      PFR(c_cw[NL * 7], p.proj, rH0, rH1, AE, 64 * 1024);
      const int rQ0 = 768 * bid / G, rQ1 = 768 * (bid + 1) / G;
      const int q0 = (rQ0 >> 6) * HD, q1 = ((rQ1 >> 6) + 1) * HD < QD ? ((rQ1 >> 6) + 1) * HD : QD;
      PFR(cwl(0, CQ), p.L[0].wq, q0, q1, H, 48 * 1024);
#endif
    }
    if (s + 1 < NCB) grid_sync(p.bar, epoch);
  }
}

static P2 g2;
static bool g2_init = false;
static int g2_grid = 0;
static int g2_gridB[5] = {0, 0, 0, 0, 0};  // per-B grid override (dd2_set_grid)
static bool g_ln12_ok = false;

template <int B>
static void launch2(const P2& p, cudaStream_t st) {
  const size_t smem = (size_t)SU<B>::ELEMS * 2 + (size_t)2 * B * H * 2;
  auto kern = frame_kernel<B>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, NT, smem);
  TORCH_CHECK(nb >= 1, "frame kernel not resident");
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeCooperative;
  at[0].val.cooperative = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(g2_gridB[B] > 0 ? g2_gridB[B] : g2_grid);
  cfg.blockDim = dim3(NT);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, p);
  TORCH_CHECK(e == cudaSuccess, "frame kernel launch failed: ", cudaGetErrorString(e));
}


#ifdef DD_STEP_TABLE
// Table build: block b handles table rows t = 14b .. 14b+13 (id = V + t). Same device functions and the same
// per-row accumulation order as the frame kernel's stage E and layer-0 S1 (per-row results do not depend on M).
constexpr int TAB_MB = 14;
static_assert(TAB_ROWS % TAB_MB == 0, "table rows must split evenly");
__global__ void __launch_bounds__(NT, 1) tab_build_k(const P2 p, bf16* ptab, bf16* ttab) {
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* sU = reinterpret_cast<bf16*>(smem);  // [MB][AE]
  bf16* sX = sU + TAB_MB * AE;              // [MB][H]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = blockIdx.x * TAB_MB;
  for (int m = warp; m < TAB_MB; m += NW) {
    const uint4* src = reinterpret_cast<const uint4*>(p.embed + (size_t)(V + t0 + m) * AE);
#pragma unroll
    for (int i = 0; i < AE / 256; ++i) reinterpret_cast<uint4*>(sU + m * AE)[i * 32 + lane] = __ldcg(src + i * 32 + lane);
  }
  __syncthreads();
  for (int n = warp; n < H; n += NW) {
    float acc[1][TAB_MB] = {};
    { const CW* cw[1] = {&c_cw[NL * 7]}; const int nr[1] = {n}; dot_rows_c<TAB_MB, 1, AE>(cw, nr, sU, acc); }
#pragma unroll
    for (int m = 0; m < TAB_MB; ++m)
      if (lane == m) __stcg(ptab + (size_t)(t0 + m) * H + n, __float2bfloat16_rn(acc[0][m]));
  }
  __threadfence();
  __syncthreads();
  load_norm<TAB_MB>(ptab + (size_t)t0 * H, p.L[0].ln1, sX, p.eps);
  for (int u = warp; u < 768; u += NW) {
    const bf16* W; int kind, hh, d; qkv_unit(p.L[0], u, W, kind, hh, d);
    const int row = hh * HD + d;
    float acc[2][TAB_MB] = {};
    { const CW* cw[2] = {&cwl(0, kind), &cwl(0, kind)}; const int nr[2] = {row, row + 64}; dot_rows_c<TAB_MB, 2, H>(cw, nr, sX, acc); }
    const int base = kind == 0 ? 0 : (kind == 1 ? QD : QD + NKV * HD);
#pragma unroll
    for (int m = 0; m < TAB_MB; ++m)
      if (lane == m) {
        ttab[(size_t)(t0 + m) * 1536 + base + row] = __float2bfloat16_rn(acc[0][m]);
        ttab[(size_t)(t0 + m) * 1536 + base + row + 64] = __float2bfloat16_rn(acc[1][m]);
      }
  }
}
#endif
}  // namespace dd

void dd2_init(std::vector<at::Tensor> w, std::vector<at::Tensor> bufs, double eps, double scaling, int64_t grid) {
  using namespace dd;
  TORCH_CHECK(w.size() == NL * 9 + 6, "bad weight count");
  auto bp = [](const at::Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16);
    return reinterpret_cast<const bf16*>(t.data_ptr());
  };
  for (int l = 0; l < NL; ++l) {
    LayerW& L = g2.L[l];
    L.ln1 = bp(w[l * 9 + 0]); L.wq = bp(w[l * 9 + 1]); L.wk = bp(w[l * 9 + 2]); L.wv = bp(w[l * 9 + 3]);
    L.wo = bp(w[l * 9 + 4]); L.ln2 = bp(w[l * 9 + 5]); L.wg = bp(w[l * 9 + 6]); L.wu = bp(w[l * 9 + 7]);
    L.wd = bp(w[l * 9 + 8]);
  }
  const int o = NL * 9;
  g2.embed = bp(w[o]); g2.proj = bp(w[o + 1]); g2.normf = bp(w[o + 2]); g2.head = bp(w[o + 3]);
  g2.cos_tab = bp(w[o + 4]); g2.sin_tab = bp(w[o + 5]);
  g2.kc = reinterpret_cast<bf16*>(bufs[0].data_ptr());
  g2.vc = reinterpret_cast<bf16*>(bufs[1].data_ptr());
  g2.xres = reinterpret_cast<bf16*>(bufs[2].data_ptr());
  g2.qb = reinterpret_cast<bf16*>(bufs[3].data_ptr());
  g2.act = reinterpret_cast<bf16*>(bufs[4].data_ptr());
  g2.bar = reinterpret_cast<unsigned*>(bufs[5].data_ptr());
  g2.logits = bufs[6].data_ptr<float>();
  g2.ob = reinterpret_cast<bf16*>(bufs[7].data_ptr());
  g2.forced = nullptr;
  g2.eps = (float)eps;
  g2.scaling = (float)scaling;
  g2_grid = (int)grid;
#ifdef DD_PF_EVICT_LAST
  {  // evict_last prefetches only help with a zero persisting-L2 set-aside (a torch process starts with a non-zero one)
    cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    TORCH_CHECK(e == cudaSuccess, "cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0): ", cudaGetErrorString(e));
    size_t v = 1;
    e = cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize);
    TORCH_CHECK(e == cudaSuccess && v == 0, "persisting L2 limit is ", (long long)v, " after setting 0");
  }
#endif
  g2_init = true;
}

// cw_* lists in order: per layer {q,k,v,o,gate,up,down} x 12, projector, head[0..14]. data: uint8 [N*K*3/2],
// meta: int32 [N,4] inline row records, ovf: int32 overflow patches, luts: int64 [n,2] (lut0, lut1).
void dd2_init_ln12(std::vector<at::Tensor> data, std::vector<at::Tensor> meta, std::vector<at::Tensor> ovf, at::Tensor luts) {
  using namespace dd;
  TORCH_CHECK((int)data.size() == DD_CW_N && (int)meta.size() == DD_CW_N && (int)ovf.size() == DD_CW_N, "bad descriptor count");
  auto lc = luts.cpu();
  TORCH_CHECK(lc.scalar_type() == at::kLong && lc.size(0) == DD_CW_N && lc.size(1) == 2);
  CW h[DD_CW_N];
  for (int i = 0; i < DD_CW_N; ++i) {
    TORCH_CHECK(data[i].is_cuda() && data[i].is_contiguous() && meta[i].is_cuda() && meta[i].is_contiguous() && ovf[i].is_cuda());
    h[i].data = reinterpret_cast<const uint4*>(data[i].data_ptr());
    h[i].meta = reinterpret_cast<const uint4*>(meta[i].data_ptr());
    h[i].ovf = reinterpret_cast<const uint32_t*>(ovf[i].data_ptr());
    h[i].lut0 = (uint32_t)lc[i][0].item<int64_t>();
    h[i].lut1 = (uint32_t)lc[i][1].item<int64_t>();
  }
  cudaError_t e = cudaMemcpyToSymbol(c_cw, h, sizeof(h));
  TORCH_CHECK(e == cudaSuccess, "c_cw upload: ", cudaGetErrorString(e));
  g_ln12_ok = true;
}
// One full frame (15 codebooks). All tensors are the official DepthDecoderGraph buffers.
void dd2_frame(const at::Tensor& first_cb, const at::Tensor& bh, at::Tensor& tokbuf, at::Tensor& outtok,
               const at::Tensor& temp, const at::Tensor& topk, const at::Tensor& topp, const at::Tensor& dosample,
               const at::Tensor& gs, const at::Tensor& q, c10::optional<at::Tensor> dbg,
               c10::optional<at::Tensor> forced) {
  using namespace dd;
  TORCH_CHECK(g2_init, "dd2_init first");
  TORCH_CHECK(dd::g_ln12_ok, "dd2_init_ln12 first");
#ifdef DD_STEP_TABLE
  TORCH_CHECK(dd::g2.ptab != nullptr && dd::g2.ttab != nullptr, "step-table build: dd2_tab_build() or dd2_tab_set() first");
#endif
  TORCH_CHECK(first_cb.scalar_type() == at::kLong && tokbuf.scalar_type() == at::kLong && outtok.scalar_type() == at::kLong);
  TORCH_CHECK(topk.scalar_type() == at::kLong && dosample.scalar_type() == at::kLong);
  TORCH_CHECK(temp.scalar_type() == at::kFloat && topp.scalar_type() == at::kFloat && gs.scalar_type() == at::kFloat && q.scalar_type() == at::kFloat);
  TORCH_CHECK(outtok.is_contiguous() && outtok.size(1) == NCB && q.is_contiguous());
  P2 p = g2;
  p.first_cb = first_cb.data_ptr<int64_t>();
  p.bh = reinterpret_cast<const bf16*>(bh.data_ptr());
  p.tokbuf = tokbuf.data_ptr<int64_t>();
  p.outtok = outtok.data_ptr<int64_t>();
  p.temp = temp.data_ptr<float>();
  p.topk = topk.data_ptr<int64_t>();
  p.topp = topp.data_ptr<float>();
  p.dosample = dosample.data_ptr<int64_t>();
  p.gs = gs.data_ptr<float>();
  p.q = q.data_ptr<float>();
  p.dbg = dbg.has_value() ? dbg->data_ptr<float>() : nullptr;
  p.forced = forced.has_value() ? forced->data_ptr<int64_t>() : nullptr;
  const int B = (int)bh.size(0);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(p.bar, 0, sizeof(unsigned), st);
  if (B == 1) launch2<1>(p, st);
  else if (B == 2) launch2<2>(p, st);
  else if (B == 4) launch2<4>(p, st);
  else TORCH_CHECK(false, "B must be 1, 2 or 4");
}


// ---------------------------------------------------------------- grid, info and step-table helpers
void dd2_set_grid(int64_t B, int64_t grid) {
  using namespace dd;
  TORCH_CHECK(B == 1 || B == 2 || B == 4, "B must be 1, 2 or 4");
  int dev = 0, nsm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  TORCH_CHECK(grid >= 1 && grid <= nsm, "grid must be in [1, #SMs] (one resident block per SM)");
  g2_gridB[B] = (int)grid;
}
int64_t dd2_l2_persist_limit() {
  size_t v = 0;
  cudaError_t e = cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize);
  TORCH_CHECK(e == cudaSuccess, "cudaDeviceGetLimit: ", cudaGetErrorString(e));
  return (int64_t)v;
}
std::string dd2_info() {
  using namespace dd;
  std::string s = "{\"NL\":" + std::to_string(NL);
  s += ",\"ln12\":1";
#ifdef DD_PF_EVICT_LAST
  s += ",\"pf_last\":1";
#else
  s += ",\"pf_last\":0";
#endif
  s += ",\"kvfirst\":1";
#ifdef DD_STEP_TABLE
  s += ",\"tab\":1,\"tab_set\":" + std::to_string(g2.ptab != nullptr ? 1 : 0);
#else
  s += ",\"tab\":0";
#endif
  s += ",\"grid\":" + std::to_string(g2_grid) + ",\"grid_B1\":" + std::to_string(g2_gridB[1]) + ",\"grid_B2\":" +
       std::to_string(g2_gridB[2]) + ",\"grid_B4\":" + std::to_string(g2_gridB[4]) +
       ",\"persisting_l2_limit\":" + std::to_string(dd2_l2_persist_limit()) + "}";
  return s;
}
#ifdef DD_STEP_TABLE
static at::Tensor g_ptab, g_ttab;
std::vector<at::Tensor> dd2_tab_build() {
  using namespace dd;
  TORCH_CHECK(g2_init, "dd2_init first");
  TORCH_CHECK(g_ln12_ok, "dd2_init_ln12 before dd2_tab_build");
  auto opt = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  at::Tensor pt = at::empty({TAB_ROWS, H}, opt), tt = at::empty({TAB_ROWS, 1536}, opt);
  const size_t smem = (size_t)TAB_MB * (AE + H) * 2;
  cudaError_t e = cudaFuncSetAttribute(tab_build_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  TORCH_CHECK(e == cudaSuccess, "tab_build smem attr: ", cudaGetErrorString(e));
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  tab_build_k<<<TAB_ROWS / TAB_MB, NT, smem, st>>>(g2, reinterpret_cast<bf16*>(pt.data_ptr()), reinterpret_cast<bf16*>(tt.data_ptr()));
  e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "tab_build launch: ", cudaGetErrorString(e));
  e = cudaStreamSynchronize(st);
  TORCH_CHECK(e == cudaSuccess, "tab_build: ", cudaGetErrorString(e));
  g_ptab = pt; g_ttab = tt;
  g2.ptab = reinterpret_cast<const bf16*>(pt.data_ptr());
  g2.ttab = reinterpret_cast<const bf16*>(tt.data_ptr());
  return {pt, tt};
}
void dd2_tab_set(at::Tensor pt, at::Tensor tt) {
  using namespace dd;
  TORCH_CHECK(pt.is_cuda() && pt.is_contiguous() && pt.scalar_type() == at::kBFloat16 && pt.size(0) == TAB_ROWS && pt.size(1) == H);
  TORCH_CHECK(tt.is_cuda() && tt.is_contiguous() && tt.scalar_type() == at::kBFloat16 && tt.size(0) == TAB_ROWS && tt.size(1) == 1536);
  g_ptab = pt; g_ttab = tt;
  g2.ptab = reinterpret_cast<const bf16*>(pt.data_ptr());
  g2.ttab = reinterpret_cast<const bf16*>(tt.data_ptr());
}
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dd2_init", &dd2_init);
  m.def("dd2_init_ln12", &dd2_init_ln12);
  m.def("dd2_frame", &dd2_frame);
  m.def("dd2_set_grid", &dd2_set_grid);
  m.def("dd2_l2_persist_limit", &dd2_l2_persist_limit);
  m.def("dd2_info", &dd2_info);
#ifdef DD_STEP_TABLE
  m.def("dd2_tab_build", &dd2_tab_build);
  m.def("dd2_tab_set", &dd2_tab_set);
#endif
}
