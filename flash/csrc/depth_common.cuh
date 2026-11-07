// Breeze-TTS-2-flash: shared device helpers for the depth-decoder megakernel (opens namespace dd; the includer
// closes it).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>
namespace dd {
using bf16 = __nv_bfloat16;
constexpr int H = 1024, AE = 2048, NH = 8, NKV = 2, HD = 128, QD = NH * HD, FF = 8192, NL = 12, V = 2051,
              NCB = 15, MAXP = 16;
#ifndef DD_NT
#define DD_NT 512
#endif
constexpr int NT = DD_NT, NW = NT / 32;

struct LayerW {
  const bf16 *ln1, *wq, *wk, *wv, *wo, *ln2, *wg, *wu, *wd;
};


__device__ __forceinline__ uint4 ld_w(const void* p) {  // weights: streaming, no L1 allocation
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
#ifdef DD_PF_EVICT_LAST
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p)); }
#else
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
#endif
__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void cvt8(const uint4& v, float* f) {
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned& epoch) {
  __syncthreads();
  ++epoch;
  if (threadIdx.x == 0) {
    const unsigned target = epoch * gridDim.x;
    __threadfence();
    atomicAdd(bar, 1u);
    unsigned v;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
    } while (v < target);
  }
  __syncthreads();
}

// Prefetch rows [r0, r1) of a row-major [*, K] bf16 matrix into L2, at most max_bytes, spread over the block.
__device__ __forceinline__ void prefetch_rows(const bf16* W, int r0, int r1, int K, int max_bytes) {
  const size_t base = (size_t)r0 * K * 2;
  size_t bytes = (size_t)(r1 - r0) * K * 2;
  if (bytes > (size_t)max_bytes) bytes = max_bytes;
  const char* c = reinterpret_cast<const char*>(W) + base;
  for (size_t off = (size_t)threadIdx.x * 128; off < bytes; off += (size_t)NT * 128) pf_l2(c + off);
}

// Register preload of the first chunk group of a unit (issued BEFORE a grid barrier: weights never depend
// on activations, so their DRAM latency overlaps the barrier wait / attention / norm of the next stage).
template <int R, int K>
struct Pre {
  static constexpr int U = (K / 256) < 4 ? (K / 256) : 4;
  uint4 w[U][R];
};
template <int R, int K>
__device__ __forceinline__ void preload(const bf16* const* wr, Pre<R, K>& pr) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int u = 0; u < Pre<R, K>::U; ++u)
#pragma unroll
    for (int r = 0; r < R; ++r) pr.w[u][r] = ld_w(wr[r] + u * 256 + lane * 8);
}

// acc[r][m] += <W row r, xs row m> over chunk group [c0, c0+U) with weights already in registers
template <int M, int R, int U>
__device__ __forceinline__ void fma_group(const uint4 (&w)[U][R], const bf16* __restrict__ xs, int K, int c0,
                                          float (&acc)[R][M]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int k = (c0 + u) * 256 + lane * 8;
    float wf[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) cvt8(w[u][r], wf[r]);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float xf[8];
      cvt8(*reinterpret_cast<const uint4*>(xs + m * K + k), xf);
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][m] = fmaf(wf[r][j], xf[j], acc[r][m]);
    }
  }
}

template <int M, int R, int K>
__device__ __forceinline__ void dot_rows_from(const bf16* const* wr, const bf16* __restrict__ xs, float (&acc)[R][M],
                                              int cstart) {
  const int lane = threadIdx.x & 31;
  constexpr int CH = K / 256;
  constexpr int U = CH < 4 ? CH : 4;
#pragma unroll 1
  for (int c0 = cstart; c0 < CH; c0 += U) {
    uint4 w[U][R];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r) w[u][r] = ld_w(wr[r] + (c0 + u) * 256 + lane * 8);
    fma_group<M, R, U>(w, xs, K, c0, acc);
  }
}

// Partial dot over weight columns [k0, k0+KR) with xs holding only those columns ([M][KR]); no reduction.
template <int M, int R, int K, int KR>
__device__ __forceinline__ void dot_rows_part(const bf16* const* wr, const bf16* __restrict__ xs, float (&acc)[R][M],
                                              int k0) {
  const int lane = threadIdx.x & 31;
  constexpr int U = 4;
  static_assert(KR % (256 * U) == 0, "KR must be a multiple of 1024");
#pragma unroll 1
  for (int c0 = k0 / 256; c0 < (k0 + KR) / 256; c0 += U) {
    uint4 w[U][R];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r) w[u][r] = ld_w(wr[r] + (c0 + u) * 256 + lane * 8);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int kx = (c0 + u) * 256 - k0 + lane * 8;
      float wf[R][8];
#pragma unroll
      for (int r = 0; r < R; ++r) cvt8(w[u][r], wf[r]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[8];
        cvt8(*reinterpret_cast<const uint4*>(xs + m * KR + kx), xf);
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[r][m] = fmaf(wf[r][j], xf[j], acc[r][m]);
      }
    }
  }
}

template <int M, int R>
__device__ __forceinline__ void reduce_acc(float (&acc)[R][M]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = wsum(acc[r][m]);
}

template <int M, int R, int K>
__device__ __forceinline__ void dot_rows(const bf16* const* wr, const bf16* __restrict__ xs, float (&acc)[R][M]) {
  dot_rows_from<M, R, K>(wr, xs, acc, 0);
  reduce_acc<M, R>(acc);
}
template <int M, int R, int K>
__device__ __forceinline__ void dot_rows_pre(const bf16* const* wr, const bf16* __restrict__ xs, float (&acc)[R][M],
                                             const Pre<R, K>& pre) {
  fma_group<M, R, Pre<R, K>::U>(pre.w, xs, K, 0, acc);
  dot_rows_from<M, R, K>(wr, xs, acc, Pre<R, K>::U);
  reduce_acc<M, R>(acc);
}

// Load M rows of the residual stream and apply BreezeRMSNorm (fp32 stats, .to(bf16), weight * x).
// Post-attention RMSNorm with inductor semantics: x_mid = x + o computed in fp32 and never rounded.
template <int M>
__device__ __forceinline__ void load_norm_sum(const bf16* __restrict__ xres, const bf16* __restrict__ ob,
                                              const bf16* __restrict__ w, bf16* xs, float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int m = warp; m < M; m += NW) {
    float v[4][8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float a[8], b[8];
      cvt8(__ldcg(reinterpret_cast<const uint4*>(xres + m * H + i * 256 + lane * 8)), a);
      cvt8(__ldcg(reinterpret_cast<const uint4*>(ob + m * H + i * 256 + lane * 8)), b);
#pragma unroll
      for (int j = 0; j < 8; ++j) { v[i][j] = a[j] + b[j]; ss = fmaf(v[i][j], v[i][j], ss); }
    }
    ss = wsum(ss);
    const float r = rsqrtf(ss / (float)H + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float wf[8];
      cvt8(__ldg(reinterpret_cast<const uint4*>(w + i * 256 + lane * 8)), wf);
      __align__(16) bf16 o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16_rn(wf[j] * (v[i][j] * r));
      *reinterpret_cast<uint4*>(xs + m * H + i * 256 + lane * 8) = *reinterpret_cast<uint4*>(o);
    }
  }
  __syncthreads();
}

template <int M>
__device__ __forceinline__ void load_norm(const bf16* __restrict__ xres, const bf16* __restrict__ w, bf16* xs,
                                          float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int m = warp; m < M; m += NW) {
    float v[4][8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 u = __ldcg(reinterpret_cast<const uint4*>(xres + m * H + i * 256 + lane * 8));
      cvt8(u, v[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j) ss = fmaf(v[i][j], v[i][j], ss);
    }
    ss = wsum(ss);
    const float r = rsqrtf(ss / (float)H + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float wf[8];
      cvt8(__ldg(reinterpret_cast<const uint4*>(w + i * 256 + lane * 8)), wf);
      __align__(16) bf16 o[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16_rn(wf[j] * (v[i][j] * r));     // inductor: single rounding
      *reinterpret_cast<uint4*>(xs + m * H + i * 256 + lane * 8) = *reinterpret_cast<uint4*>(o);
    }
  }
  __syncthreads();
}

