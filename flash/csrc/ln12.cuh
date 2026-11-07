// Breeze-TTS-2-flash: LN12 compressed-weight versions of the depth_common.cuh GEMV helpers (included INSIDE
// namespace dd, after depth_common.cuh). Same per-lane accumulation order (k = c*256 + lane*8 + j, c ascending, j ascending) and the same
// xor-butterfly reduction -> bitwise identical fp32 results to dot_rows / dot_rows_part on the BF16 weights.
// Format: see flash/ln12_format.py (data [N][K/1024][3][32] uint4, inline 16-byte row record {hdr, p0, p1, p2}, overflow list).
// Megakernel restriction (checked by the encoder for the depth tensors): <= 31 patches per row, so the 32 lanes of
// one register hold all patches of a row and no reload is ever needed (lane >= cnt holds 0xffffffff = sentinel k).
struct CW {
  const uint4* data;
  const uint4* meta;
  const uint32_t* ovf;
  uint32_t lut0, lut1;
};
#ifndef DD_CW_N
#define DD_CW_N (NL * 7 + 1 + NCB)
#endif
__constant__ CW c_cw[DD_CW_N];   // [l*7 + {q,k,v,o,gate,up,down}], [NL*7] projector, [NL*7+1+s] codebook head s
enum { CQ = 0, CK_ = 1, CV = 2, CO = 3, CG = 4, CU = 5, CD = 6 };
__device__ __forceinline__ const CW& cwl(int l, int which) { return c_cw[l * 7 + which]; }

struct RowC {
  uint32_t l0, l1, pv;
  int pi, nk;
};
__device__ __forceinline__ void rowc_init(RowC& s, const CW& w, int n) {
  const int lane = threadIdx.x & 31;
  const uint4 m = __ldg(w.meta + n);
  const int cnt = (int)(m.x & 1023u);
  const int sh = ((int)(m.x << 19)) >> 29;
  s.l0 = w.lut0 + (uint32_t)sh * 0x01010101u;
  s.l1 = w.lut1 + (uint32_t)sh * 0x01010101u;
  s.pv = lane == 0 ? m.y : lane == 1 ? m.z : lane == 2 ? m.w
       : (lane < cnt ? __ldg(w.ovf + (m.x >> 13) + lane - 3) : 0xffffffffu);
  s.pi = 0;
}
__device__ __forceinline__ void rowc_start(RowC& s) { s.nk = (int)(__shfl_sync(0xffffffffu, s.pv, 0) >> 8); }
__device__ __forceinline__ uint4 ln12_dec(uint32_t C, uint32_t L0, uint32_t L1, RowC& s, int kend) {
  const int lane = threadIdx.x & 31;
  uint32_t Hlo = __byte_perm(s.l0, s.l1, C & 0x7777u) | (C & 0x80'80'80'80u);
  uint32_t Hhi = __byte_perm(s.l0, s.l1, (C >> 16) & 0x7777u) | ((C << 4) & 0x80'80'80'80u);
  while (s.nk < kend) {  // warp-uniform; rare (~1e-4 of weights)
    const uint32_t pval = __shfl_sync(0xffffffffu, s.pv, s.pi);
    const int kk = (int)(pval >> 8);
    const uint32_t hv = pval & 0xffu;
    if (((kk >> 3) & 31) == lane) {
      const int j = kk & 7, sh = (j & 3) * 8;
      if (j < 4) Hlo = (Hlo & ~(0xffu << sh)) | (hv << sh);
      else Hhi = (Hhi & ~(0xffu << sh)) | (hv << sh);
    }
    ++s.pi;
    s.nk = (int)(__shfl_sync(0xffffffffu, s.pv, s.pi & 31) >> 8);
  }
  uint4 q;
  q.x = __byte_perm(L0, Hlo, 0x5140);
  q.y = __byte_perm(L0, Hlo, 0x7362);
  q.z = __byte_perm(L1, Hhi, 0x5140);
  q.w = __byte_perm(L1, Hhi, 0x7362);
  return q;
}
template <int u>
__device__ __forceinline__ void slot_parts(const uint4 (&g)[3], uint32_t& C, uint32_t& L0, uint32_t& L1) {
  C = u == 0 ? g[2].x : u == 1 ? g[2].y : u == 2 ? g[2].z : g[2].w;
  const uint4& L = g[u >> 1];
  L0 = (u & 1) ? L.z : L.x;
  L1 = (u & 1) ? L.w : L.y;
}
// one group (4 chunks = 1024 columns) of R compressed rows; x columns indexed as xs[m*ldx + (kx0 + u*256 + lane*8)]
template <int M, int R>
__device__ __forceinline__ void fma_group_c(const uint4 (&w)[R][3], RowC (&st)[R], const bf16* __restrict__ xs, int ldx,
                                            int kx0, int kabs0, float (&acc)[R][M]) {
  const int lane = threadIdx.x & 31;
#define DD_C_CHUNK(U_)                                                                              \
  {                                                                                                 \
    float wf[R][8];                                                                                 \
    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                 \
      uint32_t C, L0, L1;                                                                           \
      slot_parts<U_>(w[r], C, L0, L1);                                                              \
      cvt8(ln12_dec(C, L0, L1, st[r], kabs0 + (U_ + 1) * 256), wf[r]);                              \
    }                                                                                               \
    _Pragma("unroll") for (int m = 0; m < M; ++m) {                                                 \
      float xf[8];                                                                                  \
      cvt8(*reinterpret_cast<const uint4*>(xs + m * ldx + kx0 + U_ * 256 + lane * 8), xf);          \
      _Pragma("unroll") for (int r = 0; r < R; ++r)                                                 \
          _Pragma("unroll") for (int j = 0; j < 8; ++j) acc[r][m] = fmaf(wf[r][j], xf[j], acc[r][m]); \
    }                                                                                               \
  }
  DD_C_CHUNK(0) DD_C_CHUNK(1) DD_C_CHUNK(2) DD_C_CHUNK(3)
#undef DD_C_CHUNK
}
// full-K dot of R compressed rows (row nr[r] of matrix *cw[r]) with xs [M][K]; reduced over the warp
template <int M, int R, int K>
__device__ __forceinline__ void dot_rows_c(const CW* const* cw, const int* nr, const bf16* __restrict__ xs, float (&acc)[R][M]) {
  constexpr int G = K / 1024;
  const int lane = threadIdx.x & 31;
  const uint4* wp[R];
  RowC st[R];
#pragma unroll
  for (int r = 0; r < R; ++r) wp[r] = cw[r]->data + (size_t)nr[r] * (G * 96) + lane;
#pragma unroll
  for (int r = 0; r < R; ++r) rowc_init(st[r], *cw[r], nr[r]);
#pragma unroll 1
  for (int g = 0; g < G; ++g) {
    uint4 w[R][3];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int s = 0; s < 3; ++s) w[r][s] = ld_w(wp[r] + g * 96 + s * 32);
    if (g == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) rowc_start(st[r]);
    }
    fma_group_c<M, R>(w, st, xs, K, g * 1024, g * 1024, acc);
  }
  reduce_acc<M, R>(acc);
}
// partial dot over columns [k0, k0+KR) with xs holding only those columns ([M][KR]); no reduction.
// The row state (patch cursor) must persist across the parts -> caller owns st and calls rowc_init/rowc_start once.
template <int M, int R, int K, int KR>
__device__ __forceinline__ void dot_rows_part_c(const CW* const* cw, const int* nr, RowC (&st)[R], const bf16* __restrict__ xs,
                                                float (&acc)[R][M], int k0) {
  constexpr int G = K / 1024;
  const int lane = threadIdx.x & 31;
  static_assert(KR % 1024 == 0, "KR must be a multiple of 1024");
#pragma unroll 1
  for (int g = k0 / 1024; g < (k0 + KR) / 1024; ++g) {
    uint4 w[R][3];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int s = 0; s < 3; ++s) w[r][s] = ld_w(cw[r]->data + (size_t)nr[r] * (G * 96) + lane + g * 96 + s * 32);
    fma_group_c<M, R>(w, st, xs, KR, g * 1024 - k0, g * 1024, acc);
  }
}
// L2 prefetch of compressed rows [r0, r1): data (K*3/2 bytes per row) + inline row records (16 B per row)
__device__ __forceinline__ void prefetch_rows_c(const CW& w, int r0, int r1, int K, int max_bytes) {
  const size_t rb = (size_t)K * 3 / 2;
  size_t bytes = (size_t)(r1 - r0) * rb;
  if (bytes > (size_t)max_bytes) bytes = max_bytes;
  const char* c = reinterpret_cast<const char*>(w.data) + (size_t)r0 * rb;
  for (size_t off = (size_t)threadIdx.x * 128; off < bytes; off += (size_t)NT * 128) pf_l2(c + off);
  if (threadIdx.x < 16) {
    const char* m = reinterpret_cast<const char*>(w.meta + r0);
    const size_t mb = (size_t)(r1 - r0) * 16;
    if ((size_t)threadIdx.x * 128 < mb) pf_l2(m + threadIdx.x * 128);
  }
}
