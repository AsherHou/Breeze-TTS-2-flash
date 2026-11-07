"""LN12: a lossless 12-bit storage format for the BF16 depth-decoder weights (encoder, CPU reference decoder).

LN12 layout for a row-major bf16 matrix W[N, K] (K % 1024 == 0), matching the GEMV lane mapping
(lane l of a warp owns elements k = c*256 + l*8 + j, j = 0..7, of chunk c):
  per element: L = low byte of the bf16 (exponent LSB + 7 mantissa bits), stored raw
               nibble = sign bit + 3-bit index into a per-tensor 8-entry LUT of (exponent >> 1) values
  => 12 bits/weight, plus rare escapes ("patches": exact high byte) and a 32-bit header per row.
  data: [N][G=K/1024][slot 0..2][lane 0..31][16 B]
        slot0 = L bytes of chunks 4g, 4g+1 (8 B each); slot1 = chunks 4g+2, 4g+3; slot2 = 4 x u32 nibble words
  nibble word C (8 elements j): index bits of element j in bits 4j..4j+2; sign of element j<4 at bit 8j+7,
        sign of element j>=4 at bit 8(j-4)+3   -> GPU decode: H(0..3) = prmt(LUT, C & 0x7777) | (C & 0x80_80_80_80)
                                                             H(4..7) = prmt(LUT, (C>>16) & 0x7777) | ((C<<4) & 0x80_80_80_80)
  hdr[n]   = (patch_offset << 13) | ((shift & 7) << 10) | patch_count   (count < 1024, offset < 2^19,
             shift in -4..3 is added to every LUT byte for this row: handles rare rows with a shifted scale,
             e.g. one outlier row of backbone layer 27 down_proj with exponents up to 127)
  patch[p] = (k << 8) | true_high_byte                 (sorted by row, then k)
"""
import numpy as np

NLUT = 8


def choose_lut(u):
    e1 = (u >> 8) & 0x7F
    h = np.bincount(e1.ravel(), minlength=128)
    top = np.argsort(-h, kind="stable")[:NLUT]
    return np.sort(top).astype(np.uint8)


def encode(u, lut):
    N, K = u.shape
    assert K % 1024 == 0 and u.dtype == np.uint16
    G = K // 1024
    e1 = ((u >> 8) & 0x7F).astype(np.uint8)
    sign = (u >> 15).astype(np.uint32)
    L = (u & 0xFF).astype(np.uint8)
    inv = np.full(128, 255, np.uint8)
    inv[lut] = np.arange(NLUT, dtype=np.uint8)
    idx = inv[e1]
    esc = idx == 255
    idx = np.where(esc, 0, idx).astype(np.uint32)
    # per-row LUT shift (-4..3) minimizing escapes
    rh = np.bincount((np.arange(N, dtype=np.int64)[:, None] * 128 + e1).ravel(), minlength=N * 128).reshape(N, 128)
    best = None
    for sh in (0, 1, -1, 2, -2, 3, -3, -4):
        v = lut.astype(np.int64) + sh
        cov = rh[:, v[(v >= 0) & (v < 128)]].sum(axis=1) if ((v >= 0) & (v < 128)).all() else np.full(N, -1)
        if best is None:
            best, bsh = cov.copy(), np.zeros(N, np.int64)
        else:
            m = cov > best
            best[m], bsh[m] = cov[m], sh
    shift = bsh
    lutr = lut.astype(np.int64)[None, :] + shift[:, None]          # [N, 8]
    inv2 = np.full((N, 128), 255, np.uint8)
    inv2[np.arange(N)[:, None], lutr] = np.arange(NLUT, dtype=np.uint8)[None, :]
    idx = np.take_along_axis(inv2, e1.astype(np.int64), axis=1)
    esc = idx == 255
    idx = np.where(esc, 0, idx).astype(np.uint32)
    rows, cols = np.nonzero(esc)
    H = (u >> 8).astype(np.uint32)
    patches = (cols.astype(np.uint32) << 8) | H[rows, cols]
    cnt = np.bincount(rows, minlength=N).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    if N and (cnt.max() >= 1024 or off[-1] + cnt[-1] >= (1 << 19)):
        raise ValueError("patch header overflow")
    hdr = ((off << 13) | ((shift & 7) << 10) | cnt).astype(np.uint32)
    idx8 = idx.reshape(N, K // 8, 8)
    s8 = sign.reshape(N, K // 8, 8)
    C = np.zeros((N, K // 8), np.uint32)
    for j in range(8):
        C |= idx8[..., j] << np.uint32(4 * j)
        C |= s8[..., j] << np.uint32(8 * j + 7 if j < 4 else 8 * (j - 4) + 3)
    L8g = L.reshape(N, G, 4, 32, 8)
    Cg = C.reshape(N, G, 4, 32)
    out = np.empty((N, G, 3, 32, 16), np.uint8)
    out[:, :, 0, :, 0:8] = L8g[:, :, 0]
    out[:, :, 0, :, 8:16] = L8g[:, :, 1]
    out[:, :, 1, :, 0:8] = L8g[:, :, 2]
    out[:, :, 1, :, 8:16] = L8g[:, :, 3]
    out[:, :, 2] = np.ascontiguousarray(Cg.transpose(0, 1, 3, 2)).view(np.uint8).reshape(N, G, 32, 16)
    return out.reshape(-1), hdr, patches.astype(np.uint32)


def _prmt_lut(lut8, sel16):
    """emulate prmt.b32 d, lutlo, luthi, sel (default mode) for selectors whose nibbles have bit3 == 0"""
    r = np.zeros(sel16.shape, np.uint32)
    for b in range(4):
        i = (sel16 >> np.uint32(4 * b)) & np.uint32(7)
        r |= lut8[i].astype(np.uint32) << np.uint32(8 * b)
    return r


def decode(packed, hdr, patches, lut, N, K):
    """CPU reference of the GPU decoder (same bit operations as the CUDA kernel)."""
    G = K // 1024
    o = packed.reshape(N, G, 3, 32, 16)
    C = np.ascontiguousarray(o[:, :, 2]).view(np.uint32).reshape(N, G, 32, 4).transpose(0, 1, 3, 2)  # [N,G,u,l]
    sh = ((hdr >> np.uint32(10)) & np.uint32(7)).astype(np.int64)
    sh = np.where(sh >= 4, sh - 8, sh)                                   # sign-extend 3 bits
    lut8 = np.asarray(lut, np.uint8)
    lo = int(lut8[:4].view(np.uint32)[0]); hi = int(lut8[4:].view(np.uint32)[0])
    # GPU: lut0r = lut0 + shift * 0x01010101 (per row, 32-bit wrap), then prmt
    lo_r = ((lo + sh * 0x01010101) & 0xFFFFFFFF).astype(np.uint32)
    hi_r = ((hi + sh * 0x01010101) & 0xFFFFFFFF).astype(np.uint32)
    lut_rows = np.concatenate([lo_r[:, None], hi_r[:, None]], axis=1).view(np.uint8).reshape(N, 8)  # [N, 8]

    def prmt_rows(sel):  # sel [N,G,u,l]
        r = np.zeros(sel.shape, np.uint32)
        for b in range(4):
            i = ((sel >> np.uint32(4 * b)) & np.uint32(7)).astype(np.int64)
            r |= np.take_along_axis(lut_rows, i.reshape(N, -1), axis=1).reshape(sel.shape).astype(np.uint32) << np.uint32(8 * b)
        return r
    Hlo = prmt_rows(C & np.uint32(0x7777)) | (C & np.uint32(0x80_80_80_80))
    Hhi = prmt_rows((C >> np.uint32(16)) & np.uint32(0x7777)) | ((C << np.uint32(4)) & np.uint32(0x80_80_80_80))
    Hb = np.empty((N, G, 4, 32, 8), np.uint8)
    Hb[..., 0:4] = Hlo[..., None].view(np.uint8).reshape(N, G, 4, 32, 4)
    Hb[..., 4:8] = Hhi[..., None].view(np.uint8).reshape(N, G, 4, 32, 4)
    Lb = np.empty((N, G, 4, 32, 8), np.uint8)
    Lb[:, :, 0] = o[:, :, 0, :, 0:8]
    Lb[:, :, 1] = o[:, :, 0, :, 8:16]
    Lb[:, :, 2] = o[:, :, 1, :, 0:8]
    Lb[:, :, 3] = o[:, :, 1, :, 8:16]
    Hb = Hb.reshape(N, K)
    cnt = (hdr & np.uint32(1023)).astype(np.int64)
    off = (hdr >> np.uint32(13)).astype(np.int64)
    if cnt.sum():
        rows = np.repeat(np.arange(N), cnt)
        pidx = np.repeat(off, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        pv = patches[pidx]
        Hb[rows, (pv >> np.uint32(8)).astype(np.int64)] = (pv & np.uint32(0xFF)).astype(np.uint8)
    return (Hb.astype(np.uint16) << np.uint16(8)) | Lb.reshape(N, K).astype(np.uint16)


def packed_bits(N, K, npatch):  # data + 32-bit row header + 32-bit patches + 8-byte LUT
    return (N * K * 12 + N * 32 + npatch * 32 + NLUT * 8)


# ---------------------------------------------------------------------------------------------------------------
# LN12i: same data, but the row header and the first 3 patches are stored INLINE in a fixed 16-byte row record
#   meta[n] = uint4 {hdr_i, p0, p1, p2}   hdr_i = (ovf_offset << 13) | ((shift & 7) << 10) | patch_count
#   patches 3.. of a row live in ovf[ovf_offset ...]; unused inline slots = 0xFFFFFFFF
# -> the GPU needs no dependent load (address of meta[n] is computable) except for rows with > 3 patches.
def to_inline(hdr, patches):
    N = hdr.size
    cnt = (hdr & np.uint32(1023)).astype(np.int64)
    off = (hdr >> np.uint32(13)).astype(np.int64)
    sh = (hdr >> np.uint32(10)) & np.uint32(7)
    meta = np.full((N, 4), 0xFFFFFFFF, np.uint32)
    for s in range(3):
        m = cnt > s
        meta[m, 1 + s] = patches[off[m] + s]
    extra = np.maximum(cnt - 3, 0)
    ooff = np.concatenate([[0], np.cumsum(extra)[:-1]]).astype(np.int64)
    if N and ooff[-1] + extra[-1] >= (1 << 19):
        raise ValueError("ovf overflow")
    tot = int(extra.sum())
    if tot:
        rows = np.repeat(np.arange(N), extra)
        j = np.arange(tot) - np.repeat(np.cumsum(extra) - extra, extra)
        ovf = patches[off[rows] + 3 + j].astype(np.uint32)
    else:
        ovf = np.zeros(0, np.uint32)
    meta[:, 0] = ((ooff.astype(np.uint32) << np.uint32(13)) | (sh << np.uint32(10)) | cnt.astype(np.uint32))
    return meta, ovf


def from_inline(meta, ovf):
    """rebuild (hdr, patches) exactly as the GPU reads them: inline slots for p<3, ovf[ooff + p - 3] otherwise"""
    N = meta.shape[0]
    h = meta[:, 0]
    cnt = (h & np.uint32(1023)).astype(np.int64)
    ooff = (h >> np.uint32(13)).astype(np.int64)
    sh = (h >> np.uint32(10)) & np.uint32(7)
    tot = int(cnt.sum())
    rows = np.repeat(np.arange(N), cnt)
    p = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    inl = p < 3
    pv = np.empty(tot, np.uint32)
    pv[inl] = meta[rows[inl], 1 + p[inl]]
    pv[~inl] = ovf[ooff[rows[~inl]] + p[~inl] - 3]
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    hdr = ((off.astype(np.uint32) << np.uint32(13)) | (sh << np.uint32(10)) | cnt.astype(np.uint32))
    return hdr, pv


def inline_bits(N, K, novf):  # data + 128-bit row record + 32-bit overflow patches + LUT
    return N * K * 12 + N * 128 + novf * 32 + NLUT * 8
