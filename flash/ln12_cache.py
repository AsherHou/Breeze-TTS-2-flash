"""LN12 encoding of the depth-decoder weights, cached on disk.

The depth frame megakernel reads every depth matrix (12 layers x q,k,v,o,gate,up,down, the input projector and the
15 codebook heads) in the LN12 format (flash/ln12_format.py): about 530 MiB instead of 700 MiB of BF16.
Encoding runs on the CPU and proves itself with a full decode round trip; it is done once and stored in
<FLASH_CACHE>/ln12/ln12-<key>.pt, where <key> is derived from the sha256 of the source BF16 bytes of every matrix.
On later starts the file is loaded only if those source hashes match the loaded model exactly and the per-matrix
hashes of the encoded data are intact; otherwise the weights are encoded again and the file is rewritten.
"""
import hashlib
import os
import time

from . import paths

FORMAT = 'ln12-v1'


def log(msg):
    print(f'[flash] {msg}', flush=True)


def ln12_mats(W, head_t, nl=12):
    """The matrices the LN12 kernel reads, in descriptor order: per layer q,k,v,o,gate,up,down; projector; 15 heads."""
    mats = []
    for l in range(nl):
        b = l * 9
        mats += [W[b + 1], W[b + 2], W[b + 3], W[b + 4], W[b + 6], W[b + 7], W[b + 8]]
    mats.append(W[nl * 9 + 1])
    mats += [head_t[s] for s in range(15)]
    return mats


def encode_matrix_cpu(w):
    """bf16 [N,K] -> LN12 (data uint8, meta int32, ovf int32, lut0, lut1) as CPU tensors, with a round-trip proof."""
    import numpy as np
    import torch
    from . import ln12_format as llfmt
    u = w.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    N, K = u.shape
    lut = llfmt.choose_lut(u)
    packed, hdr, pat = llfmt.encode(u, lut)
    meta, ovf = llfmt.to_inline(hdr, pat)
    h2, p2 = llfmt.from_inline(meta, ovf)
    assert np.array_equal(llfmt.decode(packed, h2, p2, lut, N, K), u), 'LN12 round trip failed'
    assert int((meta[:, 0] & 1023).max()) <= 31, 'megakernel needs <= 31 patches per row'
    l8 = np.ascontiguousarray(lut.astype(np.uint8))
    lo, hi = int(l8[:4].view(np.uint32)[0]), int(l8[4:].view(np.uint32)[0])
    ovf32 = ovf.view(np.int32) if ovf.size else np.zeros(1, np.int32)
    return (torch.from_numpy(np.ascontiguousarray(packed)), torch.from_numpy(np.ascontiguousarray(meta).view(np.int32)),
            torch.from_numpy(np.ascontiguousarray(ovf32)), lo, hi)


def sha_tensor(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _enc_sha(a, b, c):
    return hashlib.sha256(a.numpy().tobytes() + b.numpy().tobytes() + c.numpy().tobytes()).hexdigest()


def cache_path(src):
    key = hashlib.sha256('\n'.join([FORMAT] + list(src)).encode()).hexdigest()[:16]
    return paths.cache_dir() / 'ln12' / f'ln12-{key}.pt'


def _load(path, src):
    import torch
    try:
        d = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as ex:  # noqa: BLE001
        log(f'LN12 cache {path} unreadable ({type(ex).__name__}: {ex}); encoding again')
        return None
    if d.get('format') != FORMAT or d.get('src_sha256') != src or len(d.get('enc', ())) != len(src):
        log(f'LN12 cache {path} does not match the loaded weights; encoding again')
        return None
    for (a, b, c, _lo, _hi), es in zip(d['enc'], d['enc_sha256']):
        if _enc_sha(a, b, c) != es:
            log(f'LN12 cache {path} is corrupt; encoding again')
            return None
    return [(a, b, c, int(lo), int(hi)) for a, b, c, lo, hi in d['enc']]


def _save(path, src, enc):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    torch.save(dict(format=FORMAT, src_sha256=list(src), enc=[tuple(e) for e in enc],
                    enc_sha256=[_enc_sha(a, b, c) for a, b, c, _lo, _hi in enc]), tmp)
    os.replace(tmp, path)


def encode_all(mats):
    enc = []
    t0 = time.perf_counter()
    for i, m in enumerate(mats):
        enc.append(encode_matrix_cpu(m))
        if (i + 1) % 14 == 0 or i + 1 == len(mats):
            log(f'LN12 encoding {i + 1}/{len(mats)} matrices ({time.perf_counter() - t0:.0f} s)')
    return enc


def get(mats, device):
    """Encoded descriptors for `mats` on `device`: [(data, meta, ovf, lut0, lut1)] and how they were obtained."""
    src = [sha_tensor(m) for m in mats]
    path = cache_path(src)
    enc, how = None, 'encoded'
    if path.is_file():
        enc = _load(path, src)
        how = 'cache' if enc is not None else 'encoded (cache mismatch)'
    if enc is None:
        log(f'LN12: encoding {len(mats)} depth-decoder matrices on the CPU (first start; cached in {path})')
        enc = encode_all(mats)
        try:
            _save(path, src, enc)
            log(f'LN12 cache written: {path} ({path.stat().st_size / 2 ** 20:.0f} MiB)')
        except Exception as ex:  # noqa: BLE001  (a read-only cache only costs the encoding time next start)
            log(f'could not write the LN12 cache {path}: {ex!r}')
    dev = [(a.to(device), b.to(device), c.to(device), lo, hi) for a, b, c, lo, hi in enc]
    return dev, how, str(path)

