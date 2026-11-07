"""Build / load the CUDA extensions (torch.utils.cpp_extension.load, JIT with ninja).

Extensions are compiled once into <FLASH_CACHE>/ext/<torch>-cu<cuda>-sm<arch>/<name>/ and reused afterwards
(ninja rebuilds only when a source or a flag changes). Sources live in flash/csrc.

  flash_bb_attn   bb_attn.cu        backbone decode attention (q/k norm + RoPE + KV write + exact-length GQA)
  flash_bb_fused  bb_fused.cu       backbone decode GEMVs with fused RMSNorm / residual / SiLU*up
  flash_dd_full   depth_frame.cu    depth frame megakernel, LN12 weights, evict_last prefetch + step table
                                    (batch rows 1 and 2)
  flash_dd_kv     depth_frame.cu    depth frame megakernel, LN12 weights (batch rows 4)
"""
import os
import threading
import time
from pathlib import Path

from . import paths
from .toolchain import ensure_cuda_home

FLAGS = ['-O3', '-std=c++17', '-lineinfo']
DEPTH_VARIANTS = {'full': ['-DDD_PF_EVICT_LAST', '-DDD_STEP_TABLE'], 'kv': []}
SPECS = {
    'flash_bb_attn': (['bb_attn.cu'], []),
    'flash_bb_fused': (['bb_fused.cu'], []),
    'flash_dd_full': (['depth_frame.cu'], DEPTH_VARIANTS['full']),
    'flash_dd_kv': (['depth_frame.cu'], DEPTH_VARIANTS['kv']),
}
_LOCK = threading.Lock()
_MODS = {}
_INFO = {}


def _arch():
    a = os.environ.get('TORCH_CUDA_ARCH_LIST', '').strip()
    if a:
        return a
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        a = f'{p.major}.{p.minor}'
    else:
        a = '8.9'                       # building without a visible GPU: the validated architecture
    os.environ['TORCH_CUDA_ARCH_LIST'] = a
    return a


def build_root():
    import torch
    arch = _arch().replace(' ', '_').replace(';', '_').replace('.', '')
    tag = f'torch{torch.__version__.split("+")[0]}-cu{(torch.version.cuda or "none").replace(".", "")}-sm{arch}'
    d = paths.cache_dir() / 'ext' / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def toolchain():
    if 'toolchain' not in _INFO:
        home, src, ver = ensure_cuda_home()
        os.environ.setdefault('MAX_JOBS', str(max(1, os.cpu_count() or 1)))
        _INFO['toolchain'] = dict(cuda_home=str(home), source=src, nvcc=ver, arch=_arch())
    return _INFO['toolchain']


def load(name, verbose=False):
    """Compile (first time) and import extension `name`."""
    with _LOCK:
        if name in _MODS:
            return _MODS[name]
        toolchain()
        from torch.utils.cpp_extension import load as _load
        sources, extra = SPECS[name]
        d = build_root() / name
        d.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        mod = _load(name=name, sources=[str(paths.CSRC / s) for s in sources], extra_include_paths=[str(paths.CSRC)],
                    extra_cuda_cflags=FLAGS + extra, build_directory=str(d), verbose=verbose)
        _MODS[name] = mod
        _INFO[name] = dict(file=str(Path(mod.__file__).resolve()), seconds=round(time.perf_counter() - t, 1))
        return mod


def load_all(verbose=False):
    return {n: load(n, verbose) for n in SPECS}


def attn():
    return load('flash_bb_attn')


def fused():
    return load('flash_bb_fused')


def depth(variant):
    return load(f'flash_dd_{variant}')


def info():
    return dict(_INFO)
