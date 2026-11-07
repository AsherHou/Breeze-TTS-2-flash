#!/usr/bin/env python3
"""Compile the CUDA extensions ahead of time (scripts/serve.py also does this on its first start).

Output: <FLASH_CACHE>/ext/<torch>-cu<cuda>-sm<arch>/<extension>/ (FLASH_CACHE default ~/.cache/breeze-tts-2-flash).
Works without a GPU (the architecture then defaults to 8.9, the validated one; override with --arch).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cache-dir', help='cache directory (sets FLASH_CACHE)')
    p.add_argument('--arch', help='TORCH_CUDA_ARCH_LIST, e.g. 8.9 (default: the visible GPU, else 8.9)')
    p.add_argument('--verbose', action='store_true', help='show the compiler commands')
    a = p.parse_args()
    if a.cache_dir:
        os.environ['FLASH_CACHE'] = str(Path(a.cache_dir).expanduser().resolve())
    if a.arch:
        os.environ['TORCH_CUDA_ARCH_LIST'] = a.arch
    sys.path.insert(0, str(ROOT))
    from flash import kernels, paths
    t0 = time.perf_counter()
    tc = kernels.toolchain()
    print(f'cache {paths.cache_dir()}\ntoolchain {json.dumps(tc)}', flush=True)
    for name in kernels.SPECS:
        kernels.load(name, verbose=a.verbose)
        print(f'{name}: {kernels.info()[name]}', flush=True)
    print(json.dumps(dict(build_root=str(kernels.build_root()), total_s=round(time.perf_counter() - t0, 1),
                          extensions={n: kernels.info()[n] for n in kernels.SPECS}), indent=1))


if __name__ == '__main__':
    main()
