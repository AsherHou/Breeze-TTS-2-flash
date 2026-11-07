"""Find the CUDA toolkit for torch.utils.cpp_extension (nvcc, CUDA headers and libcudart; CUDA 12.8 matches the
torch 2.9.1 cu128 build used for validation).

Search order: $CUDA_HOME / $CUDA_PATH -> nvcc on PATH -> $CONDA_PREFIX -> /usr/local/cuda-12.8 -> /usr/local/cuda.
Also puts the Python environment's scripts directory on PATH, so that pip's `ninja` is found even when the virtual
environment is not activated.
"""
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _ensure_scripts_on_path():
    parts = os.environ.get('PATH', '').split(os.pathsep)
    for d in (sysconfig.get_path('scripts'), str(Path(sys.executable).parent)):
        if d and d not in parts and Path(d).is_dir():
            parts.insert(0, d)
    os.environ['PATH'] = os.pathsep.join(parts)


def _nvcc_version(nvcc):
    try:
        out = subprocess.run([str(nvcc), '--version'], capture_output=True, text=True, timeout=60).stdout
        return out.strip().splitlines()[-1]
    except Exception as ex:  # noqa: BLE001
        return f'unknown ({ex})'


def find_cuda_home():
    """(cuda_home, how) or (None, None)."""
    for var in ('CUDA_HOME', 'CUDA_PATH'):
        v = os.environ.get(var)
        if v and (Path(v) / 'bin' / 'nvcc').is_file():
            return Path(v), var
    nvcc = shutil.which('nvcc')
    if nvcc:
        return Path(nvcc).resolve().parent.parent, 'nvcc on PATH'
    conda = os.environ.get('CONDA_PREFIX')
    if conda and (Path(conda) / 'bin' / 'nvcc').is_file():
        return Path(conda), 'CONDA_PREFIX'
    for d in ('/usr/local/cuda-12.8', '/usr/local/cuda'):
        if (Path(d) / 'bin' / 'nvcc').is_file():
            return Path(d), d
    return None, None


def ensure_cuda_home(cache=None):
    """Set CUDA_HOME (and PATH) so that torch's extension builder finds nvcc. Returns (cuda_home, how, version)."""
    _ensure_scripts_on_path()
    home, how = find_cuda_home()
    if home is None:
        raise RuntimeError('no CUDA toolkit found (nvcc): install the CUDA 12.8 toolkit and/or set CUDA_HOME')
    os.environ['CUDA_HOME'] = str(home)
    bindir = str(home / 'bin')
    if bindir not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
    if shutil.which('ninja') is None:
        raise RuntimeError('ninja not found (pip install ninja)')
    return home, how, _nvcc_version(home / 'bin' / 'nvcc')
