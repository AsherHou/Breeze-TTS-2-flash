"""Pre-flight checks. Every check runs before anything is patched; a failed check means the official code path is
used unchanged (the reason is printed), never a crash.

The kernels and hooks were validated on one configuration only:
  GPU     NVIDIA GeForce RTX 4080 SUPER (compute capability 8.9, 80 SMs, 64 MiB L2, 16 GB)
  stack   torch 2.9.1 (CUDA 12.8), transformers 4.57.3, qwen-tts 0.1.1, official runtime commit daa6cb2
The depth megakernel uses fixed grids of 64 and 80 persistent blocks (one per SM) and L2 prefetch windows sized
for a 64 MiB L2, so a GPU needs compute capability 8.9, at least 80 SMs and at least 64 MiB of L2.
"""
import hashlib
from importlib import metadata
from pathlib import Path

UPSTREAM_COMMIT = 'daa6cb2efe5471f09369229f322d20bb956cb3c3'
# sha256 of the official files whose internals the hooks replace or wrap (commit daa6cb2)
UPSTREAM_FILES = {
    'breeze_infer/api.py': 'acee4503bcdddfef20ab1078d98d2079e15b8c2dd1e5cde65fcc0b7245460910',
    'breeze_infer/audio.py': '3dc7977d8b1f3097918d4164e5607c20d99491877f3c4723649a36724713cff4',
    'breeze_infer/templates.py': '52057796b74b95e34adf7238cb07ed1484941a5f6c168760632ac54a0bd0ca06',
    'breeze_infer/runtime.py': 'a257d868b867009edddd8ba53b22d3b66394057801c62f198879b1c992637ab9',
    'models/fast_streaming.py': '7748cc11fef2a5f54ae82fefc96325673d71b8cb9b1bdd7bb92275c4c100a825',
    'models/cudagraph/backbone_graph.py': '942b07eb2c16a31f362e10ecc23e9d1c131fc8cc3116550650010d2ec83fd024',
    'models/cudagraph/backbone_prefill_graph.py': 'bd90e34eb8bc4e507af9e222b58db741e913e5c833cd9c5625f4168660d45d82',
    'models/cudagraph/depth_decoder_graph.py': '407bb07d0cd610f9c11748655450ad7708f8bab811d2129382925bf729b88acc',
    'models/cudagraph/sampling.py': '3dcaa35fd106c730bc630a998d3a6ed3329beb55e8df03d76d3f857346a86623',
    'models/stream_runtime/stream/kv_cache.py': '1162ba9bce3fbb2e64b6fcb2e95e20b9f6efb9dfce541be61ab8346f123c8f41',
    'models/stream_runtime/stream/lane.py': '074a88bd1c6ea367993f66b352df48be372a8236c66dbb7a7e151f50bd620486',
    'models/stream_runtime/stream/runtime.py': '6b886844b7e317a4218a9f6f68735e502d8fd853ce41dd8c90648bf1de88c320',
}
PACKAGES = {'torch': '2.9.1', 'transformers': '4.57.3', 'qwen-tts': '0.1.1'}
REQUIRED_CC = (8, 9)
MIN_SMS = 80
MIN_L2 = 64 << 20


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_upstream(upstream):
    """None if the official runtime is the validated revision, else a reason."""
    bad = []
    for rel, want in UPSTREAM_FILES.items():
        p = Path(upstream) / rel
        if not p.is_file():
            bad.append(f'{rel} missing')
        elif _sha(p) != want:
            bad.append(f'{rel} differs')
    if bad:
        return (f'official runtime at {upstream} is not the validated commit {UPSTREAM_COMMIT[:7]} '
                f'({", ".join(bad[:4])}{" ..." if len(bad) > 4 else ""})')
    return None


def check_packages():
    bad = []
    for name, want in PACKAGES.items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        if got is None or got.split('+')[0] != want:
            bad.append(f'{name} {got} (validated {want})')
    return f'package versions differ: {", ".join(bad)}' if bad else None


def check_gpu():
    """None if the current CUDA device matches the kernels' assumptions, else a reason. Initialises CUDA."""
    import torch
    if not torch.cuda.is_available():
        return 'CUDA is not available'
    dev = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(dev)
    if (p.major, p.minor) != REQUIRED_CC:
        return f'{p.name}: compute capability {p.major}.{p.minor}, the kernels are built and validated for 8.9 only'
    if p.multi_processor_count < MIN_SMS:
        return (f'{p.name}: {p.multi_processor_count} SMs; the depth megakernel needs at least {MIN_SMS} '
                f'co-resident persistent blocks')
    l2 = getattr(p, 'L2_cache_size', None)
    if l2 is None:
        return f'{p.name}: L2 cache size unknown (torch does not report it)'
    if l2 < MIN_L2:
        return f'{p.name}: L2 cache {l2 >> 20} MiB; the prefetch schedule assumes at least {MIN_L2 >> 20} MiB'
    return None


def device_summary():
    try:
        import torch
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        return dict(gpu=p.name, cc=f'{p.major}.{p.minor}', sms=p.multi_processor_count,
                    l2_mib=(getattr(p, 'L2_cache_size', 0) or 0) >> 20, memory_mib=p.total_memory >> 20,
                    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    except Exception as ex:  # noqa: BLE001
        return dict(error=str(ex))
