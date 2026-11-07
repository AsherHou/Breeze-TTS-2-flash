#!/usr/bin/env python3
"""Start the Breeze TTS 2 HTTP API with Breeze-TTS-2-flash.

Equivalent to the official `python -m breeze_infer.api <model> --fast-all` (same routes, same warmup profile: the
official configs/fast.json unless --warmup-profile is given), with every acceleration hook installed before the model
is loaded. FLASH_DISABLE=1 installs nothing, which makes this exactly the official server (for comparisons).

First start: downloads the model (if --model is not given), compiles the CUDA extensions and builds the LN12 weight
cache; later starts reuse all three.

  python scripts/serve.py                         # download BreezeBlue/Breeze-TTS-2 (fixed revision) on first use
  python scripts/serve.py --model /path/to/Breeze-TTS-2 --port 7860
Environment: FLASH_CACHE (cache dir, default ~/.cache/breeze-tts-2-flash), FLASH_UPSTREAM (official runtime
checkout, default third_party/breeze-tts), FLASH_DISABLE=1, HF_HOME (Hugging Face download cache).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = 'BreezeBlue/Breeze-TTS-2'
MODEL_REVISION = '3e28c5151381a722f1d8661b4118c298caa77aa4'
LICENSE_NOTE = ('Model weights: BreezeBlue/Breeze-TTS-2 under the BreezeBlue Research and Non-Commercial License -- '
                'research and non-commercial use only (see LICENSE in the model directory).')


def main():
    p = argparse.ArgumentParser(description='Breeze TTS 2 API (official runtime + Breeze-TTS-2-flash)')
    p.add_argument('--model', help='local checkpoint directory; default: download %s@%s via huggingface_hub'
                   % (MODEL_REPO, MODEL_REVISION[:12]))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--warmup-profile', help='warmup profile JSON (default: the official configs/fast.json)')
    p.add_argument('--cache-dir', help='cache directory (sets FLASH_CACHE)')
    p.add_argument('--upstream', help='official runtime checkout (sets FLASH_UPSTREAM)')
    p.add_argument('--manifest', help='write the warmup manifest (timings, memory, acceleration state) to this file')
    a = p.parse_args()
    if a.cache_dir:
        os.environ['FLASH_CACHE'] = str(Path(a.cache_dir).expanduser().resolve())
    if a.upstream:
        os.environ['FLASH_UPSTREAM'] = str(Path(a.upstream).expanduser().resolve())
    sys.path.insert(0, str(ROOT))
    from flash import paths
    upstream = paths.upstream_dir()
    cache = paths.cache_dir()
    if not (upstream / 'breeze_infer' / 'api.py').is_file():
        sys.exit(f'official runtime not found at {upstream} (set --upstream or FLASH_UPSTREAM)')
    sys.path.insert(0, str(upstream))
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache / 'torchinductor'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(cache / 'triton'))

    from breeze_infer import api
    print(LICENSE_NOTE, flush=True)
    if a.model:
        model_dir = Path(a.model).expanduser().resolve()
        if not model_dir.is_dir():
            sys.exit(f'--model {a.model}: not a directory')
    else:
        print(f'model: {MODEL_REPO}@{MODEL_REVISION[:12]} from the Hugging Face Hub (about 7 GB on the first start, '
              f'reused from the Hugging Face cache afterwards)', flush=True)
        model_dir = Path(api.resolve_model_dir(MODEL_REPO, MODEL_REVISION))
    if a.warmup_profile:
        api.FAST_CONFIG = Path(a.warmup_profile).expanduser().resolve()

    import flash
    flash.install(api, model_dir=model_dir)

    if a.manifest:
        out = Path(a.manifest).expanduser().resolve()
        inner = api.FastBreezeStreamingRuntime.warmup_from_profile

        def warmup_from_profile(self, profile, *args, **kw):
            manifest = inner(self, profile, *args, **kw)
            import torch
            manifest['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            manifest['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            manifest['cuda_allocated_after_warmup'] = torch.cuda.memory_allocated()
            manifest['flash'] = flash.info()
            out.write_text(json.dumps(manifest, indent=2, default=str))
            print(f'warmup manifest -> {out}', flush=True)
            return manifest
        api.FastBreezeStreamingRuntime.warmup_from_profile = warmup_from_profile

    print(f'official runtime {upstream}; warmup profile {api.FAST_CONFIG}; model {model_dir}', flush=True)
    sys.argv = [sys.argv[0], str(model_dir), '--host', a.host, '--port', str(a.port), '--fast-all']
    api.main()


if __name__ == '__main__':
    main()
