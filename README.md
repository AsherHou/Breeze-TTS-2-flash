# Breeze-TTS-2-flash

**Lossless CUDA-kernel acceleration for Breeze TTS 2: streaming text-to-speech at more than 4× real time, with 32–38 ms to first audio, on a single RTX 4080 SUPER.**

**English** | [简体中文](README.zh-CN.md)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#requirements)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](#requirements)
[![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](#requirements)
[![GPU: RTX 4080 SUPER](https://img.shields.io/badge/GPU-RTX%204080%20SUPER-76B900?logo=nvidia&logoColor=white)](#requirements)

Breeze-TTS-2-flash speeds up inference for [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), the open-weight English and Chinese text-to-speech model. The official runtime stays unchanged. On top of its `--fast-all` path, custom CUDA kernels take over the heavy compute, with the same weights, no quantization and no reduced precision. On an RTX 4080 SUPER it runs about **2× faster than official `--fast-all`** and about **3.5× faster than official eager mode**, behind the same HTTP API. That makes low-latency streaming TTS practical on a single consumer GPU.

## Quick Start

```bash
git clone https://github.com/AsherHou/Breeze-TTS-2-flash.git
cd Breeze-TTS-2-flash
pip install -r requirements.txt
python scripts/serve.py
```

Use a Python 3.12 virtual environment. As an alternative, `bash install.sh` creates `.venv` and installs the requirements into it; start the server afterwards with `.venv/bin/python scripts/serve.py`. You also need the CUDA 12.8 toolkit (`nvcc`), which pip cannot install. See [Requirements](#requirements).

The server listens on `http://127.0.0.1:7860` and exposes the same routes as the official server: `/health`, `/v1/model`, `/v1/audio/speech` and `/v1/audio/speech/batch`. When the startup log shows `[flash] acceleration ON (...)`, the custom kernels are active.

Generate speech:

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F "text=你好，欢迎使用语音合成服务。" \
  -F "instruction=年轻女声，使用标准普通话，语气自然清晰，语速适中。" \
  -F "cfg_scale=4" \
  -F "seed=42" \
  -o speech.pcm

# The response is streamed raw PCM (16-bit little-endian, mono).
# The sample rate is in the X-Sample-Rate response header (24000 for this model).
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

| Form field | Meaning |
|---|---|
| `text` | Text to synthesize (required). |
| `instruction` | Optional voice description, e.g. gender, accent, tone and speaking rate. On its own, this is the official **Voice Design** mode. |
| `cfg_scale` | Classifier-free guidance scale. Must be greater than 0; the default is 1.0, and the benchmarks below use 4. |
| `seed` | Random seed (default 42). |
| `ref_audio` + `ref_text` | Optional reference clip (`-F ref_audio=@ref.wav`) and its exact transcript for **Voice Clone**. Send both or neither. Add an `instruction` as well to steer the cloned voice (**Voice Direction**). |

**First start.** The first launch prepares everything automatically. Later launches reuse the results.

| Step | Approx. time | Where it goes |
|---|---|---|
| Download the model (about 7.2 GB) | ~1–2 min at 1 Gbit/s, ~10 min at 100 Mbit/s | Hugging Face cache (`HF_HOME`) |
| Compile the 4 CUDA extensions | ~100 s (measured on a 32-core CPU) | `~/.cache/breeze-tts-2-flash/ext/` |
| Build the LN12 weight cache (~530 MB) | ~10–20 s, then ~0.8 s to load on later starts | `~/.cache/breeze-tts-2-flash/ln12/` |
| Record a codec reference for self-checks | a few extra seconds | `~/.cache/breeze-tts-2-flash/codec-pin/` |
| Load the model and warm up (every start) | ~17 s | — |

To use a checkpoint you already have, pass `python scripts/serve.py --model /path/to/Breeze-TTS-2`. Without `--model`, the server downloads `BreezeBlue/Breeze-TTS-2` at a fixed revision.

## At a Glance

Time to generate a 4.2-second clip of speech on an RTX 4080 SUPER:

| Runtime | Time |
|---|---|
| Official (eager) | ≈ 3.4 s * |
| Official (`--fast-all`) | ≈ 2.0 s * |
| **Breeze-TTS-2-flash** | **≈ 0.9 s** |

\* Calculated from the measured RTF of each runtime (0.798 for eager, 0.468 for `--fast-all`), not timed directly.

## Performance

| | Official (eager) | Official (`--fast-all`) | **Breeze-TTS-2-flash** |
|---|---|---|---|
| RTF, short / medium / long (lower is better) | 0.798 / 0.788 / 0.787 | 0.468 / 0.463 / 0.461 | **0.233 / 0.228 / 0.227** |
| Speed vs. real time (1 / RTF) | 1.3× | 2.1–2.2× | **4.3–4.4×** |
| Time to first audio | 148–150 ms | 47–55 ms | **32–38 ms** |
| GPU time per frame (1 frame = 80 ms of audio) | 63.2 ms | 36.5 ms | **18.2 ms** |
| GPU memory | 8.4 GB (peak) | 12.8 GB | 13.6 GB (after warmup) |
| Startup warmup | none (no CUDA graphs) | not measured | 7.5 s; ~17 s from launch to ready with warm caches |

By RTF, Breeze-TTS-2-flash is 2.0× faster than official `--fast-all` and 3.4–3.5× faster than official eager mode.

**How we measured.**

- **Requests.** HTTP streaming requests to `/v1/audio/speech`. Each of the three Chinese sentences (short, medium and long) ran 3 times, and the table shows the median. Settings were `cfg_scale` 4, seed 42 + run index, and one fixed voice instruction.
- **Metrics.** RTF is the total request time divided by the audio duration. Time to first audio is the time until the first audio chunk arrives.
- **Hardware.** NVIDIA GeForce RTX 4080 SUPER, 16 GB.
- **Warmup profiles.** The official `--fast-all` default warmup profile (`configs/fast.json`, inputs up to 512 tokens) runs out of memory on a 16 GB card. The official baseline therefore uses `configs/fast-16gb.json`, which is the same profile capped at 256 tokens. Breeze-TTS-2-flash runs the official 512-token profile.
- **Reproducing.** Run `scripts/benchmark.py` against a running server. It uses the same texts and settings.

## What's Optimized

All changes run inside the official `--fast-all` CUDA-graph framework, and the official code itself is not edited.

- **Depth-decoder megakernel.** For every 80 ms audio frame, the depth decoder predicts 15 codebook tokens one after another. The official path launches many small kernels for that work. Here, the whole frame runs as one persistent megakernel, and the official sampling step is fused into the same kernel.
- **Lossless 12-bit weights (LN12).** The depth decoder's BF16 weights are stored in a 12-bit format that decodes back to exactly the same BF16 values on the fly. Each frame reads about 25% fewer bytes of weights.
- **GPU-tuned work layout.** First-layer projections, which depend only on the previous token, are precomputed into lookup tables. Tiling, the number of persistent blocks and L2 prefetching are tuned for the GPU.
- **Leaner backbone attention.** Backbone decode attention covers only the real sequence length instead of the padded maximum. Each layer is fused into a handful of kernels that combine normalization, matrix-vector products, RoPE and the KV-cache write.
- **Trimmed prefill attention.** Prefill attention skips positions that are masked out, because their output was zero anyway.
- **No CPU-GPU syncs in the main loop.** The streaming loop no longer waits on the GPU every frame. The codec runs on a side stream, and sampling results come back in a single asynchronous copy.
- **Codec race fix.** The official streaming codec shifts its KV cache with an overlapping in-place copy, which is a data race. This fix removes it, and the agreement between streaming and offline decoding of the same tokens rises from 16–22 dB to about 65 dB.
- **Stable codec convolutions.** The codec's convolution algorithm choice is fixed, so every launch produces the same output, and a self-check compares it with a reference recorded on the same machine.
- **Reference-audio cache.** Voice-cloning reference clips are encoded once and cached, keyed by file hash, so repeated requests with the same reference skip the encoder.

## Quality

- **Same weights, same precision.** The model weights are byte-for-byte identical to the official checkpoint. There is no quantization and no reduction in precision.
- **Not bit-identical to the official output.** The custom kernels add numbers in a different order, so the audio is not bit-for-bit the same as the official output. Our measurements found no loss in quality:
  - **A/B tests.** In 78 A/B pairs with the same text and seed, none of 19 objective metrics showed a regression.
  - **Closeness to FP32.** Against an FP32 reference, the mean total-variation (TV) distance of the per-token probability distributions is 0.0419 for Breeze-TTS-2-flash and 0.0430 for the official runtime, so flash is slightly closer to FP32.
- **Deterministic.** Repeating the same request gives bit-identical audio. You can check this yourself:

  ```bash
  python scripts/verify_bitexact.py --base http://127.0.0.1:7860 --repeat 3
  ```

  The script exits with code 0 when every response has the same PCM SHA-256. With `--other URL` it also compares against a second server. A server running the official path (`FLASH_DISABLE=1`) will differ slightly, for the reason given above.

## Requirements

| | |
|---|---|
| GPU | Tested only on the **NVIDIA GeForce RTX 4080 SUPER** (sm_89, 80 SMs, 64 MB L2, 16 GB). No other GPU has been tested. |
| GPU memory | About 13.6 GB after warmup. |
| OS | Linux x86_64 (tested on Ubuntu 24.04) with an NVIDIA driver that supports CUDA 12.8. |
| Python | 3.12, with development headers (`python3-dev` on Debian/Ubuntu). |
| CUDA toolkit | **CUDA 12.8 with `nvcc`**, needed to compile the extensions on the first start. |
| Compiler | `gcc`/`g++` (tested with 13.3). `ninja` is installed by `requirements.txt`. |
| PyTorch | `torch==2.9.1` built for CUDA 12.8, plus `transformers==4.57.3` and `qwen-tts==0.1.1` (all pinned by `requirements.txt`). |

**About `nvcc`.** pip cannot install a working `nvcc`. The `nvidia-cuda-nvcc-cu12` wheel ships `ptxas` but no `nvcc`, so install the CUDA 12.8 toolkit from NVIDIA, for example the `cuda-toolkit-12-8` package from NVIDIA's apt repository. The toolkit is looked up in this order: `CUDA_HOME` / `CUDA_PATH`, `nvcc` on `PATH`, `$CONDA_PREFIX`, `/usr/local/cuda-12.8`, `/usr/local/cuda`.

**About PyTorch.** If pip installs a torch build for a different CUDA version, reinstall it from the CUDA 12.8 index:

```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

**Safe fallback.** At startup the package checks the GPU (compute capability 8.9, at least 80 SMs and at least 64 MB of L2), the package versions and the bundled official files. If any check fails, or if the extensions cannot be built, it does not crash. It prints `[flash] acceleration OFF, using the official code path: <reason>` and serves with the official runtime.

## Configuration

### `scripts/serve.py`

| Option | Default | Description |
|---|---|---|
| `--model DIR` | download `BreezeBlue/Breeze-TTS-2` | Local checkpoint directory. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `7860` | Port. |
| `--warmup-profile JSON` | official `configs/fast.json` | CUDA-graph warmup profile. |
| `--cache-dir DIR` | `~/.cache/breeze-tts-2-flash` | Cache directory (same as `FLASH_CACHE`). |
| `--upstream DIR` | `third_party/breeze-tts` | Official runtime checkout (same as `FLASH_UPSTREAM`). |
| `--manifest FILE` | — | Write a warmup manifest (timings, GPU memory, acceleration status) to this file. |

### Environment variables

| Variable | Description |
|---|---|
| `FLASH_CACHE` | Cache directory for compiled extensions, the LN12 weight cache, codec reference records and the torch/Triton caches. Default `~/.cache/breeze-tts-2-flash`. |
| `FLASH_UPSTREAM` | Location of the official runtime. Default `third_party/breeze-tts`. |
| `FLASH_DISABLE=1` | Install nothing and run the plain official server, for comparisons. |
| `HF_HOME` | Hugging Face cache used for the model download. |
| `CUDA_HOME` | CUDA toolkit location, if it is not found automatically. |

### Tools

```bash
# Compile the CUDA extensions ahead of time (works without a GPU; architecture defaults to 8.9)
python scripts/build_kernels.py [--arch 8.9] [--cache-dir DIR] [--verbose]

# Benchmark a running server: 1 warmup request, then short/medium/long × --repeats, medians reported
python scripts/benchmark.py --base http://127.0.0.1:7860 --repeats 3 --cfg 4 --out results/flash
#   --cases short,medium,long   choose which texts to run
#   --label NAME                tag stored in benchmark.json
#   --out DIR                   save benchmark.json and the generated WAV files

# Check that repeated requests give bit-identical audio
python scripts/verify_bitexact.py --base http://127.0.0.1:7860 [--repeat 3] [--other URL] [--expect-sha SHA256]
```

## FAQ

**Does it work on other GPUs?**
Not out of the box: the kernels are tuned for the RTX 4080 SUPER, so other GPUs need some adaptation. The places to adjust:

- `GRIDS` in `flash/depth.py`: the depth megakernel runs on fixed grids of 64 and 80 persistent blocks (one per SM).
- The tuning constants at the top of `flash/csrc/depth_frame.cu` (`DD_PF_*`, `DD_S3_EARLY_PF`, `DD_FAST_TOPK`, `DD_NT`): L2 prefetch sizes and launch shape, chosen for 80 SMs and 64 MB of L2.
- The GPU check in `flash/support.py` (compute capability 8.9, at least 80 SMs and 64 MB of L2), and `scripts/build_kernels.py --arch` to build for another architecture.

Some other Ada GPUs, such as the RTX 4090 and L40S, already pass the check, but they have not been tested, and the speed-up there may differ. Benchmark results and patches for other cards are welcome.

**Can I use it commercially?**
The code in this repository is Apache-2.0. The Breeze TTS 2 model weights are not included and use the BreezeBlue Research and Non-Commercial License, which allows research and non-commercial use only. For commercial use, contact the official Breeze TTS team for a license.

**Why is the first start slow?**
On the first start the server downloads about 7.2 GB of weights, compiles four CUDA extensions (~100 s) and builds the LN12 weight cache (~10–20 s). All of this is cached. After that, a start takes about 17 s, mostly loading the model and warming up the CUDA graphs. You can compile ahead of time with `scripts/build_kernels.py`.

**How do I compare against the official runtime?**
Start the same entry point with `FLASH_DISABLE=1`. On a 16 GB card, also use the 256-token warmup profile, because the official 512-token profile does not fit:

```bash
FLASH_DISABLE=1 python scripts/serve.py --port 7861 --warmup-profile configs/fast-16gb.json
python scripts/benchmark.py --base http://127.0.0.1:7861 --out results/official
```

The two servers together need more than 16 GB, so on a 16 GB card stop one before starting the other.

**Is the API different from the official one?**
No. The routes, form fields and response format are the same. Like the official server, it handles one request at a time, and a concurrent request gets HTTP 409.

## License & Acknowledgments

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

- **Breeze TTS 2.** This project is built on the official [Breeze TTS 2 runtime](https://github.com/breezeblue-ai/breeze-tts) and model ([BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) on Hugging Face). A copy of the runtime is included under `third_party/breeze-tts/` under its own Apache-2.0 license, and the source commit is recorded in [NOTICE](NOTICE).
- **Qwen3-TTS.** The audio tokenizer comes from [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team (Apache-2.0).
- **Model weights.** The weights are not distributed with this repository. They are governed by the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE), and the Apache-2.0 license of this repository grants no rights to them.
