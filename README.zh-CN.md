# Breeze-TTS-2-flash

**Breeze TTS 2 的无损推理加速：用定制 CUDA 内核实现低延迟的实时流式语音合成。单张 RTX 4080 SUPER 上生成速度超过实时 4 倍，首包 32–38 ms。**

[English](README.md) | **简体中文**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#环境要求)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](#环境要求)
[![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](#环境要求)
[![GPU: RTX 4080 SUPER](https://img.shields.io/badge/GPU-RTX%204080%20SUPER-76B900?logo=nvidia&logoColor=white)](#环境要求)

Breeze-TTS-2-flash 是 [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) 的推理加速包。Breeze TTS 2 是开放权重的中英文语音合成（TTS）模型。官方推理代码一字不改，只在官方 `--fast-all` 的基础上用定制 CUDA 内核替换主要计算，属于无损加速：不量化、不降精度，权重不变。在 RTX 4080 SUPER 上，它比官方 `--fast-all` **快约 2 倍**，比官方 eager 模式**快约 3.5 倍**，HTTP 接口与官方完全相同，一张消费级显卡就能跑低延迟的实时语音合成。

## 快速开始

```bash
git clone https://github.com/AsherHou/Breeze-TTS-2-flash.git
cd Breeze-TTS-2-flash
pip install -r requirements.txt
python scripts/serve.py
```

建议在 Python 3.12 虚拟环境里安装。也可以运行 `bash install.sh`：它会创建 `.venv` 并装好依赖，之后用 `.venv/bin/python scripts/serve.py` 启动。另外需要 CUDA 12.8 工具链（`nvcc`），这个用 pip 装不了，见[环境要求](#环境要求)。

服务默认监听 `http://127.0.0.1:7860`，接口和官方服务一样：`/health`、`/v1/model`、`/v1/audio/speech`、`/v1/audio/speech/batch`。启动日志里出现 `[flash] acceleration ON (...)`，就说明加速已经生效。

合成一段语音：

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F "text=你好，欢迎使用语音合成服务。" \
  -F "instruction=年轻女声，使用标准普通话，语气自然清晰，语速适中。" \
  -F "cfg_scale=4" \
  -F "seed=42" \
  -o speech.pcm

# 返回的是流式裸 PCM（16 位小端、单声道）。
# 采样率在响应头 X-Sample-Rate 里，这个模型是 24000。
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

| 表单字段 | 说明 |
|---|---|
| `text` | 要合成的文本，必填。 |
| `instruction` | 可选，用文字描述音色，比如性别、口音、语气、语速。只给这一项，就是官方的音色设计（**Voice Design**）。 |
| `cfg_scale` | CFG 引导系数，必须大于 0。默认 1.0，下文的测速用的是 4。 |
| `seed` | 随机种子，默认 42。 |
| `ref_audio` + `ref_text` | 可选，声音克隆（**Voice Clone**）用的参考音频（`-F ref_audio=@ref.wav`）和它的逐字文本，两个要一起给或者都不给。再加上 `instruction`，可以调整克隆声音的语气和语速（**Voice Direction**）。 |

**首次启动**会自动做好下面几件事，以后启动直接复用：

| 步骤 | 大致耗时 | 存放位置 |
|---|---|---|
| 下载模型（约 7.2 GB） | 1 Gbit/s 约 1–2 分钟，100 Mbit/s 约 10 分钟 | Hugging Face 缓存（`HF_HOME`） |
| 编译 4 个 CUDA 扩展 | 约 100 秒（32 核 CPU 实测） | `~/.cache/breeze-tts-2-flash/ext/` |
| 生成 LN12 权重缓存（约 530 MB） | 约 10–20 秒，以后加载约 0.8 秒 | `~/.cache/breeze-tts-2-flash/ln12/` |
| 记录编解码器自检参考值 | 多花几秒 | `~/.cache/breeze-tts-2-flash/codec-pin/` |
| 加载模型并预热（每次启动都有） | 约 17 秒 | — |

已经下载过模型的话，用 `python scripts/serve.py --model /path/to/Breeze-TTS-2` 指定本地目录。不给 `--model` 时，会从 Hugging Face 下载固定版本的 `BreezeBlue/Breeze-TTS-2`。

## 直观感受

在 RTX 4080 SUPER 上生成一段 4.2 秒的语音：

| 运行方式 | 耗时 |
|---|---|
| 官方（eager） | 约 3.4 秒 * |
| 官方（`--fast-all`） | 约 2.0 秒 * |
| **Breeze-TTS-2-flash** | **约 0.9 秒** |

\* 这两项不是直接计时得到的，而是按实测 RTF（eager 0.798，`--fast-all` 0.468）换算出来的。

## 性能对比

| | 官方（eager） | 官方（`--fast-all`） | **Breeze-TTS-2-flash** |
|---|---|---|---|
| RTF：短 / 中 / 长（越低越好） | 0.798 / 0.788 / 0.787 | 0.468 / 0.463 / 0.461 | **0.233 / 0.228 / 0.227** |
| 相当于实时的几倍（1 / RTF） | 1.3× | 2.1–2.2× | **4.3–4.4×** |
| 首包延迟 | 148–150 ms | 47–55 ms | **32–38 ms** |
| 每帧 GPU 耗时（每帧对应 80 ms 音频） | 63.2 ms | 36.5 ms | **18.2 ms** |
| 显存 | 8.4 GB（峰值） | 12.8 GB | 13.6 GB（预热后） |
| 启动预热 | 无（不用 CUDA 图） | 未测 | 7.5 秒；有缓存时从启动到可用约 17 秒 |

按 RTF 算，Breeze-TTS-2-flash 比官方 `--fast-all` 快 2.0 倍，比官方 eager 快 3.4–3.5 倍。

**测试方法：**

- **请求方式**：通过 HTTP 流式调用 `/v1/audio/speech`，短、中、长三句中文各跑 3 次，表中是中位数。`cfg_scale` 为 4，种子为 42 加轮次，音色描述固定。
- **指标定义**：RTF = 整个请求耗时 ÷ 音频时长；首包 = 收到第一块音频的时间。
- **硬件**：NVIDIA GeForce RTX 4080 SUPER 16GB。
- **预热配置**：官方 `--fast-all` 默认的预热配置（`configs/fast.json`，输入最长 512 token）在 16 GB 显卡上会显存不足（OOM），所以官方对照组用的是 `configs/fast-16gb.json`，也就是同一配置截到 256 token。Breeze-TTS-2-flash 跑的是官方 512 配置。
- **自己复现**：对着运行中的服务跑 `scripts/benchmark.py` 即可，文本和参数与上面一致。

## 做了哪些优化

所有改动都在官方 `--fast-all` 的 CUDA 图框架内完成，没有修改官方代码本身。

- **深度解码器 megakernel**：每一帧（80 ms 音频）深度解码器要依次预测 15 个码本 token，官方路径为此启动大量小 kernel。这里把整帧合成一个持久化 megakernel 一次跑完，官方的采样规则也融合在同一个 kernel 里。
- **无损 12 bit 权重（LN12）**：深度解码器的 BF16 权重改用 12 bit 格式存放，计算时实时还原成完全相同的 BF16 数值，每帧读取的权重字节数少约 25%。
- **按 GPU 调整计算布局**：第一层里只取决于上一个 token 的投影提前算好做成查表；分块大小、常驻 block 数量和 L2 预取都针对这块 GPU 做了调整。
- **主干注意力更精简**：主干解码时的注意力只算真实长度，不再按补齐后的最大长度计算。每一层融合成少数几个 kernel，归一化、矩阵向量乘、RoPE 和写 KV 缓存合在一起完成。
- **prefill 注意力裁剪**：prefill 阶段跳过被 mask 掉的位置，这些位置的输出本来就是 0。
- **主循环去掉 CPU-GPU 同步**：流式主循环不再每帧等待 GPU。编解码器放在旁路 stream 上运行，采样结果一次异步拷回。
- **修复编解码器的数据竞争**：官方流式编解码器的 KV 缓存移位用的是源和目标重叠的原地拷贝，存在数据竞争。修复后，同一组 token 的流式解码与离线解码一致性从 16–22 dB 提高到约 65 dB。
- **固定编解码器卷积算法**：编解码器卷积的算法选择固定下来，每次启动输出都一样；启动时还会和本机记录的参考值做一次自检。
- **参考音频编码缓存**：声音克隆的参考音频按文件哈希缓存编码结果，重复使用同一段参考音频时不用再跑一遍编码器。

## 质量

- **权重相同，精度不变**：模型权重与官方 checkpoint 逐字节相同，没有量化，也没有降低精度。
- **与官方输出并非逐位相同**：定制 kernel 的求和顺序和官方不一样，因此生成的音频不会和官方逐位相同。但实测结果表明质量没有下降：
  - **A/B 对比**：在 78 对同文本、同种子的 A/B 中，19 项客观指标都没有检出变差。
  - **与 FP32 的接近程度**：以 FP32 为参照，逐 token 概率分布的平均总变差（TV）距离为 0.0419，官方为 0.0430，略更接近 FP32。
- **结果可复现**：同一请求重复生成，音频逐位相同。可以自己检查：

  ```bash
  python scripts/verify_bitexact.py --base http://127.0.0.1:7860 --repeat 3
  ```

  所有响应的 PCM SHA-256 都一致时，脚本以退出码 0 结束。加上 `--other URL` 还能和另一台服务对比；如果对方跑的是官方路径（`FLASH_DISABLE=1`），结果会有细微差别，原因见上文。

## 环境要求

| | |
|---|---|
| GPU | 只在 **NVIDIA GeForce RTX 4080 SUPER**（sm_89、80 个 SM、64 MB L2、16 GB）上测试过，其他 GPU 均未测试。 |
| 显存 | 预热后约 13.6 GB。 |
| 系统 | Linux x86_64（测试环境 Ubuntu 24.04），NVIDIA 驱动需支持 CUDA 12.8。 |
| Python | 3.12，需要 Python 头文件（Debian/Ubuntu 上是 `python3-dev`）。 |
| CUDA 工具链 | **CUDA 12.8，带 `nvcc`**，首次启动编译扩展时要用。 |
| 编译器 | `gcc`/`g++`（测试版本 13.3）。`ninja` 由 `requirements.txt` 安装。 |
| PyTorch | CUDA 12.8 版的 `torch==2.9.1`，以及 `transformers==4.57.3`、`qwen-tts==0.1.1`（都已在 `requirements.txt` 中固定）。 |

**关于 `nvcc`**：pip 装不出可用的 `nvcc`。PyPI 上的 `nvidia-cuda-nvcc-cu12` 只带 `ptxas`，不带 `nvcc`，所以要从 NVIDIA 安装 CUDA 12.8 工具链，比如 NVIDIA apt 源里的 `cuda-toolkit-12-8`。程序按以下顺序查找工具链：`CUDA_HOME` / `CUDA_PATH`、`PATH` 里的 `nvcc`、`$CONDA_PREFIX`、`/usr/local/cuda-12.8`、`/usr/local/cuda`。

**关于 PyTorch**：如果 pip 装上的 torch 不是 CUDA 12.8 版，用 cu128 的源重装：

```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

**自动回退**：启动时会检查 GPU（计算能力 8.9、至少 80 个 SM、L2 至少 64 MB）、依赖包版本和附带的官方文件。任何一项不满足，或者扩展编译失败，都不会崩溃，而是打印 `[flash] acceleration OFF, using the official code path: <原因>`，然后按官方路径照常提供服务。

## 配置

### `scripts/serve.py`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model DIR` | 自动下载 `BreezeBlue/Breeze-TTS-2` | 本地模型目录。 |
| `--host` | `127.0.0.1` | 监听地址。 |
| `--port` | `7860` | 端口。 |
| `--warmup-profile JSON` | 官方 `configs/fast.json` | CUDA 图预热配置。 |
| `--cache-dir DIR` | `~/.cache/breeze-tts-2-flash` | 缓存目录，等同于 `FLASH_CACHE`。 |
| `--upstream DIR` | `third_party/breeze-tts` | 官方推理代码所在目录，等同于 `FLASH_UPSTREAM`。 |
| `--manifest FILE` | — | 把预热清单（耗时、显存、加速状态）写到这个文件。 |

### 环境变量

| 变量 | 说明 |
|---|---|
| `FLASH_CACHE` | 缓存目录，存放编译好的扩展、LN12 权重缓存、编解码器参考值，以及 torch/Triton 缓存。默认 `~/.cache/breeze-tts-2-flash`。 |
| `FLASH_UPSTREAM` | 官方推理代码的位置，默认 `third_party/breeze-tts`。 |
| `FLASH_DISABLE=1` | 不安装任何加速，直接就是官方服务，用来做对照。 |
| `HF_HOME` | 下载模型用的 Hugging Face 缓存目录。 |
| `CUDA_HOME` | CUDA 工具链位置，自动找不到时手动指定。 |

### 工具脚本

```bash
# 提前编译 CUDA 扩展（没有 GPU 也能编，架构默认 8.9）
python scripts/build_kernels.py [--arch 8.9] [--cache-dir DIR] [--verbose]

# 对运行中的服务测速：先预热 1 次，再把短/中/长三句各跑 --repeats 次，输出中位数
python scripts/benchmark.py --base http://127.0.0.1:7860 --repeats 3 --cfg 4 --out results/flash
#   --cases short,medium,long   选择要跑的句子
#   --label NAME                写进 benchmark.json 的标签
#   --out DIR                   保存 benchmark.json 和生成的 WAV

# 检查同一请求重复生成时音频是否逐位相同
python scripts/verify_bitexact.py --base http://127.0.0.1:7860 [--repeat 3] [--other URL] [--expect-sha SHA256]
```

## 常见问题

**其他 GPU 能用吗？**
不能直接用。内核是按 RTX 4080 SUPER 调的，换别的 GPU 需要自己稍微调整适配，主要改这几处：

- `flash/depth.py` 里的 `GRIDS`：深度解码器 megakernel 用固定的 64 和 80 个常驻 block（每个 SM 一个）。
- `flash/csrc/depth_frame.cu` 开头的调优常量（`DD_PF_*`、`DD_S3_EARLY_PF`、`DD_FAST_TOPK`、`DD_NT`）：L2 预取大小和启动形状，按 80 个 SM、64 MB L2 选的。
- `flash/support.py` 里的 GPU 检查（计算能力 8.9、至少 80 个 SM、L2 至少 64 MB）；换架构编译用 `scripts/build_kernels.py --arch`。

RTX 4090、L40S 等其他 Ada 架构显卡已经能通过这项检查，但没有测试过，加速效果可能不同。欢迎反馈其他显卡上的测速结果，也欢迎提交适配补丁。

**能商用吗？**
本仓库的代码采用 Apache-2.0 许可。Breeze TTS 2 的模型权重不随仓库分发，适用 BreezeBlue 研究与非商用许可（BreezeBlue Research and Non-Commercial License），只允许研究和非商业用途。商用需要联系 Breeze TTS 官方取得授权。

**首次启动为什么慢？**
首次启动要下载约 7.2 GB 的模型、编译 4 个 CUDA 扩展（约 100 秒）、生成 LN12 权重缓存（约 10–20 秒）。这些都会缓存下来，之后每次启动约 17 秒，主要花在加载模型和预热 CUDA 图上。也可以先用 `scripts/build_kernels.py` 把扩展编好。

**怎么和官方做对照？**
用同一个入口加上 `FLASH_DISABLE=1` 启动即可。16 GB 显卡上官方 512 配置放不下，要同时换成 256 token 的预热配置：

```bash
FLASH_DISABLE=1 python scripts/serve.py --port 7861 --warmup-profile configs/fast-16gb.json
python scripts/benchmark.py --base http://127.0.0.1:7861 --out results/official
```

两个服务同时运行需要超过 16 GB 显存，所以在 16 GB 显卡上要先停掉一个再启动另一个。

**接口和官方有区别吗？**
没有。路由、表单字段、返回格式都一样。和官方服务一样，同一时间只处理一个请求，并发的请求会收到 HTTP 409。

## 许可与致谢

本仓库代码采用 [Apache License 2.0](LICENSE) 许可。

- **Breeze TTS 2**：本项目基于官方 [Breeze TTS 2 推理代码](https://github.com/breezeblue-ai/breeze-tts)和 Hugging Face 上的模型 [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)。官方代码原样放在 `third_party/breeze-tts/` 下，沿用其 Apache-2.0 许可，来源 commit 记录在 [NOTICE](NOTICE) 中。
- **Qwen3-TTS**：音频 tokenizer 来自 Qwen 团队的 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache-2.0）。
- **模型权重**：权重不随本仓库分发，适用 [BreezeBlue 研究与非商用许可](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)。本仓库的 Apache-2.0 许可不授予任何使用权重的权利。
