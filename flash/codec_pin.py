"""Codec convolution algorithm pinning, with a reference recorded on this machine.

Problem: the streaming codec decoder's CUDA graph is captured while cudnn.benchmark=True, so every process -- every
THREAD, PyTorch keeps the cuDNN plan cache per thread -- times the candidate algorithms of every convolution once and
keeps the fastest. The two 1x1 Conv1d layers quantizer.rvq_first.output_proj and quantizer.rvq_rest.output_proj (TF32)
have two near-equal algorithms each, so the same tokens can decode to slightly different PCM after a restart (other
layers flip now and then too). The reference-audio ENCODER (eager, once per request with a reference) has the same
problem, per thread and per input length.

What this module does:
  1 decoder: the two quantizer output_proj layers run with cudnn.benchmark=False (cuDNN heuristics, a fixed choice),
    installed on the module instances at the START of MultiRequestStreamRuntime.__init__, i.e. before its eager probe
    decode and the codec CUDA-graph capture. Every other layer keeps benchmark=True (speed unchanged).
  2 encoder: Qwen3TTSTokenizerV2Model.encode runs with cudnn.benchmark=False (heuristics for every input length).
  3 self-check against a reference recorded on THIS machine: a fixed probe sequence (31 frames x 16 codebooks of
    deterministic pseudo-random codes) is decoded through the runtime's normal request path and the sha256 of the
    float32 PCM is compared with the reference in <FLASH_CACHE>/codec-pin/<key>.json. The key covers GPU, driver,
    torch, CUDA, cuDNN and the codec files of the model, so a different machine or upgrade records its own reference.
      - first start (no reference yet): capture up to 6 times (re-captures in fresh threads) until one PCM hash has
        been seen twice; that hash becomes the reference, the current graph is brought to it, and the reference
        encoder codes of the probe PCM are recorded as well;
      - later starts: probe once; on a mismatch drop the codec CUDA graphs and re-capture them in a NEW thread (fresh
        plan cache), probe again, at most 10 re-captures; the encoder is checked against its recorded codes.
    The check runs at the end of MultiRequestStreamRuntime.__init__ and once more after the whole warmup profile. If
    the reference cannot be reached, a warning is printed and the server keeps serving (audio is valid, only not
    bit-identical to earlier runs).
Thread-safety note: the benchmark switch is the global cuDNN flag, flipped only around the two layers' forward (graph
capture and eager use) and around one encode call. The official API serves one inference at a time (_request_lock),
so no other thread runs cuDNN convolutions while the flag is flipped.
"""
import collections
import gc
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

from . import paths

DEC_LAYERS = ('quantizer.rvq_first.output_proj', 'quantizer.rvq_rest.output_proj')
MAX_RECAPTURES = 10
FIRST_START_CAPTURES = 6
PROBE_FRAMES, PROBE_CODEBOOKS, CODEBOOK_SIZE = 31, 16, 2048
ENCODER_SR = 24000
FORMAT = 1
_LOCK = threading.RLock()
_CTX = dict(model_dir=None)
_S = dict(installed=False, status='not_run', decoder_layers=[], encoder_pinned=False, reference=None,
          reference_file=None, recorded_now=False, env=None, checks=[], recaptures=0, recapture_log=[],
          encoder_check=None, final_check=None, warning=None, runtimes=0)


def log(msg):
    print(f'[flash] codec pin: {msg}', flush=True)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _arr_sha(a):
    import numpy as np
    return _sha(np.ascontiguousarray(a).tobytes())


def configure(model_dir=None):
    _CTX['model_dir'] = str(model_dir) if model_dir else None


def probe_tokens():
    """Deterministic, library-independent probe codes [frames, codebooks] in [0, CODEBOOK_SIZE)."""
    import numpy as np
    out = np.empty((PROBE_FRAMES, PROBE_CODEBOOKS), np.int64)
    for t in range(PROBE_FRAMES):
        for c in range(PROBE_CODEBOOKS):
            h = hashlib.sha256(f'breeze-tts-2-flash codec probe {t} {c}'.encode()).digest()
            out[t, c] = int.from_bytes(h[:4], 'little') % CODEBOOK_SIZE
    return out


def _driver():
    try:
        line = Path('/proc/driver/nvidia/version').read_text().splitlines()[0]
        for tok in line.split():
            if tok.count('.') == 2 and tok.replace('.', '').isdigit():
                return tok
    except Exception:  # noqa: BLE001
        pass
    return None


def env_info():
    import torch
    d = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), driver=_driver())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        d.update(gpu=p.name, cc=f'{p.major}.{p.minor}')
    return d


def _model_fingerprint():
    """Names and sizes of the codec files and the model config (cheap; a changed checkpoint gives a new key)."""
    d = _CTX['model_dir']
    if not d:
        return None
    root = Path(d)
    files = [root / 'config.json'] + sorted((root / 'audio_tokenizer').glob('*'))
    return [(str(p.relative_to(root)), p.stat().st_size) for p in files if p.is_file()]


def _reference_path(env):
    key = _sha(json.dumps(dict(format=FORMAT, env=env, model=_model_fingerprint(),
                               probe=[PROBE_FRAMES, PROBE_CODEBOOKS, CODEBOOK_SIZE]), sort_keys=True).encode())[:16]
    return paths.cache_dir() / 'codec-pin' / f'{key}.json'


def _load_reference(path):
    try:
        d = json.loads(path.read_text())
        return d if d.get('format') == FORMAT and isinstance(d.get('decoder_pcm_sha256'), str) else None
    except Exception:  # noqa: BLE001
        return None


def _save_reference(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)


def state():
    with _LOCK:
        s = {k: (list(v) if isinstance(v, (list, collections.deque)) else v) for k, v in _S.items()}
    s['ok'] = True if s['status'] == 'pass' else False if s['status'] in ('mismatch', 'error') else None
    return s


def _warn(msg):
    _S['warning'] = msg
    log(f'WARNING {msg}')


# ------------------------------------------------------------------------------------------------ algorithm pinning
def _bench_off(fn, tag):
    """Run fn with cudnn.benchmark=False (cuDNN heuristics), restore the previous value afterwards."""
    import torch

    def wrapped(*a, **k):
        prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            return fn(*a, **k)
        finally:
            torch.backends.cudnn.benchmark = prev
    wrapped._codec_pin = tag
    wrapped.__wrapped__ = fn
    return wrapped


def pin_decoder_layers(decoder):
    """Wrap the forward of the two quantizer output_proj instances (idempotent)."""
    import torch
    mods = dict(decoder.named_modules())
    done = []
    for name in DEC_LAYERS:
        m = mods.get(name)
        if not isinstance(m, torch.nn.Conv1d) or tuple(m.kernel_size) != (1,):
            raise RuntimeError(f'[flash] codec pin: {name}: expected a 1x1 Conv1d, found {type(m).__name__}')
        cur = m.__dict__.get('forward')
        if cur is None or getattr(cur, '_codec_pin', None) is None:
            m.forward = _bench_off(m.forward, name)
        done.append(name)
    _S['decoder_layers'] = done
    return done


def pin_encoder():
    """Qwen3TTSTokenizerV2Model.encode (the reference-audio encoder, 12 Hz tokenizer) under cuDNN heuristics."""
    from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as mod
    cls = mod.Qwen3TTSTokenizerV2Model
    if getattr(cls.encode, '_codec_pin', None) is None:
        cls.encode = _bench_off(cls.encode, 'encoder')
    _S['encoder_pinned'] = True


# ------------------------------------------------------------------------------------------------ self-check
def _decode(rt, tokens, tag):
    import torch
    dev = rt.config.device
    parts = []
    with torch.inference_mode():
        rid = f'codec-pin-{tag}-{time.time_ns()}'
        rt.open_request(rid, reset=True, is_first_decode=True)
        try:
            cd = torch.from_numpy(tokens).to(dev)
            for t in range(cd.shape[0]):
                out = rt.decode_request_chunk(rid, cd[t].view(1, cd.shape[1], 1), reset=(t == 0))
                parts.append(out.reshape(-1).float().cpu())
        finally:
            rt.close_request(rid)
        torch.cuda.synchronize(dev)
    return torch.cat(parts).numpy()


def _probe(rt, why):
    """Decode the probe; returns (sha256 of the float32 PCM, PCM)."""
    t0 = time.perf_counter()
    pcm = _decode(rt, probe_tokens(), why.replace(':', '-')).astype('float32')
    sha = _arr_sha(pcm)
    ref = (_S['reference'] or {}).get('decoder_pcm_sha256')
    rec = dict(why=why, sha16=sha[:16], equal=(sha == ref) if ref else None, samples=int(pcm.shape[0]),
               ms=round((time.perf_counter() - t0) * 1e3, 1), thread=threading.current_thread().name)
    _S['checks'].append(rec)
    log(f'probe {why}: sha {sha[:16]}' + ('' if ref is None else (' = reference' if sha == ref else ' != reference '
                                                                   + ref[:16])) + f' ({rec["ms"]} ms)')
    return sha, pcm


def _recapture_new_thread(rt, label):
    """Drop the lane CUDA graphs and capture them again in a fresh thread (fresh per-thread cuDNN plan cache); same
    steps as MultiRequestStreamRuntime._maybe_warmup_fast_codec. Each re-capture keeps a few MiB more allocated
    (cuBLAS workspace of the new thread's handle and capture stream)."""
    import torch
    box = dict(label=label)
    dev = rt.config.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()

    def body():
        try:
            t0 = time.perf_counter()
            torch.cuda.set_device(idx)
            with torch.inference_mode():
                torch.cuda.synchronize(dev)
                for lane in rt.lanes:
                    old = lane.cuda_graph
                    lane.cuda_graph = None
                    lane.cuda_graph_out = None
                    if old is not None:
                        old.reset()                    # release the old graph and its private memory pool now
                    del old
                torch.cuda.synchronize(dev)
                gc.collect()
                torch.cuda.empty_cache()              # return the freed pool blocks (startup only, rare)
                for lane in rt.lanes:
                    lane.reset_hot_state()
                    lane.warmup_cuda_graph(warmup_rounds=2)
                    lane.reset_hot_state()
                torch.cuda.synchronize(dev)
            box['captured'] = all(lane.cuda_graph is not None for lane in rt.lanes)
            box['capture_s'] = round(time.perf_counter() - t0, 3)
        except BaseException as ex:  # noqa: BLE001
            box['error'] = f'{type(ex).__name__}: {ex}'
            box['traceback'] = traceback.format_exc()[-3000:]
    th = threading.Thread(target=body, name=f'codec-pin-{label}', daemon=True)
    th.start()
    th.join()
    _S['recapture_log'].append(box)
    _S['recaptures'] += 1
    log(f're-captured the codec graph in thread {th.name}: ' + (box.get('error') or f'{box["capture_s"]} s'))
    if 'error' in box or not box.get('captured'):
        raise RuntimeError(f're-capture failed: {box.get("error") or "graph not captured"}')
    return box


def _reach(rt, phase, sha, pcm, target):
    """Re-capture in new threads until the probe hash equals target (or MAX_RECAPTURES)."""
    k = 0
    while sha != target and k < MAX_RECAPTURES:
        k += 1
        _recapture_new_thread(rt, f'{phase}-r{k}')
        sha, pcm = _probe(rt, f'{phase}:recapture{k}')
    return sha == target, k, pcm


def _encode_sha(tokenizer, pcm):
    import numpy as np
    try:
        on_cuda = next(tokenizer.model.encoder.parameters()).is_cuda
    except Exception:  # noqa: BLE001
        on_cuda = False
    if not on_cuda:
        return None
    out = tokenizer.encode(pcm.astype(np.float32), sr=ENCODER_SR)
    return _arr_sha(out.audio_codes[0].detach().cpu().numpy().astype(np.int64))


def _record_reference(rt, tokenizer, path, env):
    """First start on this machine/configuration: establish the decoder reference by majority, then record it."""
    seen = collections.Counter()
    pcms = {}
    sha, pcm = _probe(rt, 'first-start:capture0')
    seen[sha] += 1
    pcms[sha] = pcm
    n = 1
    while seen.most_common(1)[0][1] < 2 and n < FIRST_START_CAPTURES:
        _recapture_new_thread(rt, f'first-start-c{n}')
        sha, pcm = _probe(rt, f'first-start:capture{n}')
        seen[sha] += 1
        pcms.setdefault(sha, pcm)
        n += 1
    target, count = seen.most_common(1)[0]
    if count < 2:
        _S['status'] = 'mismatch'
        _warn(f'{n} codec captures gave {len(seen)} different PCM hashes; no reference recorded (outputs on this '
              f'machine may differ between restarts)')
        return
    ok, k, pcm = _reach(rt, 'first-start', sha, pcm, target)
    if not ok:
        _S['status'] = 'mismatch'
        _warn('could not bring the codec graph to the majority result; no reference recorded')
        return
    rec = dict(format=FORMAT, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), env=env,
               model=_model_fingerprint(), decoder_pcm_sha256=target, decoder_samples=int(pcms[target].shape[0]),
               captures=dict(seen), encoder_codes_sha256=None)
    try:
        rec['encoder_codes_sha256'] = _encode_sha(tokenizer, pcms[target]) if tokenizer is not None else None
    except Exception as ex:  # noqa: BLE001
        log(f'encoder reference not recorded: {type(ex).__name__}: {ex}')
    _save_reference(path, rec)
    _S['reference'] = rec
    _S['recorded_now'] = True
    _S['encoder_check'] = dict(status='recorded', codes_sha16=(rec['encoder_codes_sha256'] or '')[:16] or None)
    _S['status'] = 'pass'
    log(f'reference recorded in {path} (decoder {target[:16]}, {n} captures)')


def _check_encoder(tokenizer, pcm):
    want = (_S['reference'] or {}).get('encoder_codes_sha256')
    if tokenizer is None or not want:
        _S['encoder_check'] = dict(status='skipped')
        return
    try:
        got = _encode_sha(tokenizer, pcm)
    except Exception as ex:  # noqa: BLE001
        _S['encoder_check'] = dict(status='error', error=f'{type(ex).__name__}: {ex}'[:600])
        log(f'encoder check raised {type(ex).__name__} (reported only)')
        return
    if got is None:
        _S['encoder_check'] = dict(status='skipped', reason='encoder not on CUDA')
        return
    eq = got == want
    _S['encoder_check'] = dict(status='pass' if eq else 'mismatch', codes_sha16=got[:16], reference_sha16=want[:16])
    if not eq:
        _warn('reference-audio encoder codes differ from the recorded reference (reported only)')


def startup(rt, tokenizer=None):
    """Called at the end of MultiRequestStreamRuntime.__init__ (graph captured, nothing else used the codec yet)."""
    with _LOCK:
        _S['runtimes'] += 1
        try:
            env = env_info()
            _S['env'] = env
            path = _reference_path(env)
            _S['reference_file'] = str(path)
            _S['reference'] = _load_reference(path)
            if _S['reference'] is None:
                _record_reference(rt, tokenizer, path, env)
                return
            target = _S['reference']['decoder_pcm_sha256']
            sha, pcm = _probe(rt, 'startup:capture0')
            ok, k, pcm = _reach(rt, 'startup', sha, pcm, target)
            if ok:
                _S['status'] = 'pass'
                _check_encoder(tokenizer, pcm)
            else:
                _S['status'] = 'mismatch'
                _warn(f'codec PCM still differs from the reference after {k} re-captures (serving anyway)')
        except Exception as ex:  # noqa: BLE001
            _S['status'] = 'error'
            _warn(f'self-check error {type(ex).__name__}: {ex}')
            _S['error_traceback'] = traceback.format_exc()[-3000:]
        finally:
            log(f'startup: status={_S["status"]} re-captures={_S["recaptures"]}')


def final(fs_runtime):
    """After the whole warmup profile: probe once more (re-capture loop if needed)."""
    with _LOCK:
        rt = getattr(fs_runtime, '_codec_runtime', None)
        if rt is None or _S['status'] != 'pass':
            _S['final_check'] = dict(skipped=True, status=_S['status'], runtime=rt is not None)
            return
        try:
            sha, pcm = _probe(rt, 'final:after-warmup')
            ok, k, _ = _reach(rt, 'final', sha, pcm, _S['reference']['decoder_pcm_sha256'])
            _S['final_check'] = dict(ok=ok, recaptures=k)
            if not ok:
                _S['status'] = 'mismatch'
                _warn(f'after the warmup profile the codec differs from the reference ({k} re-captures)')
        except Exception as ex:  # noqa: BLE001
            _S['status'] = 'error'
            _warn(f'final check error {type(ex).__name__}: {ex}')


# ------------------------------------------------------------------------------------------------ installation
def install(api):
    """Patch MultiRequestStreamRuntime.__init__ (pin before, self-check after), the encoder and the warmup."""
    with _LOCK:
        if _S['installed']:
            return
        import models.stream_runtime as sr
        cls = sr.MultiRequestStreamRuntime
        orig = cls.__init__

        def __init__(self, tokenizer, config, *a, **k):
            fast = bool(getattr(config, 'fast', False))
            if fast:
                pin_decoder_layers(tokenizer.model.decoder)     # before the eager probe and the graph capture
            orig(self, tokenizer, config, *a, **k)
            dev = getattr(self.config, 'device', None)
            if fast and int(self.config.chunk_frames) == 1 and dev is not None and dev.type == 'cuda':
                startup(self, tokenizer)
            else:
                _S['status'] = 'not_applicable'
        cls.__init__ = __init__
        pin_encoder()

        fcls = api.FastBreezeStreamingRuntime
        inner = fcls.warmup_from_profile

        def warmup_from_profile(self, profile, *a, **k):
            manifest = inner(self, profile, *a, **k)
            final(self)
            return manifest
        warmup_from_profile.__wrapped__ = inner
        fcls.warmup_from_profile = warmup_from_profile
        _S['installed'] = True
        log(f'installed: {list(DEC_LAYERS)} + reference encoder under cuDNN heuristics; probe self-check against '
            f'the reference in {paths.cache_dir() / "codec-pin"} (max {MAX_RECAPTURES} re-captures)')
