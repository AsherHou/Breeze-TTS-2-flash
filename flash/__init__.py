"""Breeze-TTS-2-flash: lossless inference acceleration for Breeze TTS 2, installed as hooks on the official runtime.

    from breeze_infer import api
    import flash
    flash.install(api, model_dir=...)     # before api.main(); returns False (official path) if anything is unsupported

What is installed (all inside the official --fast-all CUDA-graph framework; same weights, no quantization; outputs
are deterministic run to run and differ from the official kernels only at floating-point rounding level):
  weight_pack    contiguous packing of the decode weights               (must patch api.load_runtime first)
  codec_kv_fix   overlapping-copy data race in the streaming codec KV cache
  prefill_attn   backbone prefill attention trimmed to the prefill length
  backbone       fused backbone decode step (exact-length GQA attention + fused GEMVs)
  depth          depth decoder: one frame megakernel per audio frame, lossless LN12 12-bit weights
  main_loop      streaming loop without per-frame host syncs, codec on a side stream
  ref_cache      reference-audio code cache (sha256-keyed LRU)
  codec_pin      codec convolution algorithm pinning + self-check against a reference recorded on this machine

FLASH_DISABLE=1 installs nothing (the official code path, for comparisons).
"""
import time

from . import paths, support

__version__ = '1.0.0'
_STATE = dict(enabled=None, reason=None, installed=[], kernels=None, device=None, seconds=None)


def log(msg):
    print(f'[flash] {msg}', flush=True)


def _fallback(reason):
    _STATE.update(enabled=False, reason=reason)
    log(f'acceleration OFF, using the official code path: {reason}')
    return False


def install(api, model_dir=None):
    """Install every hook on the imported breeze_infer.api module. Idempotent. Returns True if active."""
    if _STATE['enabled'] is not None:
        return _STATE['enabled']
    t0 = time.perf_counter()
    if paths.disabled():
        return _fallback('FLASH_DISABLE=1')
    for check in (lambda: support.check_upstream(paths.upstream_dir()), support.check_packages, support.check_gpu):
        reason = check()
        if reason:
            return _fallback(reason)
    _STATE['device'] = support.device_summary()
    from . import kernels
    try:
        kernels.load_all()                     # compiles on the first start (cached in FLASH_CACHE/ext)
    except Exception as ex:  # noqa: BLE001
        return _fallback(f'CUDA extensions could not be built or loaded: {type(ex).__name__}: {ex}')
    _STATE['kernels'] = kernels.info()

    from . import backbone, codec_kv_fix, codec_pin, depth, main_loop, prefill_attn, ref_cache, weight_pack
    weight_pack.install(api)
    codec_kv_fix.install(api)
    prefill_attn.install(api)
    backbone.install(api)
    depth.install(api)
    main_loop.install(api)
    ref_cache.install(api)
    codec_pin.configure(model_dir)
    codec_pin.install(api)
    _STATE.update(enabled=True, installed=['weight_pack', 'codec_kv_fix', 'prefill_attn', 'backbone', 'depth',
                                           'main_loop', 'ref_cache', 'codec_pin'],
                  seconds=round(time.perf_counter() - t0, 1))
    log(f'acceleration ON ({", ".join(_STATE["installed"])}); cache {paths.cache_dir()}')
    return True


def info():
    """Summary for the warmup manifest."""
    out = dict(version=__version__, **_STATE)
    if _STATE['enabled']:
        from . import codec_pin, depth, main_loop
        out['depth'] = depth.manifest()
        out['codec_pin'] = codec_pin.state()
        out['main_loop'] = dict(main_loop.STATS)
    return out
