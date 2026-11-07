"""Codec fix: remove the data race in the official streaming codec KV cache.

models/stream_runtime/stream/kv_cache.py StaticShiftKVCache.update shifts the sliding window IN PLACE with
    layer.k[..., :-shift, :].copy_(layer.k[..., shift:, :])
Source and destination overlap inside the same buffer; the CUDA copy kernel runs its elements in parallel, so a
thread may read a slot that another thread already overwrote -> nondeterministic, corrupted KV in the codec
decoder's transformer (observed: decoding the SAME tokens twice gives PCM SNR ~27 dB, max |diff| 0.08).
Fix: shift through a separate scratch buffer (preallocated per layer, graph-capture safe). Mathematically this is
exactly what the official code intends (out[j] = in[j + shift]).
"""
import torch


def _update_fixed(self, key_states, value_states, layer_idx, cache_kwargs):
    layer = self.layers[layer_idx]
    t_new = key_states.shape[-2]
    if not torch._dynamo.is_compiling() and t_new > self.window:
        raise ValueError(f"StaticShiftKVCache assumes T_new <= window={self.window}, got {t_new}.")
    shift = min(int(t_new), self.window)
    if shift < self.window:
        scratch = getattr(layer, '_cx_scratch', None)
        if scratch is None or scratch[0].shape != layer.k.shape or scratch[0].dtype != layer.k.dtype:
            scratch = (torch.empty_like(layer.k), torch.empty_like(layer.v))
            layer._cx_scratch = scratch
        sk, sv = scratch
        sk[..., : self.window - shift, :].copy_(layer.k[..., shift:, :])
        sv[..., : self.window - shift, :].copy_(layer.v[..., shift:, :])
        layer.k[..., :-shift, :].copy_(sk[..., : self.window - shift, :])
        layer.v[..., :-shift, :].copy_(sv[..., : self.window - shift, :])
        layer.k[..., -shift:, :].copy_(key_states[..., -shift:, :])
        layer.v[..., -shift:, :].copy_(value_states[..., -shift:, :])
    else:
        layer.k.copy_(key_states[..., -self.window:, :])
        layer.v.copy_(value_states[..., -self.window:, :])
    return layer.k, layer.v


def install(api=None):
    from models.stream_runtime.stream import kv_cache
    cls = kv_cache.StaticShiftKVCache
    if getattr(cls, '_cx_installed', False):
        return
    cls.update = _update_fixed
    cls._cx_installed = True
    print('[flash] codec KV-cache overlapping-copy race fixed', flush=True)
