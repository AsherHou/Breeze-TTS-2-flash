"""Backbone PREFILL attention trimming.

The official prefill graph runs eager attention against the whole 2048-slot StaticCache: scores/probs are
[B, 16, L, 2048] fp32 per layer although prefill rows can only see keys in slots [0, L) (the causal+padding
mask built by BackbonePrefillGraphCache sets every key >= L to finfo.min, whose exp underflows to exactly 0).
'bz_prefill' slices K/V/mask to the first L slots and then calls the OFFICIAL eager_attention_forward, so every
rounding point is unchanged; work and temporary memory drop by 2048/L (8x at L=256).
Only active inside BackbonePrefillGraphCache._forward (backbone prefill); decode and depth decoder untouched.
"""
import torch

NAME = 'bz_prefill'


def bz_prefill_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    L = query.shape[2]
    if (L > 1 and key.shape[2] > L and attention_mask is not None and attention_mask.dim() == 4
            and attention_mask.shape[-1] == key.shape[2]):
        key = key[:, :, :L]
        value = value[:, :, :L]
        attention_mask = attention_mask[..., :L]
    return eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                                   **kwargs)


def install(api=None):
    from transformers import AttentionInterface
    from models.cudagraph.backbone_prefill_graph import BackbonePrefillGraphCache
    if getattr(BackbonePrefillGraphCache, '_pf_installed', False):
        return
    AttentionInterface.register(NAME, bz_prefill_attention)
    orig = BackbonePrefillGraphCache._forward

    def _forward(self, *a, **k):
        cfg = self.model.layers[0].self_attn.config  # the Qwen3 layer config (not the Breeze config)
        prev = cfg._attn_implementation
        cfg._attn_implementation = NAME
        try:
            return orig(self, *a, **k)
        finally:
            cfg._attn_implementation = prev

    BackbonePrefillGraphCache._forward = _forward
    BackbonePrefillGraphCache._pf_installed = True
    print('[flash] backbone prefill attention trimmed to the prefill length', flush=True)
