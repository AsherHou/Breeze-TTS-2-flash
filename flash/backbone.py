"""Backbone decode step: custom kernels inside the official BackboneGraph._decode_step (captured by its CUDA graph).

Per layer: [RMSNorm + q|k|v GEMV] -> [q/k norm + RoPE + KV write + exact-length GQA flash-decoding attention]
           -> [o_proj GEMV + residual] -> [RMSNorm + gate|up GEMV + SiLU*mul] -> [down GEMV + residual]
then the final RMSNorm kernel; embedding, rotary cos/sin, the fp32 lm_head and CFG stay eager (and are captured).
Every kernel reproduces the official rounding points (see the comments in csrc/bb_attn.cu and csrc/bb_fused.cu), so
logits are bitwise identical. The attention skips masked positions in whole 64-position chunks, so its cost follows
the real sequence length instead of the 2048-slot static cache.

Only BackboneGraph._decode_step is replaced; prefill graphs (same layer modules, same StaticCache) are untouched.
Unsupported shapes / dtypes fall back to the original decode step.
"""
import torch

from . import kernels

# (R rows/warp, U unroll, warps) per batch rows M, tuned on an RTX 4080 SUPER (cold L2, CUDA-graph timed)
GEMV_CFG = {
    1: dict(qkv=(2, 1, 4), o=(1, 1, 4), gateup=(1, 1, 4), down=(2, 1, 8)),
    2: dict(qkv=(2, 1, 8), o=(1, 1, 8), gateup=(1, 1, 8), down=(1, 1, 8)),
    4: dict(qkv=(2, 1, 8), o=(1, 1, 8), gateup=(1, 1, 8), down=(1, 1, 8)),
}
_WS = {}
ORIG = {}  # original BackboneGraph methods


def _workspace(device, B, NH, NKV, S, D=128):
    key = (str(device), B, NH, NKV, S)
    w = _WS.get(key)
    if w is None:
        nch = (S + 63) // 64
        w = dict(scores=torch.empty(B, NH, S, device=device, dtype=torch.float32),
                 cmax=torch.empty(B, NH, nch, device=device, dtype=torch.float32),
                 part=torch.empty(B, NH, nch, D, device=device, dtype=torch.float32),
                 counters=torch.zeros(B, NKV, 4, device=device, dtype=torch.int32))
        _WS[key] = w
    return w


class _Layer:
    __slots__ = ('ln1_w', 'ln1_eps', 'wq', 'wk', 'wv', 'qn_w', 'kn_w', 'qk_eps', 'scaling', 'wo', 'ln2_w',
                 'ln2_eps', 'wg', 'wu', 'wd')


def _bf16_2d(t):
    return t is not None and t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous()


def _prep(graph):
    """Collect weight handles + buffers for this BackboneGraph instance; None -> unsupported, use the original."""
    st = getattr(graph, '_bb_state', None)
    if st is not None and st['B'] == graph.batch_size:
        return st
    try:
        model = graph.model
        layers = list(model.layers)
        B = graph.batch_size
        if B not in GEMV_CFG or len(layers) == 0:
            return None
        Ls = []
        for layer in layers:
            at, mlp = layer.self_attn, layer.mlp
            L = _Layer()
            L.ln1_w, L.ln1_eps = layer.input_layernorm.weight, float(layer.input_layernorm.variance_epsilon)
            L.ln2_w, L.ln2_eps = layer.post_attention_layernorm.weight, float(layer.post_attention_layernorm.variance_epsilon)
            L.wq, L.wk, L.wv, L.wo = at.q_proj.weight, at.k_proj.weight, at.v_proj.weight, at.o_proj.weight
            L.qn_w, L.kn_w = at.q_norm.weight, at.k_norm.weight
            L.qk_eps = float(at.q_norm.variance_epsilon)
            if float(at.k_norm.variance_epsilon) != L.qk_eps:
                return None
            L.scaling = float(at.scaling)
            L.wg, L.wu, L.wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
            for b_ in (at.q_proj, at.k_proj, at.v_proj, at.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                if getattr(b_, 'bias', None) is not None:
                    return None
            if not all(_bf16_2d(t) for t in (L.ln1_w, L.ln2_w, L.wq, L.wk, L.wv, L.wo, L.qn_w, L.kn_w, L.wg, L.wu, L.wd)):
                return None
            if getattr(mlp, 'act_fn', None).__class__.__name__ not in ('SiLU', 'SiLUActivation'):
                return None
            if at.head_dim != 128 or at.num_key_value_groups != 2 or getattr(at, 'sliding_window', None):
                return None
            Ls.append(L)
        H = graph.hidden_size
        I = Ls[0].wg.shape[0]
        nq, nk = Ls[0].wq.shape[0], Ls[0].wk.shape[0]
        S = graph.max_seq_len
        if S > 2048 or not _bf16_2d(model.norm.weight):
            return None
        dev = graph.device
        st = dict(B=B, layers=Ls, cfg=GEMV_CFG[B], norm_w=model.norm.weight, norm_eps=float(model.norm.variance_epsilon),
                  qkv=torch.empty(B, nq + 2 * nk, dtype=torch.bfloat16, device=dev),
                  ao=torch.empty(B, 1, nq // 128, 128, dtype=torch.bfloat16, device=dev),
                  h1=torch.empty(B, H, dtype=torch.bfloat16, device=dev),
                  act=torch.empty(B, I, dtype=torch.bfloat16, device=dev),
                  x=torch.empty(B, H, dtype=torch.bfloat16, device=dev),
                  ws=_workspace(dev, B, nq // 128, nk // 128, S))
        kernels.attn(); kernels.fused()
        graph._bb_state = st
        print(f'[flash] fused backbone decode step enabled (B={B}, layers={len(Ls)}, cfg={GEMV_CFG[B]})', flush=True)
        return st
    except Exception as e:  # pragma: no cover
        print(f'[flash] fused backbone path unavailable ({e!r}); using the original decode step', flush=True)
        return None


def fused_decode_step(graph):
    st = _prep(graph)
    if st is None:
        return ORIG['_decode_step'](graph)
    fx, ax = kernels.fused(), kernels.attn()
    B = graph.batch_size
    cfg, ws = st['cfg'], st['ws']
    qkv, ao, h1, act, xb = st['qkv'], st['ao'], st['h1'], st['act'], st['x']
    emb = graph.embed_tokens(graph.input_ids_buf)  # [B, 1, H] eager (gather + sum)
    cos, sin = graph.model.rotary_emb(emb, graph.position_ids)  # eager, identical cos/sin
    cos, sin = cos.contiguous(), sin.contiguous()
    x = emb.reshape(B, -1)
    mask = graph.attn_mask
    layers = graph.static_cache.layers
    for li, L in enumerate(st['layers']):
        kc, vc = layers[li].keys, layers[li].values
        fx.gemv_seg(x, L.ln1_w, L.ln1_eps, L.wq, L.wk, L.wv, qkv, *cfg['qkv'])
        ax.attn_decode_fused(qkv, L.qn_w, L.kn_w, L.qk_eps, cos, sin, kc, vc, mask, graph.cache_position, L.scaling,
                             ws['scores'], ws['cmax'], ws['part'], ws['counters'], ao)
        fx.gemv_resid(ao.view(B, -1), L.wo, x, h1, *cfg['o'])
        fx.gemv_gateup(h1, L.ln2_w, L.ln2_eps, L.wg, L.wu, act, *cfg['gateup'])
        fx.gemv_resid(act, L.wd, h1, xb, *cfg['down'])
        x = xb
    fx.rmsnorm(x, st['norm_w'], st['norm_eps'], graph.hidden_buf.view(B, -1))
    logits = graph.lm_head(graph.hidden_buf[:, -1, :].float())
    graph.logits_buf.copy_(logits)
    if graph.batch_size >= 2:
        cond_logits = graph.logits_buf[: graph.half]
        uncond_logits = graph.logits_buf[graph.half:]
        cfg_result = uncond_logits + graph.guidance_scale * (cond_logits - uncond_logits)
    else:
        cfg_result = graph.logits_buf[: graph.half]
    graph.cfg_logits_buf.copy_(cfg_result)


def install(api=None):
    from models.cudagraph.backbone_graph import BackboneGraph
    if '_decode_step' in ORIG:
        return
    kernels.attn(); kernels.fused()  # build/load before the runtime warms up
    ORIG['_decode_step'] = BackboneGraph._decode_step
    BackboneGraph._decode_step = fused_decode_step
    print('[flash] installed: fused backbone decode step', flush=True)
