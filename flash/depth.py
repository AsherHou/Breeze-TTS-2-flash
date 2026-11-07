"""Depth decoder: ONE megakernel launch per audio frame (15 codebook steps + the official sampler), LN12 weights.

DepthDecoderGraph._full_loop is replaced (inside the official CUDA-graph capture) by
    15 x exponential_() draws  (same sizes and order as the official 15 torch.multinomial calls)
  + ONE launch of the frame megakernel (csrc/depth_frame.cu) that writes _tok_buf and output_tokens.
The kernel follows the official numerics (BF16 storage, fp32 accumulation, rounding where the official graph
materialises BF16) and the official sampling rule. Its summation order differs from the official kernels, so
results match them closely but not bit for bit; the LN12 weight format itself is lossless (decoded values are the
original BF16 bits).

Weights are read in the lossless LN12 format (flash/ln12_cache.py). The build is chosen per batch size at capture
time (each CUDA graph records one fixed kernel):
    B=1, B=2 -> flash_dd_full (evict_last prefetch + step table) on 64 blocks
    B=4      -> flash_dd_kv on 80 blocks
All builds give bitwise identical logits/tokens; they only differ in memory scheduling and block count.
Graphs with other bucket sizes (e.g. the batch endpoint's buckets of 8+) keep the official loop.

Process-wide state: the LN12 encoding and the step table are built ONCE per process and kept alive. The LN12
descriptors live in __constant__ memory of the extension (global, not per CUDA graph); re-encoding per instance (the
batch endpoint creates and drops bucket-4 graphs) would leave them pointing at freed memory for the streaming graph.
"""
import json
import time

import torch

from . import kernels, ln12_cache

BUILDS = {1: 'full', 2: 'full', 4: 'kv'}
GRIDS = {1: 64, 2: 64, 4: 80}
_state = {}


def log(msg):
    print(f'[flash] {msg}', flush=True)


def _orig(mod):
    return getattr(mod, '_orig_mod', mod)


class _DepthShim:
    """Looks like a BreezeDepthDecoderForCausalLM (model + codebooks_head) for DepthMegakernel."""

    def __init__(self, graph):
        self.model = graph.depth_model
        self.codebooks_head = graph._orig_codebooks_head


def _ln12(md, device):
    """Encode (or load from the cache) once per process; later instances must carry the same weights."""
    mats = ln12_cache.ln12_mats(md._weights, md.head_t)
    key = [m.data_ptr() for m in mats[:-15]]
    if 'ln12' in _state:
        st = _state['ln12']
        assert st['key'] == key, 'second depth megakernel with different depth weights: LN12 state would be stale'
        assert torch.equal(st['head_t'], md.head_t), 'codebook heads differ between depth megakernel instances'
        return st
    t = time.perf_counter()
    enc, how, path = ln12_cache.get(mats, device)
    st = dict(key=key, head_t=md.head_t, enc=enc, how=how, path=path, s=round(time.perf_counter() - t, 1),
              ln12_bytes=sum(x[0].numel() + x[1].numel() * 4 + x[2].numel() * 4 for x in enc),
              bf16_bytes=sum(m.numel() * 2 for m in mats))
    _state['ln12'] = st
    return st


def _init_ext(e, md, variant, ln12):
    """Initialise extension e (build `variant`) on md's weights and buffers."""
    bufs = [md.kc, md.vc, md.xres, md.qb, md.act, md.bar2, md.logits, md.obuf]
    e.dd2_init(md._weights, bufs, float(md._eps), 128 ** -0.5, int(GRIDS[2]))   # evict_last builds: L2 set-aside 0
    for b, g in GRIDS.items():
        e.dd2_set_grid(b, int(g))
    enc = ln12['enc']
    e.dd2_init_ln12([x[0] for x in enc], [x[1] for x in enc], [x[2] for x in enc],
                    torch.tensor([[x[3], x[4]] for x in enc], dtype=torch.int64))
    if variant == 'full':
        if 'tab' not in _state:                       # one table per process, shared by every table build
            t = time.perf_counter()
            pt, tt = e.dd2_tab_build()
            _state['tab'] = dict(pt=pt, tt=tt, s=round(time.perf_counter() - t, 3))
        else:
            e.dd2_tab_set(_state['tab']['pt'], _state['tab']['tt'])
    return json.loads(e.dd2_info())


class _Dispatch:
    """Per-batch-size choice of build; called at CUDA-graph capture time, so the graph records one fixed kernel."""

    def __init__(self, by_b):
        self.by_b = by_b

    def dd2_frame(self, first_cb, bh, *rest):
        return self.by_b[int(bh.size(0))].dd2_frame(first_cb, bh, *rest)


class DepthMegakernel:
    MAXB = 4

    def __init__(self, depth_decoder, device='cuda'):
        model = depth_decoder.model
        cfg = model.config
        assert cfg.hidden_size == 1024 and cfg.num_hidden_layers == 12 and cfg.intermediate_size == 8192
        assert cfg.num_attention_heads == 8 and cfg.num_key_value_heads == 2 and cfg.vocab_size == 2051
        assert getattr(model, 'backbone_hidden_state_projector', None) is None
        ws = []
        for layer in model.layers:
            L = _orig(layer)
            a, mlp = L.self_attn, L.mlp
            ws += [L.input_layernorm.weight, a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
                   L.post_attention_layernorm.weight, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]
        head = _orig(depth_decoder.codebooks_head).weight  # [15, 1024, 2051], fp32 after DepthDecoderGraph init
        head_t = head.detach().permute(0, 2, 1).contiguous().to(torch.bfloat16)  # [15, 2051, 1024]
        # lossless check: the fp32 head was cast up from bf16 checkpoint values
        assert torch.equal(head_t.float(), head.detach().float().permute(0, 2, 1)), 'head is not bf16-exact'
        self.head_t = head_t
        pos = torch.arange(17, device=device).unsqueeze(0)
        dummy = torch.zeros(1, 17, 1024, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            cos, sin = model.rotary_emb(dummy, pos)
        self.cos = cos[0].contiguous().to(torch.bfloat16)
        self.sin = sin[0].contiguous().to(torch.bfloat16)
        normf = _orig(model.norm).weight
        ws += [model.embed_tokens.weight, model.inputs_embeds_projector.weight, normf, self.head_t, self.cos, self.sin]
        self._weights = [w.detach().contiguous() for w in ws]
        for w in self._weights:
            assert w.dtype == torch.bfloat16 and w.is_cuda, w.shape
        B, M = self.MAXB, self.MAXB * 2
        bf = dict(dtype=torch.bfloat16, device=device)
        self.kc = torch.zeros(12, B, 2, 16, 128, **bf)
        self.vc = torch.zeros(12, B, 2, 16, 128, **bf)
        self.xres = torch.zeros(M, 1024, **bf)
        self.qb = torch.zeros(M, 1024, **bf)
        self.act = torch.zeros(M, 8192, **bf)
        self.logits = torch.zeros(self.MAXB, 2051, dtype=torch.float32, device=device)
        self.bar2 = torch.zeros(4, dtype=torch.int32, device=device)
        self.obuf = torch.zeros(2 * self.MAXB, 1024, dtype=torch.bfloat16, device=device)
        self._eps = float(cfg.rms_norm_eps)
        ln = _ln12(self, device)
        exts, infos = {}, {}
        for v in dict.fromkeys(BUILDS.values()):
            e = kernels.depth(v)
            infos[v] = _init_ext(e, self, v, ln)
            exts[v] = e
        if any(i['pf_last'] for i in infos.values()):
            assert all(i['persisting_l2_limit'] == 0 for i in infos.values()), infos
        self.e2 = _Dispatch({b: exts[v] for b, v in BUILDS.items()})
        info = dict(builds=dict(BUILDS), grids=dict(GRIDS), ext=infos, ln12_how=ln['how'], ln12_cache=ln['path'],
                    ln12_s=ln['s'], ln12_MiB=round(ln['ln12_bytes'] / 2 ** 20, 1),
                    bf16_MiB=round(ln['bf16_bytes'] / 2 ** 20, 1))
        if 'tab' in _state:
            info['tab_build_s'] = _state['tab']['s']
        _state['info'] = info
        log(f'depth megakernel -> {json.dumps(info)}')

    def frame(self, first_cb, bh, tokbuf, outtok, temp, topk, topp, dosample, gs, q, dbg=None, forced=None):
        self.e2.dd2_frame(first_cb, bh, tokbuf, outtok, temp, topk, topp, dosample, gs, q, dbg, forced)


def _full_loop_mk(self):
    """Drop-in replacement for DepthDecoderGraph._full_loop: 15 exponential_ draws (same sizes/order as the
    official 15 multinomial calls) + ONE frame-megakernel launch that writes _tok_buf and output_tokens."""
    B = self.batch_size
    q = self._mk2_q[B]
    for c in range(self.num_decode_codebooks):
        q[c].exponential_(1)
    self._mega2.frame(self.first_cb_token_buf, self.backbone_hidden_buf, self._tok_buf, self.output_tokens,
                      self.temperature_buf.view(-1), self.top_k_buf.view(-1), self.top_p_buf.view(-1),
                      self.do_sample_buf.view(-1), self.guidance_scale.view(-1), q, getattr(self, '_mk2_dbg', None))


def _eligible(graph):
    return all(b in (1, 2, 4) for b in (graph.bucket_sizes or [graph.batch_size]))


def attach(graph):
    if not _eligible(graph):  # e.g. the /v1/audio/speech/batch graphs (bucket 4, 8, ...): keep the official path
        graph._mega2 = None
        return graph
    if getattr(graph, '_mega2', None) is None:
        graph._mega2 = DepthMegakernel(_DepthShim(graph), device=graph.device)
        graph._mk2_q = {b: torch.zeros(15, max(1, b // 2), graph.vocab_size, dtype=torch.float32, device=graph.device)
                        for b in (1, 2, 4)}
    return graph


def install(api=None):
    from models.cudagraph import depth_decoder_graph as ddg
    cls = ddg.DepthDecoderGraph
    if getattr(cls, '_flash_depth', False):
        return
    for v in dict.fromkeys(BUILDS.values()):
        kernels.depth(v)                                   # build/load before any model work
    orig_capture = cls.capture
    orig_prepare = cls.prepare_eager
    orig_set_top_k = cls.set_top_k
    orig_full_loop = cls._full_loop

    def capture(self, num_warmup=3):
        attach(self)
        return orig_capture(self, num_warmup)

    def prepare_eager(self):
        attach(self)
        return orig_prepare(self)

    def set_top_k(self, v):
        vv = v if isinstance(v, (int, float)) else int(v.max().item())
        if not (0 <= vv <= 1024):
            raise ValueError('top_k out of range for megakernel sampler')
        return orig_set_top_k(self, v)

    def full_loop(self):
        if getattr(self, '_mega2', None) is not None and self.batch_size in (1, 2, 4):
            return _full_loop_mk(self)
        return orig_full_loop(self)

    cls.capture = capture
    cls.prepare_eager = prepare_eager
    cls.set_top_k = set_top_k
    cls._full_loop = full_loop
    cls._flash_depth = True
    log(f'installed: depth frame megakernel (LN12; builds {BUILDS}, grids {GRIDS}; other buckets use the official loop)')


def manifest():
    return dict(_state.get('info', {}))
