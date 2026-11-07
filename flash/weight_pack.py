"""Weight packing: put the streamed decode weights into ONE contiguous allocation per model part.

Inside the official runtime the per-parameter allocations of the depth layers are spread over many GB of virtual
address space (poor TLB / page locality); the depth frame megakernel streams the same weights about 4% faster from a
contiguous copy. After load_runtime() we copy the depth-decoder and backbone layer
weights, in the order the kernels stream them, into one buffer per model part and re-point every Parameter's .data
to its view (values bitwise unchanged; old storage freed; tied parameters handled once by identity).
Must run before any CUDA graph capture / kernel pointer setup (it patches api.load_runtime).
"""
import torch

ALIGN = 128  # elements (256 bytes)


def _pack(params, device):
    seen, uniq = set(), []
    for p in params:
        if p is not None and id(p) not in seen and p.is_cuda:
            seen.add(id(p)); uniq.append(p)
    dt = uniq[0].dtype
    assert all(p.dtype == dt for p in uniq)
    total = sum((p.numel() + ALIGN - 1) // ALIGN * ALIGN for p in uniq)
    buf = torch.empty(total, dtype=dt, device=device)
    off = 0
    for p in uniq:
        n = p.numel()
        v = buf[off:off + n].view_as(p)
        v.copy_(p.data)
        p.data = v
        off += (n + ALIGN - 1) // ALIGN * ALIGN
    return buf, len(uniq), total * buf.element_size()


def _layer_params(layer):
    L = getattr(layer, '_orig_mod', layer)
    a, m = L.self_attn, L.mlp
    ps = [L.input_layernorm.weight, a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]
    for nm in ('q_norm', 'k_norm'):
        if hasattr(a, nm):
            ps.append(getattr(a, nm).weight)
    ps += [a.o_proj.weight, L.post_attention_layernorm.weight, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight]
    return ps


def pack_model(model):
    dev = next(model.parameters()).device
    out = {}
    with torch.no_grad():
        dd = model.depth_decoder.model
        ps = [dd.inputs_embeds_projector.weight]
        for layer in dd.layers:
            ps += _layer_params(layer)
        ps.append(dd.norm.weight)
        out['depth'] = _pack(ps, dev)
        bb = model.backbone_model
        ps = []
        for layer in bb.layers:
            ps += _layer_params(layer)
        ps.append(bb.norm.weight)
        out['backbone'] = _pack(ps, dev)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model._wp_buffers = {k: v[0] for k, v in out.items()}
    return {k: dict(params=v[1], bytes=v[2]) for k, v in out.items()}


def install(api):
    if getattr(api, '_wp_installed', False):
        return
    orig = api.load_runtime

    def load_runtime(*a, **k):
        tokenizer, model, audio_tokenizer = orig(*a, **k)
        info = pack_model(model)
        print(f'[flash] packed decode weights contiguously: {info}', flush=True)
        return tokenizer, model, audio_tokenizer

    api.load_runtime = load_runtime
    api._wp_installed = True
