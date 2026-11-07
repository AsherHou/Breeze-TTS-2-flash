"""Streaming main loop without per-frame host synchronisation.

Replaces FastBreezeStreamingRuntime.iter_audio_chunks (models/fast_streaming.py, commit daa6cb2). Same kernels, same
data, same order on the main stream and the same RNG consumption order as the official loop, so tokens and PCM are
bitwise identical; what changes is when the host waits and where the codec runs.

Per frame t (steady state), official loop vs this loop:

  official host                                          this loop
  1 is_backbone_eos_token(token)  .item()      SYNC      (token id already on the host, read at 9)
  2 depth(t).run                                         2 depth(t).run
                                                         3 yield chunk(t-1) (its codec finished during sample(t));
                                                           the HTTP round trip now overlaps depth(t) on the GPU
  3 should_decode_codec_frame .all().item()    SYNC      4 exact host deduction: frame[0] == token(t) != pad id
    -> host idles until depth(t) ends                      -> cannot be a terminal pad frame (device check only if
                                                              token(t) == pad id, impossible while it is suppressed)
  4 codec(t) on the main stream, sync D2H      SYNC      5 launch codec(t) on a SIDE stream (pinned D2H + event),
                                                           right behind depth(t) in main-stream order
  5 backbone(t+1): cache_position[0]=int H2D   SYNC      6 backbone(t+1): cache_position.fill_(pos), mask from the
                   cache_position[0].item()    SYNC        host-known pos (prefill_len + step), replay
  6 yield chunk(t)                                       (moved to 3 of the next frame)
  7 sample_logits: unique()                    SYNC      7 flash.sampling.sample_logits: presence mask, device
                   logits[..., list] = -inf    SYNC x2     suppress index, validity flag; token+flag -> ONE pinned
                                                           async D2H + event
  8 is_backbone_eos_token(token)  .item()      SYNC      8 (first chunk only: wait codec(0) event, yield chunk(0))
                                                         9 wait token event (the one real data dependency: EOS
                                                           decides whether depth(t+1) is launched); flag false ->
                                                           InvalidDistribution for THIS request, CUDA context intact
codec(t) and backbone(t+1) both depend only on frame t, so running codec(t) on a side stream reorders nothing
semantically. The first chunk keeps the official main-stream codec call (reset=True) and the final / token-limit
chunk is decoded synchronously as in the official loop. Host RNG calls keep their order
(depth(t) replay, codec(t), sample(t+1) exponential_, depth(t+1) replay).
"""
import time
import uuid

import torch

import models.fast_streaming as fs
from models.fast_streaming import BackbonePrefillGraphCache, FastStreamingChunk, select_fast_cfg, \
    should_decode_codec_frame

from .sampling import InvalidDistribution, sample_logits

STATS = dict(invalid_distribution=0, pad_device_checks=0, bb_fallback=0)


# ------------------------------------------------------------------ codec(t) on the side stream
def _launch_codec_async(self, frames, request_id, timing):
    side = self._flash_side
    frame_tensor = torch.stack(frames).to(self.device, dtype=torch.long)
    codes = frame_tensor.transpose(0, 1).unsqueeze(0).contiguous()
    t0 = time.perf_counter()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        audio = self._codec().decode_request_chunk(request_id, codes, reset=False)
        a = audio
        while a.dim() > 1:
            a = a[0]
        host = torch.empty(a.shape, dtype=torch.float32, pin_memory=True)
        host.copy_(a.detach().float(), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(side)
    codes.record_stream(side)
    frame_tensor.record_stream(side)
    timing = {**timing, 'codec_launch_ms': (time.perf_counter() - t0) * 1e3, 'overlapped': True}
    return dict(host=host, ev=ev, n=len(frames), timing=timing, keep=(audio, codes))


def _finish_codec(self, pending, is_final=False):
    t0 = time.perf_counter()
    pending['ev'].synchronize()
    audio_np = pending['host'].numpy().copy()
    timing = {**pending['timing'], 'audio_wait_ms': (time.perf_counter() - t0) * 1e3}
    return FastStreamingChunk(audio=audio_np, sample_rate=self.sample_rate, codec_frames=pending['n'],
                              is_final=is_final, timing=timing)


# ------------------------------------------------------------------ per-runtime buffers (allocated once)
def _state(self):
    V = int(self._backbone_graph.cfg_logits_buf.shape[-1])
    st = getattr(self, '_flash_loop_state', None)
    if st is None or st['V'] != V:
        dev = self.device
        st = dict(V=V,
                  present=torch.zeros(1, V, dtype=torch.bool, device=dev),
                  sup=torch.tensor(tuple(self._reserved_codec_token_ids), dtype=torch.long, device=dev),
                  ok_true=torch.ones((), dtype=torch.bool, device=dev),
                  dev_out=torch.zeros(2, dtype=torch.long, device=dev),
                  pin_out=torch.zeros(2, dtype=torch.long, pin_memory=True),
                  ev_tok=torch.cuda.Event())
        st['pin_np'] = st['pin_out'].numpy()
        torch.cuda.synchronize(dev)
        self._flash_loop_state = st
    return st


def _sample(st, logits, params, present, repetition_penalty):
    """Enqueue sampling + one async D2H of (token, ok). No host sync."""
    token, ok = sample_logits(logits, present=present, repetition_penalty=repetition_penalty,
                              suppress_idx=st['sup'], ok_true=st['ok_true'], **params)
    token = token.view(1)
    st['dev_out'][0:1].copy_(token)
    st['dev_out'][1:2].copy_(ok.view(1))
    st['pin_out'].copy_(st['dev_out'], non_blocking=True)
    st['ev_tok'].record()
    return token


def _read_token(st, where):
    st['ev_tok'].synchronize()
    tok, ok = int(st['pin_np'][0]), bool(st['pin_np'][1])
    if not ok:
        STATS['invalid_distribution'] += 1
        raise InvalidDistribution(
            f'backbone sampling at {where}: probability tensor contains nan/inf/negative values or sums to 0 '
            f'(the official torch.multinomial would trigger a device-side assert here); request aborted, '
            f'CUDA context intact')
    return tok


# ------------------------------------------------------------------ backbone step without the two syncs
def _bb_run_nosync(g, input_ids, step_idx):
    """BackboneGraph.run with the position taken from the host (prefill_len + step_idx is what run() writes into
    cache_position[0] and then reads back with .item()). Same buffers, same values, same kernels, same replay."""
    pos = g._prefill_len + step_idx
    if type(pos) is not int or g.cache_position.numel() != 1:
        STATS['bb_fallback'] += 1
        return g.run(input_ids, step_idx)                      # unexpected state: official path (with its syncs)
    g.input_ids_buf.copy_(input_ids)
    g.position_ids.copy_((g._base_position + step_idx).unsqueeze(-1))
    g.cache_position.fill_(pos)                                # kernel argument, no pageable H2D copy
    g._set_attention_mask(pos)
    if g.no_graph:
        g._decode_step()
    else:
        g.graph.replay()
        if g.debug:
            g.lm_head(g.hidden_buf[:, -1, :].float())
    return g.hidden_buf, g.cfg_logits_buf


# ------------------------------------------------------------------ first chunk: codec(0) on the main stream
def _launch_codec_main(self, frames, request_id, timing):
    """Same call as the official _decode_codec_frames (main stream, reset=True), but the D2H goes to pinned memory
    with an event, so backbone(1) + sample(1) can be queued behind it before the host waits for the audio."""
    frame_tensor = torch.stack(frames).to(self.device, dtype=torch.long)
    codes = frame_tensor.transpose(0, 1).unsqueeze(0).contiguous()
    t0 = time.perf_counter()
    audio = self._codec().decode_request_chunk(request_id, codes, reset=True)
    a = audio
    while a.dim() > 1:
        a = a[0]
    host = torch.empty(a.shape, dtype=torch.float32, pin_memory=True)
    host.copy_(a.detach().float(), non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    timing = {**timing, 'codec_launch_ms': (time.perf_counter() - t0) * 1e3, 'overlapped': False}
    return dict(host=host, ev=ev, n=len(frames), timing=timing, keep=(audio, codes))


def _first_timing(chunk, t_start, ev0, ev1):
    t = {**chunk.timing, 'ttfa_internal_ms': (time.perf_counter() - t_start) * 1000.0, 'flash': True}
    if ev0 is not None and ev1 is not None:
        t['prefill_gpu_ms'] = ev0.elapsed_time(ev1)
    return FastStreamingChunk(audio=chunk.audio, sample_rate=chunk.sample_rate, codec_frames=chunk.codec_frames,
                              is_final=chunk.is_final, timing=t)


# ------------------------------------------------------------------ the loop
@torch.inference_mode()  # same decorator as the official iter_audio_chunks (the API iterates it in a threadpool)
def iter_audio_chunks(self, inputs, *, request_id=None, seed=None, token_observer=None):
    cfg = select_fast_cfg(inputs)
    branch_batch_size = 2 if cfg.mode == "single_cfg" else 1
    self._ensure_graphs(branch_batch_size, cfg.guidance_scale)
    assert self._backbone_graph is not None
    assert self._depth_decoder_graph is not None
    if getattr(self, '_flash_side', None) is None:
        self._flash_side = torch.cuda.Stream(device=self.device)
    st = _state(self)                                           # lazy one-time setup BEFORE the seed reset

    codec = self._codec()
    branch = self._build_branch_batch(inputs)
    request_id = request_id or f"local-{uuid.uuid4().hex}"
    codec.open_request(request_id, reset=True, is_first_decode=True)

    backbone_params = self._sampling_params(self.model.generation_config)
    depth_params = self._sampling_params(self.model.depth_decoder.generation_config)
    eos_id = int(self.model.config.vocab_size)                  # is_backbone_eos_token
    pad_id = int(self.model.config.codebook_pad_token_id)       # is_terminal_pad_frame
    rp = self.config.repetition_penalty
    max_new = self.config.max_new_tokens
    max_seq = self.config.max_seq_len
    bbg, ddg = self._backbone_graph, self._depth_decoder_graph
    present = st['present']                                     # == set(backbone_token_history[:step_idx + 1])
    present.zero_()
    chunk_buffer = []
    chunk_index = 0
    total_frames = 0
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    first_decode = True
    t_start = time.perf_counter()
    t_chunk = t_start
    prefill_start_event = None
    prefill_end_event = None
    if self.config.collect_timing:
        prefill_start_event = torch.cuda.Event(enable_timing=True)
        prefill_end_event = torch.cuda.Event(enable_timing=True)
        prefill_start_event.record()
    pending = None          # codec(t) on the side stream; yielded right after depth(t+1) is queued
    first_pending = None    # codec(0) on the main stream; yielded after backbone(1) + sample(1) are queued

    try:
        attention_mask = branch.attention_mask
        if self._fast_backbone_prefill:
            if self._backbone_prefill_graph is None:
                self._backbone_prefill_graph = BackbonePrefillGraphCache(self._backbone_graph, token_granularity=32)
            prefill_output = self._backbone_prefill_graph(branch.inputs_embeds, attention_mask)
            hidden = prefill_output.hidden_states
            logits = prefill_output.logits
            prefill_len = prefill_output.prefill_len
            generation_attention_mask = prefill_output.attention_mask
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            backbone_out = self.model.backbone_model(inputs_embeds=branch.inputs_embeds,
                                                     attention_mask=attention_mask, position_ids=position_ids,
                                                     past_key_values=None, cache_position=None, use_cache=True)
            hidden = backbone_out.last_hidden_state
            logits = self.model.lm_head(hidden[:, -1, :].float()).float()
            prefill_len = self._backbone_graph.prefill_kv(backbone_out.past_key_values)
            generation_attention_mask = attention_mask

        if branch.branch_batch_size == 2:
            cond_logits = logits[:1]
            uncond_logits = logits[1:]
            logits = uncond_logits + branch.cfg.guidance_scale * (cond_logits - uncond_logits)
        else:
            logits = logits[:1]
        token = _sample(st, logits, backbone_params, None, rp)  # official: token_history=None
        self._backbone_graph.set_generation_state(generation_attention_mask)
        if prefill_end_event is not None:
            prefill_end_event.record()
        tok_host = _read_token(st, 'prefill')

        for step_idx in range(max_new):
            if tok_host == eos_id:
                break
            if prefill_len + step_idx >= max_seq - 1:
                break

            if branch.branch_batch_size == 2:
                token_batch = token.repeat(2)
                depth_hidden = hidden[:, -1, :]
            else:
                token_batch = token
                depth_hidden = hidden[:1, -1, :]

            depth_tokens = ddg.run(depth_hidden, token_batch, guidance_scale=branch.cfg.guidance_scale,
                                   **depth_params)
            if pending is not None:     # chunk(t-1): its HTTP round trip overlaps depth(t)
                chunk = _finish_codec(self, pending)
                pending = None
                yield chunk
                chunk_index += 1
                t_chunk = time.perf_counter()

            frame = torch.cat([token.view(1), depth_tokens[0]], dim=0)
            if token_observer is not None:
                token_observer(frame)
            if tok_host != pad_id:      # frame[0] != pad -> not all-pad: should_decode_codec_frame is True
                decodable = True
            else:
                STATS['pad_device_checks'] += 1
                decodable = should_decode_codec_frame(frame, self.model.config)
            if decodable:
                chunk_buffer.append(frame.detach())

            reached_limit = step_idx == max_new - 1
            chunk_ready = len(chunk_buffer) >= self._codec_chunk_frames
            if chunk_ready or reached_limit:
                if chunk_buffer:
                    decode_started = time.perf_counter()
                    frames = chunk_buffer
                    chunk_buffer = []
                    total_frames += len(frames)
                    timing = {"chunk_index": chunk_index, "codec_frames": len(frames),
                              "decode_launch_ms": (decode_started - t_chunk) * 1000.0,
                              "total_frames": total_frames, "is_final": reached_limit}
                    if reached_limit:   # last frame of the token budget: official synchronous decode
                        chunk = self._decode_codec_frames(frames=frames, request_id=request_id, reset=first_decode,
                                                          is_final=True, timing=timing)
                        if chunk_index == 0:
                            chunk = _first_timing(chunk, t_start, prefill_start_event, prefill_end_event)
                        yield chunk
                        first_decode = False
                        chunk_index += 1
                        t_chunk = time.perf_counter()
                    elif first_decode:
                        first_pending = _launch_codec_main(self, frames, request_id, timing)
                        first_decode = False
                    else:
                        pending = _launch_codec_async(self, frames, request_id, timing)
                if reached_limit:
                    break

            frame_for_backbone = frame.view(1, 1, -1)
            if branch.branch_batch_size == 2:
                frame_for_backbone = frame_for_backbone.repeat(2, 1, 1)

            hidden, logits = _bb_run_nosync(bbg, frame_for_backbone, step_idx)
            logits = logits.float()
            present.index_fill_(1, token, True)                 # backbone_token_history[step_idx] = token[0]
            token = _sample(st, logits, backbone_params, present, rp)

            if first_pending is not None:   # chunk(0): leaves while backbone(1) + sample(1) run on the GPU
                chunk = _finish_codec(self, first_pending)
                first_pending = None
                if chunk_index == 0:
                    chunk = _first_timing(chunk, t_start, prefill_start_event, prefill_end_event)
                yield chunk
                chunk_index += 1
                t_chunk = time.perf_counter()

            tok_host = _read_token(st, f'decode step {step_idx}')

        if pending is not None:
            chunk = _finish_codec(self, pending)
            pending = None
            yield chunk
            chunk_index += 1
            t_chunk = time.perf_counter()

        if chunk_buffer:
            frames = chunk_buffer
            total_frames += len(frames)
            yield self._decode_codec_frames(frames=frames, request_id=request_id, reset=first_decode, is_final=True,
                                            timing={"chunk_index": chunk_index, "codec_frames": len(frames),
                                                    "decode_launch_ms": (time.perf_counter() - t_chunk) * 1000.0,
                                                    "total_frames": total_frames, "is_final": True})
    finally:
        side = getattr(self, '_flash_side', None)
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        codec.close_request(request_id)


def install(api=None):
    cls = fs.FastBreezeStreamingRuntime
    if getattr(cls, '_flash_loop', False):
        return
    cls.iter_audio_chunks = iter_audio_chunks
    cls._flash_loop = True
    print('[flash] installed: sync-free streaming main loop (codec on a side stream)', flush=True)
