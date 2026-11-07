"""Host-sync-free backbone sampler, bitwise equal to the official sample_logits.

Official reference: models/cudagraph/sampling.py (sample_logits + apply_repetition_penalty, torch 2.9.1).
Same op sequence and the same elementwise arithmetic; only the three host-synchronising constructs are replaced:

  official                                         | here (no host sync)                                 | why bitwise equal
  -------------------------------------------------+-----------------------------------------------------+--------------------------
  token_history.unique(); logits[..., u] = where(.)| torch.where(present, where(x>0, x/p, x*p), x)       | same per-element ops on the
                                                   |   present = bool[1,V] presence mask of the history  | same set of ids (set = unique)
  logits[..., list(suppress)] = -inf  (2 H2D)      | logits.index_fill_(-1, suppress_idx_on_device, -inf)| writes the same -inf values
  sorted_logits[mask] = -inf  (top_p < 1 only)     | sorted_logits.masked_fill(mask, -inf)               | writes the same -inf values
  torch.multinomial(probs, 1)                      | torch.multinomial(where(ok, probs, 1.0), 1)         | the official call itself; for
                                                   |                                                     | valid probs where() is a copy

Validity flag: the official multinomial does `_assert_async(max < inf & min >= 0)` and `_assert_async(!(sum(1)==0).any())`
on the device; a failure is a device-side assert that poisons the whole CUDA context (the server must be restarted).
Here the SAME predicate is computed into a 0-dim bool tensor `ok` (no sync), multinomial gets a sanitised copy that can
never assert, and the caller copies `ok` to the host together with the token (one pinned D2H) and fails only this
request. Greedy (do_sample=False) is plain argmax in the official code and never asserts -> ok is constant True.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_INF = float('inf')


class InvalidDistribution(RuntimeError):
    """The backbone sampling distribution is invalid (nan / inf / negative / all-zero probabilities).

    The official torch.multinomial would hit a device-side assert here (CUDA context lost for every later request);
    this is raised for the current request only."""


def valid_flag(probs: torch.Tensor) -> torch.Tensor:
    """0-dim bool device tensor, exactly ATen multinomial's two checks (torch 2.9.1, n_sample == 1 path):
    is_valid = (max < INFINITY) & (min >= 0);  zero = dim==1 ? sum()==0 : (sum(1)==0).any();  ok = is_valid & ~zero."""
    is_valid = (probs.max() < _INF) & (probs.min() >= 0)
    zero = (probs.sum() == 0) if probs.dim() == 1 else (probs.sum(1) == 0).any()
    return is_valid & ~zero


def multinomial1(probs: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """torch.multinomial(probs, 1) without any host sync and without a device-side assert on invalid input:
    the official call on a copy that is bit-identical when ok, uniform (always valid) otherwise."""
    return torch.multinomial(torch.where(ok, probs, 1.0), 1)


def penalize_present(logits: torch.Tensor, present: torch.Tensor, repetition_penalty: float) -> torch.Tensor:
    """HF repetition penalty on exactly the ids marked in `present` (bool, broadcastable to logits)."""
    pen = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
    return torch.where(present, pen, logits)


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    present: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    suppress_idx: torch.Tensor | None = None,
    ok_true: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (token, ok). token has the official shape (logits.shape[:-1]); ok is a 0-dim bool device tensor.

    present      : bool presence mask of token_history (None == official token_history=None / empty history)
    suppress_idx : long device tensor of suppress_tokens (None or empty == no suppression)
    ok_true      : optional preallocated torch.tensor(True) on the device (greedy path); created if None."""
    logits = logits.clone().float()
    if present is not None and repetition_penalty != 1.0:
        logits = penalize_present(logits, present, repetition_penalty)
    if suppress_idx is not None and suppress_idx.numel() > 0:
        logits.index_fill_(-1, suppress_idx, float('-inf'))
    if not do_sample:
        if ok_true is None:
            ok_true = torch.ones((), dtype=torch.bool, device=logits.device)
        return torch.argmax(logits, dim=-1), ok_true
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove[..., :-1], (1, 0), value=False)
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter_(-1, sorted_indices, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    ok = valid_flag(probs)
    return multinomial1(probs, ok).squeeze(-1), ok
