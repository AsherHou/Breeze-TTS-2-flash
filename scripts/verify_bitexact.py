#!/usr/bin/env python3
"""Bit-exactness check over HTTP: the same request several times against one server, and optionally against a
second server (e.g. another Breeze-TTS-2-flash instance, one started with FLASH_DISABLE=1, or the official
`python -m breeze_infer.api ... --fast-all`).

Note: flash vs. the official runtime is NOT expected to be bit-identical. The fused kernels sum in a different
order, so the audio differs at floating-point level while staying perceptually equivalent. Expect identical
results for repeated requests to one flash server and between two flash servers on the same machine.

Compares the sha256 of the returned 16-bit PCM. If the servers were started with BREEZE_DEBUG_TOKEN_HASH=1 (an
official switch that prints `breeze token trace: ... sha256=<tokens>` per request) and their logs are given with
--log / --other-log, the token-sequence hashes are compared as well.
  python scripts/verify_bitexact.py --base http://127.0.0.1:7860 --other http://127.0.0.1:7861
Exit code 0 = everything identical (and equal to --expect-sha if given), 1 = a difference.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TRACE = re.compile(r'breeze token trace: request_id=(\S+) seed=(\S+) frames=(\d+) sha256=([0-9a-f]{64})')


def speech(base, text, instruction, seed, cfg):
    payload = dict(text=text, cfg_scale=cfg, seed=seed)
    if instruction:
        payload['instruction'] = instruction
    req = urllib.request.Request(base.rstrip('/') + '/v1/audio/speech', data=urllib.parse.urlencode(payload).encode())
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        rate = int(r.headers['X-Sample-Rate'])
        pcm = r.read()
    return pcm, rate, time.perf_counter() - t0


def _log_size(path):
    return Path(path).stat().st_size if path and Path(path).is_file() else 0


def _token_trace(path, offset):
    """Last token trace printed into `path` after byte `offset` (the server prints it when the request ends)."""
    if not path:
        return None
    for _ in range(20):
        data = Path(path).read_bytes()[offset:].decode('utf-8', 'replace')
        m = TRACE.findall(data)
        if m:
            return dict(frames=int(m[-1][2]), sha256=m[-1][3])
        time.sleep(0.25)
    return None


def run(base, log, n, a):
    rows = []
    for i in range(n):
        off = _log_size(log)
        pcm, rate, secs = speech(base, a.text, a.instruction, a.seed, a.cfg)
        row = dict(server=base, i=i, pcm_sha256=hashlib.sha256(pcm).hexdigest(), samples=len(pcm) // 2,
                   audio_s=round(len(pcm) / 2 / rate, 3), total_s=round(secs, 3))
        tok = _token_trace(log, off)
        if tok:
            row['tokens'] = tok
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description='PCM / token bit-exactness check over HTTP')
    p.add_argument('--base', default='http://127.0.0.1:7860')
    p.add_argument('--other', help='second server to compare with')
    p.add_argument('--text', default='部署检查一切正常。')
    p.add_argument('--instruction', default='年轻女声，使用标准普通话，语气自然清晰，语速适中。')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cfg', type=float, default=4.0)
    p.add_argument('--repeat', type=int, default=2, help='requests per server')
    p.add_argument('--log', help='log file of --base (started with BREEZE_DEBUG_TOKEN_HASH=1)')
    p.add_argument('--other-log', help='log file of --other')
    p.add_argument('--expect-sha', help='expected PCM sha256')
    a = p.parse_args()
    rows = run(a.base, a.log, a.repeat, a)
    if a.other:
        rows += run(a.other, a.other_log, a.repeat, a)
    pcm = {r['pcm_sha256'] for r in rows}
    tok = {r['tokens']['sha256'] for r in rows if 'tokens' in r}
    ok = len(pcm) == 1 and len(tok) <= 1 and (not a.expect_sha or pcm == {a.expect_sha})
    print(json.dumps(dict(identical_pcm=len(pcm) == 1, pcm_sha256=sorted(pcm), identical_tokens=(len(tok) == 1)
                          if tok else None, token_sha256=sorted(tok), expect_sha=a.expect_sha, ok=ok), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
