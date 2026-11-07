#!/usr/bin/env python3
"""End-to-end HTTP benchmark of a running server (/v1/audio/speech, streaming): RTF and time to first audio.

One warm-up request, then --repeats rounds over the short / medium / long texts (seed 42 + round, cfg 4); the
summary is the median per text. RTF = wall time of the whole request / duration of the returned audio.
  python scripts/benchmark.py --base http://127.0.0.1:7860 --out results/flash
"""
import argparse
import array
import hashlib
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

CASES = {
    'short': '你好，欢迎使用语音合成服务。',
    'medium': '你好，欢迎使用语音合成服务。今天我们测试中文发音、语气和生成速度。',
    'long': '清晨，阳光透过窗帘照进房间。小林打开电脑，准备开始今天的工作。他先检查语音服务是否正常，再把一段文字转换成自然的'
            '声音。测试不仅要关注速度，也要检查发音是否清晰、停顿是否合理，让每一句话都能准确地传达给听众。',
}
INSTRUCTION = '年轻女声，使用标准普通话，语气自然清晰，语速适中。'


def speech(base, text, instruction, seed, cfg, timeout=600):
    payload = urllib.parse.urlencode(dict(text=text, instruction=instruction, cfg_scale=cfg, seed=seed)).encode()
    start = time.perf_counter()
    first = None
    chunks = []
    with urllib.request.urlopen(urllib.request.Request(base + '/v1/audio/speech', data=payload), timeout=timeout) as r:
        rate = int(r.headers['X-Sample-Rate'])
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk)
    elapsed = time.perf_counter() - start
    pcm = b''.join(chunks)
    if not pcm or len(pcm) % 2:
        raise RuntimeError(f'bad PCM stream ({len(pcm)} bytes)')
    return pcm, rate, elapsed, first


def main():
    p = argparse.ArgumentParser(description='HTTP RTF / first-audio benchmark')
    p.add_argument('--base', default='http://127.0.0.1:7860')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cases', default='short,medium,long')
    p.add_argument('--cfg', type=float, default=4.0)
    p.add_argument('--label', default='')
    p.add_argument('--out', help='directory for benchmark.json and the WAV files (optional)')
    a = p.parse_args()
    base = a.base.rstrip('/')
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    cases = a.cases.split(',')

    def run(case, seed, name):
        pcm, rate, elapsed, first = speech(base, CASES[case], INSTRUCTION, seed, a.cfg)
        samples = array.array('h', pcm)
        duration = len(samples) / rate
        rms = math.sqrt(sum(v * v for v in samples) / len(samples)) / 32768
        if out:
            with wave.open(str(out / f'{name}.wav'), 'wb') as f:
                f.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                f.writeframes(pcm)
        row = dict(case=case, seed=seed, cfg_scale=a.cfg, audio_s=duration, total_s=elapsed,
                   first_audio_ms=first * 1000, rtf=elapsed / duration, rms=rms,
                   pcm_sha256=hashlib.sha256(pcm).hexdigest())
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    run('short', 42, 'warmup')
    rows = [run(case, 42 + i, f'{case}-{i}') for i in range(a.repeats) for case in cases]
    summary = {case: {k: statistics.median(r[k] for r in rows if r['case'] == case)
                      for k in ('audio_s', 'total_s', 'first_audio_ms', 'rtf')} for case in cases}
    result = dict(label=a.label, base=base, time=datetime.now(timezone.utc).isoformat(), instruction=INSTRUCTION,
                  texts={c: CASES[c] for c in cases}, rows=rows, summary=summary)
    if out:
        (out / 'benchmark.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
