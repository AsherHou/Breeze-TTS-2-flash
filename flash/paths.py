"""Locations used by Breeze-TTS-2-flash. Everything is configurable; nothing is hard-coded to a machine.

FLASH_CACHE      cache directory (compiled extensions, LN12 weight cache, codec reference records, torch caches);
                 default ~/.cache/breeze-tts-2-flash
FLASH_UPSTREAM   checkout of the official Breeze TTS runtime; default third_party/breeze-tts inside this package
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repository root of Breeze-TTS-2-flash
CSRC = Path(__file__).resolve().parent / 'csrc'


def cache_dir() -> Path:
    d = os.environ.get('FLASH_CACHE', '').strip()
    p = Path(d).expanduser() if d else Path.home() / '.cache' / 'breeze-tts-2-flash'
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def upstream_dir() -> Path:
    d = os.environ.get('FLASH_UPSTREAM', '').strip()
    return (Path(d).expanduser() if d else ROOT / 'third_party' / 'breeze-tts').resolve()


def disabled() -> bool:
    return os.environ.get('FLASH_DISABLE', '0').strip() == '1'
