"""Reference-audio code cache.

Every request with a reference audio re-encodes it with the codec ENCODER (~18 ms, eager cuDNN convolutions), and
clients tend to send the same few reference clips again and again. This caches
breeze_infer.audio.encode_prompt_audio results keyed by sha256(file bytes) (LRU, 128 entries). Returned tensors are
clones of the cached CPU int16 codes, so callers can't mutate the cache. Side benefit: within one process a given
reference always maps to the same codes (the official encoder's cuDNN plan choice can differ by thread).
"""
import hashlib
import threading
from collections import OrderedDict

_LOCK = threading.Lock()
_CACHE = OrderedDict()
_MAX = 128
STATS = {'hits': 0, 'misses': 0}


def _cached(orig):
    def encode_prompt_audio(audio_tokenizer, audio_path):
        with open(audio_path, 'rb') as f:
            key = (id(audio_tokenizer), hashlib.sha256(f.read()).hexdigest())
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                _CACHE.move_to_end(key)
                STATS['hits'] += 1
                return hit.clone()
        codes = orig(audio_tokenizer, audio_path)
        with _LOCK:
            _CACHE[key] = codes.clone()
            _CACHE.move_to_end(key)
            while len(_CACHE) > _MAX:
                _CACHE.popitem(last=False)
            STATS['misses'] += 1
        return codes
    encode_prompt_audio._rc_wrapped = True
    return encode_prompt_audio


def install(api=None):
    import breeze_infer.audio as audio
    import breeze_infer.templates as templates
    if getattr(audio.encode_prompt_audio, '_rc_wrapped', False):
        return
    wrapped = _cached(audio.encode_prompt_audio)
    audio.encode_prompt_audio = wrapped
    templates.encode_prompt_audio = wrapped
    print('[flash] reference-audio code cache installed', flush=True)
