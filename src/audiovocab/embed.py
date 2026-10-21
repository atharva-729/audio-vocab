"""Shared audio-text embeddings. Two backends behind one interface (copied from podcast-vibe-search embed.py):

  gemma: EmbeddingGemma 2 (16 kHz, 768-d)
  clap:  LAION-CLAP htsat-unfused (48 kHz, max 10 s input -> mean of fixed 10 s windows, 512-d)

All outputs are L2-normalised float32.
"""
import os

import numpy as np
import torch

from . import cache
from .config import model_path

torch.set_num_threads(int(os.environ.get("VOCAB_THREADS", torch.get_num_threads())))


def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


class Gemma:
    name, sr = "gemma", 16000

    def __init__(self, text_only=False):
        from sentence_transformers import SentenceTransformer
        drop = {"vision_config": None, **({"audio_config": None} if text_only else {})}
        # float32: bf16 is ~12x slower on CPUs without native support; fp16 gives NaNs (model card).
        self.m = SentenceTransformer(model_path("gemma"), device="cpu", config_kwargs=drop,
                                     model_kwargs={"dtype": torch.float32})

    def text(self, texts, prompt_name="SearchQuery"):
        return _l2(self.m.encode(list(texts), prompt_name=prompt_name, batch_size=32))

    def audio(self, clips):
        return _l2(self.m.encode([{"array": c, "sampling_rate": self.sr} for c in clips], batch_size=4))


class Clap:
    name, sr = "clap", 48000
    window_s = 10

    def __init__(self):
        from transformers import ClapModel, ClapProcessor
        self.m = ClapModel.from_pretrained(model_path("clap")).eval()
        self.p = ClapProcessor.from_pretrained(model_path("clap"))

    @staticmethod
    def _out(x):
        return x.pooler_output if hasattr(x, "pooler_output") else x

    @torch.inference_mode()
    def text(self, texts, prompt_name=None):
        inp = self.p(text=list(texts), return_tensors="pt", padding=True)
        return _l2(self._out(self.m.get_text_features(**inp)).numpy())

    @torch.inference_mode()
    def audio(self, clips):
        w = self.window_s * self.sr
        # Deterministic fixed windows instead of the processor's random truncation.
        wins = [c[i:i + w] for c in clips for i in range(0, max(len(c) - w // 2, 1), w)]
        counts = [len(range(0, max(len(c) - w // 2, 1), w)) for c in clips]
        inp = self.p(audio=wins, sampling_rate=self.sr, return_tensors="pt")
        e = _l2(self._out(self.m.get_audio_features(**inp)).numpy())
        out, i = [], 0
        for n in counts:
            out.append(e[i:i + n].mean(0))
            i += n
        return _l2(out)


BACKENDS = {"gemma": Gemma, "clap": Clap}


def load(name, **kw):
    return BACKENDS[name](**kw)


def embed_cached(model, keys, load_clip, batch=8):
    """Audio embeddings for `keys` (JSON-able dicts naming dataset, clip id and preprocessing), cached one file per
    clip under runs/cache/emb_<model>/. `load_clip(key, sr)` returns the samples for a cache miss."""
    ns = f"emb_{model.name}"
    out = [cache.get_array(ns, k) for k in keys]
    todo = [i for i, v in enumerate(out) if v is None]
    for j in range(0, len(todo), batch):
        idx = todo[j:j + batch]
        vecs = model.audio([load_clip(keys[i], model.sr) for i in idx])
        for i, v in zip(idx, vecs):
            out[i] = cache.put_array(ns, keys[i], v)
    return np.stack(out) if out else np.zeros((0, 0), np.float32)
