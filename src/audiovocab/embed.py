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


def clip_key(row):
    """Cache key for a clips.parquet row: dataset + clip + exact window + source; perturbations add their params."""
    return {"dataset": row["dataset"], "clip": row["clip_id"], "source": row["source"],
            "start": round(float(row["start"]), 3), "dur": round(float(row["dur"]), 3),
            **({"perturb": row["perturb"]} if row.get("perturb") else {})}


def load_window(key, sr):
    from .audio import decode
    from .config import path
    if "perturb" not in key:
        return decode(path("raw") / key["source"], sr, start=key["start"] or None, dur=key["dur"])
    from .perturb import apply, source_factor
    x = decode(path("raw") / key["source"], sr, start=key["start"] or None, dur=key["dur"] * source_factor(key["perturb"]))
    return apply(x, sr, key["perturb"], out_len=int(key["dur"] * sr))


def keep_awake():
    """Stop Windows sleeping while a long job runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass


def embed_index(name, datasets=None, batch=8):
    """Embed every clips.parquet row (cached per clip) -> index/<name>.npy + index/<name>_clip_ids.json, in row order."""
    import json
    import time

    import pandas as pd

    from .config import path
    keep_awake()
    clips = pd.read_parquet(path("data") / "clips.parquet")
    if datasets:
        clips = clips[clips.dataset.isin(datasets)]
    model = load(name)
    keys = [clip_key(r) for r in clips.to_dict("records")]
    out, t0 = [], time.time()
    for ds, idx in clips.groupby("dataset", sort=False).indices.items():
        k = [keys[i] for i in idx]
        for j in range(0, len(k), 400):
            out.append(embed_cached(model, k[j:j + 400], load_window, batch=batch))
            print(f"{name} {ds}: {min(j + 400, len(k))}/{len(k)}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    order = np.concatenate(list(clips.groupby("dataset", sort=False).indices.values()))
    emb = np.zeros((len(clips), out[0].shape[1]), np.float32)
    emb[order] = np.concatenate(out)
    tag = "" if not datasets else "_" + "_".join(datasets)
    path("index").mkdir(exist_ok=True)
    np.save(path("index") / f"{name}{tag}.npy", emb)
    (path("index") / f"{name}{tag}_clip_ids.json").write_text(json.dumps(clips.clip_id.tolist()))
    return emb.shape, time.time() - t0


def embed_cached(model, keys, load_clip, batch=8):
    """Audio embeddings for `keys` (JSON-able dicts naming dataset, clip id and preprocessing), cached one file per
    clip under runs/cache/emb_<model>/. `load_clip(key, sr)` returns the samples for a cache miss.
    Clips that can't be decoded (corrupt or near-empty files) get an all-zero row and are logged to
    runs/undecodable.jsonl; analyses treat zero rows as missing."""
    import json

    from .config import path
    ns = f"emb_{model.name}"
    dim = 768 if model.name == "gemma" else 512
    out = [cache.get_array(ns, k) for k in keys]
    todo = [i for i, v in enumerate(out) if v is None]
    for j in range(0, len(todo), batch):
        idx, clips = [], []
        for i in todo[j:j + batch]:
            try:
                x = load_clip(keys[i], model.sr)
                if len(x) < 0.25 * model.sr:
                    raise ValueError(f"only {len(x)} samples")
                idx.append(i)
                clips.append(x)
            except Exception as e:
                out[i] = np.zeros(dim, np.float32)
                with open(path("runs") / "undecodable.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"model": model.name, "key": keys[i], "error": str(e)[:200]}) + "\n")
        if clips:
            for i, v in zip(idx, model.audio(clips)):
                out[i] = cache.put_array(ns, keys[i], v)
    return np.stack(out) if out else np.zeros((0, dim), np.float32)
