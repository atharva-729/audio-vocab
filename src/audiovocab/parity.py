"""Phase 0 parity check: the copied embedding code must reproduce podcast-vibe-search's stored vectors.

Reads the sister repo by path (never writes to it):
  - 10 podcast segments vs index/audio_emb.npy (Gemma)
  - 5 transcripts vs index/text_emb.npy (Gemma text, "title: none | text: ..." with no prompt, as textemb.py did)
  - 5 segments vs the Phase 1 CLAP cache (runs/cache/phase1/clap_sample300_seed0.npy)
Also times one 20 s clip per model.
"""
import json
import time

import numpy as np
import pandas as pd

from . import embed
from .audio import decode
from .config import path

TOL = 1e-3  # max |difference| per dimension; L2-normalised vectors, float32 on the same CPU


def sister(*parts):
    return path("sister").joinpath(*parts)


def _clips(rows, sr):
    """Decode each episode whole, then slice: the sister's exact path (embed.segment_clips)."""
    eps = pd.read_parquet(sister("data", "episodes.parquet")).set_index("episode_id")
    out = {}
    for ep_id, g in rows.groupby("episode_id"):
        x = decode(sister("data", "audio_cache", "raw", eps.loc[ep_id, "audio_file"]), sr)
        for i, r in g.iterrows():
            out[i] = x[int(r.start_s * sr):int(r.end_s * sr)]
    return [out[i] for i in rows.index]


def _compare(new, old):
    d = np.abs(new - old).max(1)
    cos = (new * old).sum(1)
    return {"n": len(d), "max_abs_diff": float(d.max()), "min_cosine": float(cos.min()), "pass": bool(d.max() < TOL)}


def run():
    seg = pd.read_parquet(sister("index", "segments.parquet"))
    res = {}

    gemma = embed.load("gemma")
    idx = np.linspace(0, len(seg) - 1, 10).astype(int)
    rows = seg.iloc[idx]
    clips = _clips(rows, gemma.sr)
    new = gemma.audio(clips)
    res["gemma_audio"] = _compare(new, np.load(sister("index", "audio_emb.npy"))[idx])
    t = time.time()
    gemma.audio(clips[:1])
    res["gemma_s_per_20s_clip"] = round(time.time() - t, 2)

    tr = pd.read_parquet(sister("index", "transcripts.parquet")).set_index("segment_id")
    tidx = [i for i in np.linspace(0, len(seg) - 1, 40).astype(int) if tr.loc[seg.segment_id[i], "text"].strip()][:5]
    docs = [f"title: none | text: {tr.loc[seg.segment_id[i], 'text']}" for i in tidx]
    del gemma
    gtext = embed.load("gemma", text_only=True)
    res["gemma_text"] = _compare(gtext.text(docs, prompt_name=None), np.load(sister("index", "text_emb.npy"))[tidx])
    del gtext

    # The Phase 1 caches were built on the sister's original segment set (10 s hop, minutes 0-15), which was later
    # replaced. Rows of the Gemma cache that exactly match a current audio_emb row identify the same clip, and the
    # CLAP cache shares the Gemma cache's row order.
    clap = embed.load("clap")
    g_cache = np.load(sister("runs", "cache", "phase1", "gemma_sample300_seed0.npy"))
    c_cache = np.load(sister("runs", "cache", "phase1", "clap_sample300_seed0.npy"))
    sims = g_cache @ np.load(sister("index", "audio_emb.npy")).T
    hits = np.flatnonzero(sims.max(1) > 0.9999)[:5]
    crow = seg.iloc[sims[hits].argmax(1)]
    cclips = _clips(crow, clap.sr)
    res["clap_audio"] = _compare(clap.audio(cclips), c_cache[hits])
    t = time.time()
    clap.audio(cclips[:1])
    res["clap_s_per_20s_clip"] = round(time.time() - t, 2)

    out = path("reports") / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
