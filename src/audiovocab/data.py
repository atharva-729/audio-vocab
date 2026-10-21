"""Load clips, embeddings and texts, aligned, for the analysis phases."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import embed
from .config import cfg, path
from .splits import folds

SOUND_DATASETS = ["esc50", "fsd50k", "vocalsound", "xenocanto"]


@lru_cache
def clips():
    df = pd.read_parquet(path("data") / "clips.parquet")
    df["fold"] = folds(df)
    return df


@lru_cache
def audio(model):
    """Embeddings for every clips.parquet row, from the per-clip cache (index/<model>.npy is the saved copy)."""
    df = clips()
    f, ids = path("index") / f"{model}.npy", path("index") / f"{model}_clip_ids.json"
    if f.exists() and json.loads(ids.read_text()) == df.clip_id.tolist():
        return np.load(f)
    from . import cache
    out = np.zeros((len(df), 768 if model == "gemma" else 512), np.float32)
    have = np.zeros(len(df), bool)
    for i, r in enumerate(df.to_dict("records")):
        v = cache.get_array(f"emb_{model}", embed.clip_key(r))
        if v is not None:
            out[i], have[i] = v, True
    return out, have


def subset(model, dataset):
    """(clips rows, X) for one dataset, only rows that have an embedding."""
    df = clips()
    a = audio(model)
    X, have = (a, np.ones(len(df), bool)) if isinstance(a, np.ndarray) else a
    m = (df.dataset == dataset).values & have
    return df[m].reset_index(drop=True), X[m]


@lru_cache
def texts(model, prefix="SearchQuery"):
    t = pd.read_parquet(path("index") / "text.parquet")
    name = {"clap": "text_clap", "gemma": "text_gemma" if prefix == "SearchQuery" else "text_gemma_none"}[model]
    return t, np.load(path("index") / f"{name}.npy")


def tune_fold():
    return cfg()["analysis"]["tune_fold"]
