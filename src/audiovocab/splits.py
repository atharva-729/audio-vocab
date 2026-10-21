"""Folds that never split a source recording, speaker, recordist, uploader or artist across train and test.

ESC-50 keeps its official folds (built so fragments of one Freesound file share a fold). Every other dataset gets
5 folds by a stable hash of group_id, so the assignment doesn't depend on row order or on which clips exist.
Fold `tune_fold` (config) is the tuning split for prefix/centring choices; everything is reported on the rest.
"""
import hashlib

import numpy as np
import pandas as pd

N_FOLDS = 5


def hash_fold(group_id, n=N_FOLDS):
    return int(hashlib.sha1(str(group_id).encode()).hexdigest(), 16) % n + 1


def folds(df):
    """Fold number (1..5) per row."""
    official = (df.dataset == "esc50") & df.split.astype(str).str.fullmatch(r"[1-5]")
    f = df.group_id.map(hash_fold)
    return pd.Series(np.where(official, pd.to_numeric(df.split, errors="coerce").fillna(0).astype(int), f),
                     index=df.index).astype(int)


def check_grouped(df, fold):
    """True if no group appears in more than one fold."""
    return bool((pd.DataFrame({"g": df.group_id.values, "f": np.asarray(fold)}).groupby("g").f.nunique() <= 1).all())
