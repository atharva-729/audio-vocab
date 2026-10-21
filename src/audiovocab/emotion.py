"""Phase 6 (Q4): how it's said vs what's said vs who says it, on CREMA-D (12 sentences x 6 emotions x 91 actors).

Probes (fixed C), each with splits that make the task honest:
  emotion   6 classes, train/test on different actors (actor-grouped folds)
  sentence 12 classes, train/test on different actors
  speaker  91 classes, train/test on different sentences (the same actors must appear in both)
Zero-shot emotion from emotion prompts (raw and centred). Acoustic baselines: prosody (energy, pitch, pace, voicing,
brightness) and prosody + MFCC means, in the same probes.
Writes reports/phase6_emotion.json.
"""
import json

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import centering, data, metrics
from .audio import decode
from .config import cfg, path
from .splits import hash_fold

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def acoustic_features(df):
    f = path("data") / "cremad_acoustic.parquet"
    if f.exists():
        feats = pd.read_parquet(f)
        if set(df.clip_id) <= set(feats.clip_id):
            return feats.set_index("clip_id").loc[df.clip_id].reset_index()
    rows = []
    for r in df.itertuples():
        x = decode(path("raw") / r.source, 16000)
        rms = librosa.feature.rms(y=x, frame_length=400, hop_length=160)[0]
        f0 = librosa.yin(x, fmin=65, fmax=500, sr=16000, frame_length=1024, hop_length=160)
        voiced = rms > 0.5 * np.median(rms)
        f0v = f0[voiced[:len(f0)]] if voiced[:len(f0)].any() else f0
        onsets = librosa.onset.onset_detect(y=x, sr=16000, hop_length=160, units="time")
        mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=13).mean(1)
        rows.append({"clip_id": r.clip_id, "energy_mean": float(np.log(rms.mean() + 1e-8)),
                     "energy_std": float(np.std(np.log(rms + 1e-8))), "energy_max": float(np.log(rms.max() + 1e-8)),
                     "pitch_median": float(np.median(np.log(f0v))), "pitch_iqr": float(np.subtract(*np.percentile(np.log(f0v), [75, 25]))),
                     "pitch_std": float(np.std(np.log(f0v))), "voiced_ratio": float(voiced.mean()),
                     "duration": len(x) / 16000, "onset_rate": len(onsets) / (len(x) / 16000),
                     "centroid": float(np.log(librosa.feature.spectral_centroid(y=x, sr=16000).mean() + 1)),
                     **{f"mfcc{i}": float(v) for i, v in enumerate(mfcc)}})
    feats = pd.DataFrame(rows)
    feats.to_parquet(f, index=False)
    return feats


PROSODY = ["energy_mean", "energy_std", "energy_max", "pitch_median", "pitch_iqr", "pitch_std", "voiced_ratio",
           "duration", "onset_rate", "centroid"]


def _probe(X, y, fold, groups, A):
    p = metrics.probe_predict(X, y, fold, C=A["probe_C"])
    m, lo, hi = metrics.bootstrap_ci(p == y, groups, n=A["bootstrap"])
    chance = 1 / len(np.unique(y))
    return p, {"acc": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "chance": round(chance, 4),
               "skill": round((m - chance) / (1 - chance), 4)}


def run(models=("gemma", "clap")):
    A = cfg()["analysis"]
    res = json.loads((path("reports") / "phase4_resolution.json").read_text())
    out = {}
    for model in models:
        df, X = data.subset(model, "cremad")
        actor_fold, sent_fold = df.fold.values, df.sentence.map(hash_fold).values
        y_emo, y_sent, y_spk = df.label.values, df.sentence.values, df.group_id.values
        reps = {model: X}
        if model == models[0]:
            feats = acoustic_features(df)
            pros = StandardScaler().fit_transform(feats[PROSODY].values)
            full = StandardScaler().fit_transform(feats[PROSODY + [f"mfcc{i}" for i in range(13)]].values)
            reps.update({"prosody": pros, "prosody_mfcc": full})
        for name, R in reps.items():
            r = {}
            pe, r["emotion"] = _probe(R, y_emo, actor_fold, y_spk, A)
            _, r["sentence"] = _probe(R, y_sent, actor_fold, y_spk, A)
            _, r["speaker"] = _probe(R, y_spk, sent_fold, y_sent, A)
            r["emotion_recall"] = metrics.recall_per_class(y_emo, pe)
            r["emotion_confusion"] = pd.crosstab(pd.Series(y_emo, name="true"), pd.Series(pe, name="pred")) \
                .reindex(index=EMOTIONS, columns=EMOTIONS, fill_value=0).values.tolist()
            out.setdefault(name, {}).update(r)
        # zero-shot emotion
        prefix = res.get(model, {}).get("prefix") or "SearchQuery"
        t, T = data.texts(model, prefix)
        m = (t.kind == "emotion").values | ((t.dataset == "cremad") & t.kind.isin(["class_name", "class_desc"])).values
        V = metrics.class_text_vectors(T[m], t.target[m].values, EMOTIONS)
        gen = T[(t.kind == "generic").values].mean(0)
        tune = actor_fold == data.tune_fold()
        Xc = centering.center(X, X[tune].mean(0))
        zs = {}
        for tag, XX, VV in (("raw", X, V), ("centred", Xc, centering.center(V, gen))):
            p = metrics.zero_shot_predict(XX[~tune], VV, EMOTIONS)
            mm, lo, hi = metrics.bootstrap_ci(p == y_emo[~tune], y_spk[~tune], n=A["bootstrap"])
            zs[tag] = {"acc": round(mm, 4), "ci": [round(lo, 4), round(hi, 4)],
                       "recall": metrics.recall_per_class(y_emo[~tune], p),
                       "confusion": pd.crosstab(pd.Series(y_emo[~tune], name="true"), pd.Series(p, name="pred"))
                       .reindex(index=EMOTIONS, columns=EMOTIONS, fill_value=0).values.tolist()}
        out[model]["zero_shot"] = zs
        out[model]["prefix"] = prefix
        print(model, {k: out[model][k]["acc"] for k in ("emotion", "sentence", "speaker")}, "zs", zs["raw"]["acc"], zs["centred"]["acc"], flush=True)
    out["n"] = {"clips": int(len(df)), "actors": int(df.group_id.nunique()), "sentences": int(df.sentence.nunique())}
    out["labels"] = EMOTIONS
    (path("reports") / "phase6_emotion.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
