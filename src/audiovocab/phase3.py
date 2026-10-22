"""Phase 3 first look: counts, sanity checks, and UMAP maps (all clips by family; joint audio + text).

Sanity checks (stop if they fail):
  1. ESC-50 clips are closer to their own class than to random clips (per class, within > overall mean).
  2. An ESC-50 linear probe on the official folds is far above chance.
  3. The text "dog barking" retrieves ESC-50 dog clips more often than chance.
  4. The music genre probe with artist-grouped folds is above chance (checked in genre.py; copied here if present).
Writes reports/phase3_sanity.json and reports/phase3_umap.json (2-D coordinates for the page).
"""
import json

import numpy as np

from . import data, metrics
from .config import cfg, path

UMAP_MAX_PER_DATASET = 1200


def sanity(model):
    A = cfg()["analysis"]
    df, X = data.subset(model, "esc50")
    out = {}
    y = df.label.values
    S = X @ X.T
    np.fill_diagonal(S, np.nan)
    overall = np.nanmean(S)
    within = {c: float(np.nanmean(S[np.ix_(y == c, y == c)])) for c in np.unique(y)}
    out["1_within_class_vs_random"] = {"overall_mean_cos": round(float(overall), 4),
                                       "classes_within_gt_overall": int(sum(v > overall for v in within.values())),
                                       "n_classes": len(within), "pass": all(v > overall for v in within.values())}
    p = metrics.probe_predict(X, y, df.fold.values, C=A["probe_C"])
    m, lo, hi = metrics.bootstrap_ci(p == y, df.group_id.values, n=A["bootstrap"])
    out["2_esc50_probe_official_folds"] = {"acc": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "chance": 0.02,
                                           "pass": lo > 0.02 * 5}
    r = json.loads((path("reports") / "phase4_resolution.json").read_text()) if (path("reports") / "phase4_resolution.json").exists() else {}
    prefix = r.get(model, {}).get("prefix") or "SearchQuery"
    from . import embed
    enc = embed.load("gemma", text_only=True) if model == "gemma" else embed.load("clap")
    q = enc.text(["dog barking"], prompt_name=prefix if model == "gemma" and prefix != "none" else None)[0]
    top = np.argsort(-(X @ q))[:25]
    hits = int((y[top] == "dog").sum())
    out["3_dog_barking_top25"] = {"dog_in_top25": hits, "expected_by_chance": round(25 * float((y == "dog").mean()), 2),
                                  "pass": hits > 25 * float((y == "dog").mean()) * 3}
    g = path("reports") / "phase7_genre.json"
    if g.exists():
        gj = json.loads(g.read_text())
        pr = gj.get(model, {}).get("genre", {}).get("probe")
        if pr:
            out["4_music_genre_probe_artist_grouped"] = {**pr, "pass": pr["ci"][0] > pr["chance"]}
    return out


def umap_maps(model="gemma", seed=0):
    import umap
    df = data.clips()
    X, have = data.audio(model)
    df = df[have].copy()
    X = X[have]
    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.permutation(np.flatnonzero((df.dataset == ds).values))[:UMAP_MAX_PER_DATASET]
                           for ds in df.dataset.unique()])
    keep.sort()
    sub, Xs = df.iloc[keep], X[keep]
    r = json.loads((path("reports") / "phase4_resolution.json").read_text()) if (path("reports") / "phase4_resolution.json").exists() else {}
    t, T = data.texts(model, r.get(model, {}).get("prefix") or "SearchQuery")
    tm = t.kind.isin(["class_name"]).values & t.dataset.isin(list(df.dataset.unique())).values
    tm |= (t.kind == "genre").values & (t.form == "bare").values
    out = {"model": model, "n": int(len(sub))}
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(Xs)
    out["audio"] = {"x": np.round(emb[:, 0], 3).tolist(), "y": np.round(emb[:, 1], 3).tolist(),
                    "family": sub.family.tolist(), "dataset": sub.dataset.tolist(), "label": sub.label.astype(str).tolist()}
    J = np.concatenate([Xs, T[tm]])
    emb2 = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(J)
    n = len(Xs)
    # modality gap on the joint map: share of each text's 10 nearest map neighbours that are clips
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=11).fit(emb2).kneighbors(emb2[n:], return_distance=False)[:, 1:]
    out["joint"] = {"x": np.round(emb2[:, 0], 3).tolist(), "y": np.round(emb2[:, 1], 3).tolist(),
                    "is_text": [False] * n + [True] * int(tm.sum()),
                    "family": sub.family.tolist() + ["text"] * int(tm.sum()),
                    "label": sub.label.astype(str).tolist() + t.text[tm].tolist(),
                    "text_neighbours_that_are_clips": round(float((nn < n).mean()), 4)}
    S = Xs @ Xs.T
    out["cosines"] = {"clip_clip": round(float(S[~np.eye(n, dtype=bool)].mean()), 4),
                      "text_text": round(float((T[tm] @ T[tm].T)[~np.eye(int(tm.sum()), dtype=bool)].mean()), 4),
                      "text_clip": round(float((T[tm] @ Xs.T).mean()), 4)}
    return out


def run():
    df = data.clips()
    counts = {}
    for model in ("gemma", "clap"):
        have = data.audio(model)[1]
        counts[model] = df[have].groupby("dataset").size().to_dict()
    out = {"clips_per_dataset": df.groupby("dataset").size().to_dict(),
           "items_per_dataset": df.groupby("dataset").item_id.nunique().to_dict(),
           "embedded": counts, "sanity": {m: sanity(m) for m in ("gemma", "clap")}}
    (path("reports") / "phase3_sanity.json").write_text(json.dumps(out, indent=1, default=int), encoding="utf-8")
    (path("reports") / "phase3_umap.json").write_text(json.dumps(umap_maps("gemma")), encoding="utf-8")
    return out
