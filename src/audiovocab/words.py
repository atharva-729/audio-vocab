"""Phase 5 (Q3): words vs sounds, and the modality gap.

1. Synonym clouds: for each concept's ~10 synonyms, how tight they are in text space, how much their top-25 clips
   overlap (pooled sound clips), and how many of those clips are the right sound.
2. Structure match: Spearman between the class x class similarity of audio centroids and of class texts, with a
   permutation test; per dataset and per family (>= 4 classes).
3. Gap per family: clip <-> own-class-text cosine vs clip <-> other-class-text cosine.
4. Centring strength sweep: zero-shot accuracy per family for alpha in config; alpha chosen on the tune fold.
Writes reports/phase5_words.json.
"""
import json
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from . import centering, data, metrics
from .config import cfg, path

DATASETS = ["esc50", "fsd50k", "vocalsound", "xenocanto"]
TOPK = 25


def _prefix(model):
    r = json.loads((path("reports") / "phase4_resolution.json").read_text())
    return r.get(model, {}).get("prefix") or "SearchQuery"


def pooled(model):
    dfs, Xs = [], []
    for ds in DATASETS:
        df, X = data.subset(model, ds)
        if len(df):
            dfs.append(df)
            Xs.append(X)
    import pandas as pd
    return pd.concat(dfs, ignore_index=True), np.concatenate(Xs)


def synonym_clouds(model, prefix, df, X, gen):
    t, T = data.texts(model, prefix)
    syn = cfg()["synonyms"]
    sm = (t.kind == "synonym").values
    St, Tt = t[sm].reset_index(drop=True), T[sm]
    a_mean = X[(df.fold == data.tune_fold()).values].mean(0)
    out = {}
    for tag, XX, TT in (("raw", X, Tt), ("centred", centering.center(X, a_mean), centering.center(Tt, gen))):
        res = {}
        for concept, spec in syn.items():
            idx = np.flatnonzero(St.target.values == concept)
            other = np.flatnonzero(St.target.values != concept)
            V = TT[idx]
            within = float(np.mean([V[a] @ V[b] for a, b in combinations(range(len(V)), 2)]))
            across = float((V @ TT[other].T).mean())
            tops = [set(np.argsort(-(XX @ v))[:TOPK]) for v in V]
            overlap = float(np.mean([len(a & b) / len(a | b) for a, b in combinations(tops, 2)]))
            right = df.label.isin(spec["labels"]).values
            prec = [float(right[list(s)].mean()) for s in tops]
            res[concept] = {
                "text_within": round(within, 4), "text_across": round(across, 4), "text_margin": round(within - across, 4),
                "retrieval_overlap_jaccard": round(overlap, 4), "precision_at_25_mean": round(float(np.mean(prec)), 4),
                "precision_at_25_by_word": {w: round(p, 3) for w, p in zip(St.text.values[idx], prec)},
                "base_rate": round(float(right.mean()), 4), "n_right_clips": int(right.sum())}
        out[tag] = res
    return out


def structure_match(model, prefix, n_perm=2000, seed=0):
    t, T = data.texts(model, prefix)
    rng = np.random.default_rng(seed)
    out = {}
    for ds in DATASETS:
        df, X = data.subset(model, ds)
        if not len(df):
            continue
        names = np.array(sorted(df.label.unique()))
        A = np.stack([X[df.label.values == n].mean(0) for n in names])
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        tm = (t.dataset == ds).values & t.kind.isin(["class_name", "class_desc"]).values
        V = metrics.class_text_vectors(T[tm], t.target[tm].values, names)
        fam = df.groupby("label").family.first().reindex(names).values

        def corr(sel):
            n = sel.sum()
            if n < 4:
                return None
            iu = np.triu_indices(n, 1)
            SA, SV = (A[sel] @ A[sel].T)[iu], (V[sel] @ V[sel].T)
            rho = spearmanr(SA, SV[iu]).statistic
            perm = []
            for _ in range(n_perm):
                p = rng.permutation(n)
                perm.append(spearmanr(SA, SV[np.ix_(p, p)][iu]).statistic)
            return {"rho": round(float(rho), 4), "p_perm": round(float((np.sum(np.array(perm) >= rho) + 1) / (n_perm + 1)), 4),
                    "n_classes": int(n)}
        out[ds] = {"all": corr(np.ones(len(names), bool)),
                   "families": {f: c for f in sorted(set(fam)) if (c := corr(fam == f))}}
    return out


def gap_per_family(model, prefix, df, X):
    t, T = data.texts(model, prefix)
    rows = {}
    rep = (df.fold != data.tune_fold()).values
    for ds in df.dataset.unique():
        m = (df.dataset == ds).values & rep
        names = np.array(sorted(df.label[df.dataset == ds].unique()))
        tm = (t.dataset == ds).values & t.kind.isin(["class_name", "class_desc"]).values
        V = metrics.class_text_vectors(T[tm], t.target[tm].values, names)
        S = X[m] @ V.T
        own = np.searchsorted(names, df.label.values[m])
        match = S[np.arange(len(own)), own]
        nonmatch = (S.sum(1) - match) / (len(names) - 1)
        aa = X[m] @ X[m].T
        for fam in np.unique(df.family.values[m]):
            f = df.family.values[m] == fam
            rows.setdefault(fam, []).append((match[f], nonmatch[f], aa[np.ix_(f, f)][~np.eye(f.sum(), dtype=bool)]))
    out = {}
    for fam, parts in rows.items():
        mt = np.concatenate([p[0] for p in parts])
        nm = np.concatenate([p[1] for p in parts])
        aa = np.concatenate([p[2] for p in parts])
        out[fam] = {"text_audio_matching": round(float(mt.mean()), 4), "text_audio_nonmatching": round(float(nm.mean()), 4),
                    "margin": round(float((mt - nm).mean()), 4), "audio_audio_within_family": round(float(aa.mean()), 4),
                    "gap_audio_minus_text": round(float(aa.mean() - mt.mean()), 4), "n": int(len(mt))}
    return out


def alpha_sweep(model, prefix, df, X, gen):
    t, T = data.texts(model, prefix)
    alphas = cfg()["analysis"]["alphas"]
    preds = {a: np.empty(len(df), dtype=object) for a in alphas}
    for ds in df.dataset.unique():
        m = (df.dataset == ds).values
        sub = df[m]
        names = np.array(sorted(sub.label.unique()))
        tm = (t.dataset == ds).values & t.kind.isin(["class_name", "class_desc"]).values
        V = metrics.class_text_vectors(T[tm], t.target[tm].values, names)
        a_mean = X[m][(sub.fold == data.tune_fold()).values].mean(0)
        for a in alphas:
            preds[a][m] = metrics.zero_shot_predict(centering.center(X[m], a_mean, a), centering.center(V, gen, a), names)
    y, tune = df.label.values, (df.fold == data.tune_fold()).values
    tune_acc = {a: float((preds[a][tune] == y[tune]).mean()) for a in alphas}
    best = max(alphas, key=lambda a: (tune_acc[a], -a))
    out = {"tune_acc": {str(a): round(v, 4) for a, v in tune_acc.items()}, "chosen_alpha": best, "report": {}}
    for fam in sorted(df.family.unique()):
        f = (df.family == fam).values & ~tune
        out["report"][fam] = {str(a): round(float((preds[a][f] == y[f]).mean()), 4) for a in alphas}
        out["report"][fam]["n"] = int(f.sum())
    out["report"]["all"] = {str(a): round(float((preds[a][~tune] == y[~tune]).mean()), 4) for a in alphas}
    return out


def run(models=("gemma", "clap")):
    out = {}
    for model in models:
        prefix = _prefix(model)
        df, X = pooled(model)
        t, T = data.texts(model, prefix)
        gen = T[(t.kind == "generic").values].mean(0)
        out[model] = {"prefix": prefix,
                      "synonyms": synonym_clouds(model, prefix, df, X, gen),
                      "structure": structure_match(model, prefix),
                      "gap": gap_per_family(model, prefix, df, X),
                      "alpha": alpha_sweep(model, prefix, df, X, gen)}
        print(model, "alpha", out[model]["alpha"]["chosen_alpha"], out[model]["alpha"]["report"]["all"], flush=True)
    (path("reports") / "phase5_words.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
