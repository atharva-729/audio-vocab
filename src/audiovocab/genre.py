"""Phase 7 (Q5 part 1): can a genre or region word find instrumental music of that genre or region?

Protocol (fixed before results):
  - analysis set: FMA tracks Whisper calls instrumental; genres/regions need >= MIN_TRACKS such tracks;
  - folds are artist-grouped (hash of artist id); the tune fold picks the word form; all else is reported on the
    other folds;
  - top-25 precision per genre word on the report folds, with a Wilson 95% interval, against the base rate;
  - a genre qualifies for Phase 8 if precision@25 >= 2 x base rate and its Wilson lower bound > base rate (Gemma);
    the qualifying genres plus the weakest genre (contrast) are fixed in reports/phase7_genre.json.
Writes reports/phase7_genre.json.
"""
import json

import numpy as np
import pandas as pd

from . import centering, data, metrics
from .config import cfg, path

MIN_TRACKS = 30
TOPK = 25


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def music_table(model):
    df, X = data.subset(model, "fma")
    flags = pd.read_parquet(path("data") / "vocal_flags.parquet")
    flags = flags[flags.dataset == "fma"].set_index("item_id")
    df = df.assign(vocal=df.item_id.map(flags.vocal), words=df.item_id.map(flags.words))
    keep = df.vocal.notna().values & ~df.pool.isin(["calib_instrumental", "calib_vocal"]).values
    return df[keep].reset_index(drop=True), X[keep]


def word_vectors(model, prefix, kind, words, form):
    t, T = data.texts(model, prefix)
    m = (t.kind == kind).values & (t.form == form).values
    sub = t[m]
    idx = [np.flatnonzero(sub.target.values == w)[0] for w in words]
    return T[m][idx]


def genre_score(X, W, j):
    """cos(clip, word j) minus mean cos(clip, the other words): the model's 'how <genre> is this?'."""
    S = X @ W.T
    return S[:, j] - np.delete(S, j, axis=1).mean(1)


def analyse_axis(model, prefix, df, X, axis, kind, A, gen):
    tax_words = sorted(df[axis].dropna().unique())
    inst = df[(df.vocal == False) & df[axis].notna()]  # noqa: E712
    counts = inst[axis].value_counts()
    words = [w for w in tax_words if counts.get(w, 0) >= MIN_TRACKS]
    m_inst = ((df.vocal == False) & df[axis].isin(words)).values  # noqa: E712
    m_voc = ((df.vocal == True) & df[axis].isin(words)).values  # noqa: E712
    y, groups, fold = df[axis].values, df.group_id.values, df.fold.values
    tune = fold == data.tune_fold()
    out = {"words": words, "counts_instrumental": {w: int(counts.get(w, 0)) for w in tax_words},
           "artists_instrumental": {w: int(inst[inst[axis] == w].group_id.nunique()) for w in tax_words},
           "chance": round(1 / max(len(words), 1), 4)}
    if len(words) < 2:
        return out
    # probe (artist-grouped CV over all folds; reported on non-tune folds)
    Xi, yi, gi, fi = X[m_inst], y[m_inst], groups[m_inst], fold[m_inst]
    p = metrics.probe_predict(Xi, yi, fi, C=A["probe_C"])
    rep = fi != data.tune_fold()
    mm, lo, hi = metrics.bootstrap_ci(p[rep] == yi[rep], gi[rep], n=A["bootstrap"])
    out["probe"] = {"acc": round(mm, 4), "ci": [round(lo, 4), round(hi, 4)], "chance": out["chance"], "n": int(rep.sum())}
    out["probe_recall"] = metrics.recall_per_class(yi[rep], p[rep])
    out["probe_confusion"] = pd.crosstab(pd.Series(yi[rep], name="t"), pd.Series(p[rep], name="p")) \
        .reindex(index=words, columns=words, fill_value=0).values.tolist()
    # zero-shot per form, raw and centred
    a_mean = X[tune & m_inst].mean(0)
    Xc = centering.center(X, a_mean)
    zs = {}
    for form in ("bare", "phrase", "sentence"):
        W = word_vectors(model, prefix, kind, words, form)
        for tag, XX, WW in (("raw", X, W), ("centred", Xc, centering.center(W, gen))):
            pred = metrics.zero_shot_predict(XX, WW, words)
            r = {}
            for subset, msk in (("tune", m_inst & tune), ("report", m_inst & ~tune), ("vocal_report", m_voc & ~tune)):
                if msk.sum():
                    a, l, h = metrics.bootstrap_ci(pred[msk] == y[msk], groups[msk], n=A["bootstrap"])
                    r[subset] = {"acc": round(a, 4), "ci": [round(l, 4), round(h, 4)], "n": int(msk.sum())}
            S = XX @ WW.T
            rk = np.argsort(-S, axis=1)
            msk = m_inst & ~tune
            top3 = np.array([y[i] in [words[j] for j in rk[i, :3]] for i in range(len(y))])
            r["top3_report"] = round(float(top3[msk].mean()), 4)
            r["recall_report"] = metrics.recall_per_class(y[msk], pred[msk])
            prec = {}
            for j, w in enumerate(words):
                cand = np.flatnonzero(msk)
                top = cand[np.argsort(-S[cand, j])[:TOPK]]
                k = int((y[top] == w).sum())
                base = float((y[cand] == w).mean())
                lo_w, hi_w = wilson(k, len(top))
                prec[w] = {"precision": round(k / len(top), 4), "wilson": [round(lo_w, 4), round(hi_w, 4)],
                           "base_rate": round(base, 4)}
            r["precision_at_25"] = prec
            r["confusion_report"] = pd.crosstab(pd.Series(y[msk], name="t"), pd.Series(pred[msk], name="p")) \
                .reindex(index=words, columns=words, fill_value=0).values.tolist()
            zs[f"{form}_{tag}"] = r
    best = max(zs, key=lambda k: zs[k]["tune"]["acc"])
    out["zero_shot"] = zs
    out["best_form"] = best
    out["gap_probe_minus_zero_shot"] = {w: round(out["probe_recall"][w] - zs[best]["recall_report"].get(w, 0.0), 4)
                                        for w in words}
    out["lyrics_effect"] = {"instrumental": zs[best]["report"], "vocal": zs[best].get("vocal_report")}
    return out


def canonical_classical(model, prefix, form_tag, genre_words, gen, fma_df, fma_X):
    mo, Xm = data.subset(model, "musopen")
    if not len(mo) or "classical" not in genre_words:
        return None
    form, tag = form_tag.rsplit("_", 1)
    W = word_vectors(model, prefix, "genre", genre_words, form)
    j = genre_words.index("classical")
    if tag == "centred":
        a_mean = fma_X[((fma_df.fold == data.tune_fold()) & (fma_df.vocal == False)).values].mean(0)  # noqa: E712
        Xm, W, fma_X = centering.center(Xm, a_mean), centering.center(W, gen), centering.center(fma_X, a_mean)
    # track-level for Musopen: mean of its windows
    items = mo.item_id.values
    Xt = np.stack([Xm[items == i].mean(0) for i in np.unique(items)])
    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
    pred = metrics.zero_shot_predict(Xt, W, genre_words)
    inst = (fma_df.vocal == False).values & fma_df.genre.isin(genre_words).values  # noqa: E712
    fpred = metrics.zero_shot_predict(fma_X[inst], W, genre_words)
    fy = fma_df.genre.values[inst]
    sc_mo = genre_score(Xt, W, j)
    sc_f = genre_score(fma_X[inst], W, j)
    return {"musopen_tracks": int(len(Xt)), "musopen_works": int(mo.group_id.nunique()),
            "musopen_predicted_classical": round(float((pred == "classical").mean()), 4),
            "musopen_predictions": pd.Series(pred).value_counts().to_dict(),
            "fma_classical_recall": round(float((fpred[fy == "classical"] == "classical").mean()), 4),
            "classical_score_mean": {"musopen": round(float(sc_mo.mean()), 4),
                                     "fma_classical": round(float(sc_f[fy == "classical"].mean()), 4),
                                     "fma_other_genres": round(float(sc_f[fy != "classical"].mean()), 4)}}


def choose_phase8(gj):
    g = gj["genre"]
    best = gj["genre"]["best_form"]
    prec = g["zero_shot"][best]["precision_at_25"]
    qual = [w for w, p in prec.items() if p["precision"] >= 2 * p["base_rate"] and p["wilson"][0] > p["base_rate"]]
    fails = sorted([w for w in prec if w not in qual], key=lambda w: prec[w]["precision"] / max(prec[w]["base_rate"], 1e-9))
    return {"qualifying": qual, "contrast_failing": fails[0] if fails else None,
            "rule": "precision@25 >= 2 x base rate and Wilson lower bound > base rate, Gemma, report folds, best form",
            "evidence": {w: prec[w] for w in prec}}


def run(models=("gemma", "clap")):
    A = cfg()["analysis"]
    res = json.loads((path("reports") / "phase4_resolution.json").read_text())
    out = {}
    for model in models:
        prefix = res.get(model, {}).get("prefix") or "SearchQuery"
        df, X = music_table(model)
        t, T = data.texts(model, prefix)
        gen = T[(t.kind == "generic").values].mean(0)
        g = analyse_axis(model, prefix, df, X, "genre", "genre", A, gen)
        r = analyse_axis(model, prefix, df, X, "region", "region", A, gen)
        out[model] = {"prefix": prefix, "genre": g, "region": r,
                      "n": {"tracks": int(len(df)), "instrumental": int((df.vocal == False).sum()),  # noqa: E712
                            "vocal": int((df.vocal == True).sum())}}  # noqa: E712
        if "best_form" in g:
            out[model]["canonical_classical"] = canonical_classical(model, prefix, g["best_form"], g["words"], gen, df, X)
        print(model, "genre probe", g.get("probe"), "best", g.get("best_form"), flush=True)
    out["phase8"] = choose_phase8(out["gemma"])
    (path("reports") / "phase7_genre.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
