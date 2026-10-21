"""Phase 4: resolution (Q1) and knows-vs-can-name (Q2), per dataset, family, sibling group and class, both models.

Protocol (fixed before results):
  - probe and kNN predictions come from grouped cross-validation over all 5 folds (fixed C, k);
  - zero-shot choices (Gemma prefix; centring on/off is reported both ways) are made on the tune fold only;
  - every reported number is computed on the other folds ("report rows"), with 95% cluster-bootstrap CIs over
    groups (source recording / speaker / recordist / uploader).
Writes reports/phase4_resolution.json.
"""
import json

import numpy as np

from . import centering, data, metrics
from .config import cfg, path

DATASETS = ["esc50", "fsd50k", "vocalsound", "xenocanto"]


def _acc(y, pred, groups, n):
    m, lo, hi = metrics.bootstrap_ci(pred == y, groups, n=n)
    return {"acc": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n": int(len(y)), "groups": int(len(np.unique(groups)))}


def zero_shot_setup(model, prefix, df, X):
    """Class text vectors, plus centred versions (audio mean from the tune fold, text mean from GENERIC_PHRASES)."""
    t, T = data.texts(model, prefix)
    names = np.array(sorted(df.label.unique()))
    tsub = t.dataset == df.dataset.iloc[0]
    V = metrics.class_text_vectors(T[tsub.values], t.target[tsub].values, names)
    gen = T[(t.kind == "generic").values].mean(0)
    a_mean = X[(df.fold == data.tune_fold()).values].mean(0)
    return names, V, centering.center(V, gen), centering.center(X, a_mean)


def choose_prefix(model):
    """Gemma only: zero-shot accuracy on the tune fold, SearchQuery vs no prefix, pooled over datasets."""
    if model != "gemma":
        return None, {}
    scores = {}
    for prefix in ("SearchQuery", "none"):
        hits = n = 0
        for ds in DATASETS:
            df, X = data.subset(model, ds)
            if not len(df):
                continue
            names, V, _, _ = zero_shot_setup(model, prefix, df, X)
            tune = (df.fold == data.tune_fold()).values
            hits += int((metrics.zero_shot_predict(X[tune], V, names) == df.label.values[tune]).sum())
            n += int(tune.sum())
        scores[prefix] = hits / max(n, 1)
    return max(scores, key=scores.get), scores


def analyse(model, ds, prefix, A):
    df, X = data.subset(model, ds)
    if not len(df):
        return None
    y, g, fold = df.label.values, df.group_id.values, df.fold.values
    rep = fold != data.tune_fold()
    nb = A["bootstrap"]
    probe = metrics.probe_predict(X, y, fold, C=A["probe_C"])
    knn = metrics.knn_predict(X, y, g, k=A["knn_k"])
    names, V, Vc, Xc = zero_shot_setup(model, prefix, df, X)
    zs = metrics.zero_shot_predict(X, V, names)
    zsc = metrics.zero_shot_predict(Xc, Vc, names)
    tight = metrics.tightness(X, y)

    res = {"n_clips": int(len(df)), "n_classes": int(len(names)), "chance": round(1 / len(names), 4),
           "probe": _acc(y[rep], probe[rep], g[rep], nb), "knn": _acc(y[rep], knn[rep], g[rep], nb),
           "zero_shot": _acc(y[rep], zs[rep], g[rep], nb), "zero_shot_centred": _acc(y[rep], zsc[rep], g[rep], nb)}
    fam_of = df.groupby("label").family.first().to_dict()
    res["classes"] = {
        str(c): {"family": fam_of[c], "n": int((y == c).sum()),
                 "probe": round(float((probe[rep & (y == c)] == c).mean()), 4),
                 "knn": round(float((knn[rep & (y == c)] == c).mean()), 4),
                 "zero_shot": round(float((zs[rep & (y == c)] == c).mean()), 4),
                 "zero_shot_centred": round(float((zsc[rep & (y == c)] == c).mean()), 4),
                 **{k: round(v, 4) if isinstance(v, float) else v for k, v in tight[c].items()}}
        for c in names}

    # restricted problems: families and sibling groups (classes of the subset only)
    def restricted(classes):
        m = np.isin(y, classes)
        sub_names = np.array(sorted(classes))
        Vi = V[np.searchsorted(names, sub_names)]
        Vci = Vc[np.searchsorted(names, sub_names)]
        p = metrics.probe_predict(X[m], y[m], fold[m], C=A["probe_C"])
        k = metrics.knn_predict(X[m], y[m], g[m], k=A["knn_k"])
        z = metrics.zero_shot_predict(X[m], Vi, sub_names)
        zc = metrics.zero_shot_predict(Xc[m], Vci, sub_names)
        r = rep[m]
        out = {"classes": [str(c) for c in sub_names], "chance": round(1 / len(sub_names), 4),
               "probe": _acc(y[m][r], p[r], g[m][r], nb), "knn": _acc(y[m][r], k[r], g[m][r], nb),
               "zero_shot": _acc(y[m][r], z[r], g[m][r], nb), "zero_shot_centred": _acc(y[m][r], zc[r], g[m][r], nb),
               "per_class": {str(c): {"probe": round(float((p[r & (y[m] == c)] == c).mean()), 4),
                                      "zero_shot": round(float((z[r & (y[m] == c)] == c).mean()), 4)}
                             for c in sub_names}}
        return out

    res["families"] = {}
    for fam in sorted(set(fam_of.values())):
        classes = [c for c in names if fam_of[c] == fam]
        if len(classes) < 2:
            continue
        r = restricted(classes)
        m = np.isin(y, classes)
        labels, M = metrics.pairwise_accuracy(X[m], y[m], fold[m], C=A["probe_C"])
        vocab = metrics.effective_vocabulary(labels, M, A["pairwise_threshold"])
        r["effective_vocabulary"] = {"size": len(vocab), "of": len(classes), "classes": vocab,
                                     "threshold": A["pairwise_threshold"],
                                     "pairwise_mean": round(float(np.nanmean(M)), 4),
                                     "pairs_below": [[str(labels[a]), str(labels[b]), round(float(M[a, b]), 3)]
                                                     for a in range(len(labels)) for b in range(a + 1, len(labels))
                                                     if M[a, b] < A["pairwise_threshold"]]}
        res["families"][fam] = r
    groups = {}
    for c in names:
        for grp in filter(None, str(df[df.label == c].groups.iloc[0]).split("|")):
            groups.setdefault(grp, []).append(c)
    res["sibling_groups"] = {grp: restricted(cl) for grp, cl in sorted(groups.items()) if len(cl) >= 2}
    return res


def run(models=("gemma", "clap")):
    A = cfg()["analysis"]
    out = {"protocol": {"probe_C": A["probe_C"], "knn_k": A["knn_k"], "tune_fold": A["tune_fold"],
                        "bootstrap": A["bootstrap"], "pairwise_threshold": A["pairwise_threshold"]}}
    for model in models:
        prefix, scores = choose_prefix(model)
        out[model] = {"prefix": prefix, "prefix_tune_scores": scores, "datasets": {}}
        for ds in DATASETS:
            r = analyse(model, ds, prefix or "SearchQuery", A)
            if r:
                out[model]["datasets"][ds] = r
                print(f"{model} {ds}: probe {r['probe']['acc']} knn {r['knn']['acc']} zs {r['zero_shot']['acc']} "
                      f"zs_c {r['zero_shot_centred']['acc']} (chance {r['chance']})", flush=True)
    f = path("reports") / "phase4_resolution.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
