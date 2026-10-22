"""Deterministic metrics on L2-normalised embeddings. All take numpy arrays; labels/groups/folds are 1-D arrays."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def knn_predict(X, y, groups, k=10):
    """Leave-one-source-out kNN: a clip's neighbours never include clips from its own group."""
    S = X @ X.T
    S[groups[:, None] == groups[None, :]] = -np.inf
    labels = np.unique(y)
    lab_idx = np.searchsorted(labels, y)
    nn = np.argpartition(-S, kth=min(k, len(S) - 1) - 1, axis=1)[:, :k]
    pred = np.empty(len(y), dtype=y.dtype)
    for i in range(len(y)):
        sims = S[i, nn[i]]
        ok = np.isfinite(sims)
        votes = np.bincount(lab_idx[nn[i][ok]], minlength=len(labels)).astype(float)
        votes += 1e-6 * np.bincount(lab_idx[nn[i][ok]], weights=sims[ok], minlength=len(labels))  # tie-break
        pred[i] = labels[np.argmax(votes)] if ok.any() else labels[0]
    return pred


C_GRID = (0.001, 0.01, 0.1, 1.0)


def _fit(X, y, C, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=seed)).fit(X, y)


def choose_C(X, y, fold, grid=C_GRID, seed=0, max_inner=3):
    """Pick C by leave-one-fold-out over (up to `max_inner` of) the folds given; folds are group-disjoint."""
    inner = np.unique(fold)[:max_inner]
    if len(inner) < 2:
        return 1.0
    scores = {}
    for C in grid:
        acc = []
        for k in inner:
            tr, va = fold != k, fold == k
            if len(np.unique(y[tr])) < 2:
                continue
            acc.append(np.mean(_fit(X[tr], y[tr], C, seed).predict(X[va]) == y[va]))
        scores[C] = np.mean(acc) if acc else 0.0
    return max(grid, key=lambda C: (scores[C], -C))


def probe_predict(X, y, fold, C="nested", seed=0, return_C=False):
    """Grouped cross-validated linear probe: each fold is predicted by a model trained on the others.
    Features are standardised on the training folds (Gemma's unit vectors sit in a narrow cone, per-dimension std
    ~0.015). C="nested" picks C per outer fold by cross-validation inside its training folds only (decision 21):
    one fixed C can't suit both 2,000-clip sound sets and ~150-track genres."""
    pred = np.empty(len(y), dtype=y.dtype)
    chosen = {}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        c = choose_C(X[tr], y[tr], fold[tr], seed=seed) if C == "nested" else C
        chosen[int(f)] = c
        pred[te] = _fit(X[tr], y[tr], c, seed).predict(X[te])
    return (pred, chosen) if return_C else pred


def zero_shot_predict(X, T, names):
    """Label each clip by its nearest class text vector (rows of T, one per name)."""
    return np.asarray(names)[np.argmax(X @ T.T, axis=1)]


def class_text_vectors(text_emb, text_targets, names):
    """Mean of every text standing for each class (name + descriptions), re-normalised."""
    V = np.stack([text_emb[text_targets == n].mean(0) for n in names])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def tightness(X, y):
    """Per class: mean within-class cosine minus mean cosine to the nearest other class."""
    labels = np.unique(y)
    S = X @ X.T
    out = {}
    means = np.zeros((len(labels), len(labels)))
    for a, la in enumerate(labels):
        ia = y == la
        for b, lb in enumerate(labels):
            ib = y == lb
            block = S[np.ix_(ia, ib)]
            if a == b:
                n = ia.sum()
                means[a, b] = (block.sum() - n) / max(n * (n - 1), 1)
            else:
                means[a, b] = block.mean()
    for a, la in enumerate(labels):
        other = np.delete(means[a], a)
        nearest = labels[np.delete(np.arange(len(labels)), a)][np.argmax(other)]
        out[la] = {"within": float(means[a, a]), "nearest_other": float(other.max()), "nearest_class": str(nearest),
                   "tightness": float(means[a, a] - other.max())}
    return out


def recall_per_class(y, pred):
    return {str(l): float((pred[y == l] == l).mean()) for l in np.unique(y)}


def bootstrap_ci(correct, groups, n=1000, seed=0, alpha=0.05):
    """Cluster bootstrap over groups for a mean of 0/1 (or any per-row score). Returns (mean, lo, hi)."""
    correct = np.asarray(correct, dtype=float)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=correct)
    cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), size=(n, len(ug)))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(correct.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def pairwise_accuracy(X, y, fold, C=1.0):
    """Binary grouped-CV probe accuracy for every pair of classes. Returns (labels, matrix)."""
    labels = np.unique(y)
    M = np.full((len(labels), len(labels)), np.nan)
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            m = (y == labels[a]) | (y == labels[b])
            if len(np.unique(fold[m])) < 2:
                continue
            p = probe_predict(X[m], y[m], fold[m], C=C if C != "nested" else 0.01)  # fixed: thousands of pairs
            M[a, b] = M[b, a] = float(np.mean(p == y[m]))
    return labels, M


def effective_vocabulary(labels, M, threshold):
    """Largest set of classes in which every pair is separated with accuracy >= threshold (a maximum clique)."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            if np.isfinite(M[a, b]) and M[a, b] >= threshold:
                G.add_edge(a, b)
    clique, _ = nx.max_weight_clique(G, weight=None)
    return [str(labels[i]) for i in sorted(clique)]
