import numpy as np

from audiovocab import metrics


def _blobs(n_per=20, k=3, d=16, spread=0.1, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    X = np.concatenate([c + spread * rng.normal(size=(n_per, d)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(np.array([f"c{i}" for i in range(k)]), n_per)
    groups = np.array([f"g{i // 2}" for i in range(len(y))])  # pairs of clips share a source
    fold = np.array([i % 5 + 1 for i in range(len(y) // 2) for _ in range(2)])
    return X, y, groups, fold


def test_knn_separable_and_excludes_own_group():
    X, y, g, _ = _blobs()
    assert (metrics.knn_predict(X, y, g, k=5) == y).mean() == 1.0
    # duplicate every clip within its own group: if own-group neighbours leaked, noise labels would be recovered
    Xd = np.repeat(X, 2, axis=0)
    yd = np.random.default_rng(1).permutation(np.repeat(y, 2))
    gd = np.repeat(np.arange(len(X)), 2).astype(str)
    acc = (metrics.knn_predict(Xd, yd, gd, k=1) == yd).mean()
    assert acc < 0.6


def test_probe_separable():
    X, y, _, fold = _blobs()
    assert (metrics.probe_predict(X, y, fold) == y).mean() > 0.95


def test_zero_shot_nearest_text():
    X = np.array([[1.0, 0], [0, 1.0]])
    T = np.array([[0, 1.0], [1.0, 0]])
    assert list(metrics.zero_shot_predict(X, T, ["b", "a"])) == ["a", "b"]


def test_tightness_signs():
    X, y, _, _ = _blobs(spread=0.05)
    t = metrics.tightness(X, y)
    assert all(v["tightness"] > 0 for v in t.values())


def test_bootstrap_ci_contains_mean_and_degenerate():
    c = np.array([1, 0, 1, 1, 0, 1, 1, 1])
    m, lo, hi = metrics.bootstrap_ci(c, np.arange(8), n=500)
    assert lo <= m <= hi
    assert metrics.bootstrap_ci(np.ones(10), np.arange(10))[1:] == (1.0, 1.0)


def test_effective_vocabulary_clique():
    labels = np.array(["a", "b", "c", "d"])
    M = np.array([[np.nan, .9, .9, .5], [.9, np.nan, .9, .9], [.9, .9, np.nan, .5], [.5, .9, .5, np.nan]])
    assert metrics.effective_vocabulary(labels, M, 0.7) == ["a", "b", "c"]
