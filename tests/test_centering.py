import numpy as np

from audiovocab.centering import GENERIC_PHRASES, center


def test_generic_phrases_count():
    assert len(GENERIC_PHRASES) == 77 and len(set(GENERIC_PHRASES)) == 77


def test_strength_zero_is_identity():
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    assert center(x, x.mean(0), 0) is x


def test_full_centring_removes_mean_and_normalises():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 16)) + 5.0  # shared offset, like the modality gap
    m = x.mean(0)
    c = center(x, m, 1.0)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0)
    # offset gone: mean pairwise cosine drops from ~1 to ~0
    raw = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert (raw @ raw.T).mean() > 0.9 and abs((c @ c.T)[~np.eye(50, dtype=bool)].mean()) < 0.1


def test_partial_strength_is_between():
    x = np.array([[3.0, 1.0], [3.0, -1.0]])
    m = x.mean(0)
    cos = lambda c: float(c[0] @ c[1] / np.linalg.norm(c[0]) / np.linalg.norm(c[1]))
    assert cos(center(x, m, 1.0)) < cos(center(x, m, 0.5)) < cos(x)
