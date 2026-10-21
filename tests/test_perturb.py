import numpy as np
import pytest

from audiovocab.perturb import PERTURBATIONS, apply, source_factor

SR = 16000


def _tone(sec=2.0):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_deterministic_and_loudness_matched(name):
    x = _tone()
    a, b = apply(x, SR, name), apply(x, SR, name)
    assert np.array_equal(a, b)
    assert abs(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(x ** 2)) - 1) < 0.05 or np.abs(a).max() >= 0.999


def test_lengths():
    x = _tone(2.0)
    assert len(apply(x, SR, "reverse")) == len(x)
    assert len(apply(x, SR, "shuffle_0.5s")) == len(x)
    src = _tone(2.0 * source_factor("tempo_x1.33"))
    assert abs(len(apply(src, SR, "tempo_x1.33", out_len=len(x))) - len(x)) <= 1


def test_reverse_and_shuffle_keep_samples():
    x = _tone()
    assert np.allclose(np.sort(apply(x, SR, "reverse")), np.sort(x), atol=1e-6)
    assert np.allclose(np.sort(apply(x, SR, "shuffle_0.5s")), np.sort(x), atol=1e-6)


def test_lowpass_removes_high_tone():
    t = np.arange(SR) / SR
    hi = (0.3 * np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
    lo = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    y = apply(hi + lo, SR, "lowpass_1500")
    spec = np.abs(np.fft.rfft(y))
    assert spec[5000] < 0.05 * spec[200]
