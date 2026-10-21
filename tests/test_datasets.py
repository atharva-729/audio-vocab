import numpy as np

from audiovocab.datasets import _loudest


def test_loudest_window_finds_the_burst():
    sr = 1000
    x = np.zeros(20 * sr)
    x[12 * sr:14 * sr] = 1.0  # a 2 s burst at 12-14 s
    start, dur = _loudest(x, sr, win=5.0, step=0.5)
    assert dur == 5.0 and 9.0 <= start <= 12.0 and start + dur >= 14.0


def test_loudest_short_clip_is_whole_clip():
    assert _loudest(np.ones(3000), 1000, win=5.0) == (0.0, 3.0)
