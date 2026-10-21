"""Phase 8 perturbations: change one thing at a time. Deterministic; every parameter is in PERTURBATIONS.

The name is part of the embedding cache key, so a changed parameter needs a new name.
Time-stretch reads extra source audio (`source_factor`) so the output is still a 10 s window.
"""
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

PERTURBATIONS = {
    "harmonic_only":   {"what": "remove percussion (HPSS harmonic part)", "kind": "hpss", "keep": "harmonic"},
    "percussive_only": {"what": "remove harmony (HPSS percussive part)", "kind": "hpss", "keep": "percussive"},
    "tempo_x0.75":     {"what": "time-stretch to 0.75x tempo, pitch kept", "kind": "stretch", "rate": 0.75},
    "tempo_x1.33":     {"what": "time-stretch to 1.33x tempo, pitch kept", "kind": "stretch", "rate": 1.33},
    "pitch_-3":        {"what": "pitch shift -3 semitones", "kind": "pitch", "steps": -3},
    "pitch_+3":        {"what": "pitch shift +3 semitones", "kind": "pitch", "steps": 3},
    "lowpass_1500":    {"what": "darker: 4th-order low-pass at 1.5 kHz", "kind": "filter", "btype": "lowpass", "hz": 1500},
    "highpass_500":    {"what": "brighter/thinner: 4th-order high-pass at 500 Hz", "kind": "filter", "btype": "highpass", "hz": 500},
    "compress":        {"what": "flatten dynamics: divide by a 50 ms RMS envelope", "kind": "compress", "frame_s": 0.05},
    "shuffle_0.5s":    {"what": "shuffle 0.5 s chunks (seed 0): keeps timbre, destroys melody and phrasing", "kind": "shuffle", "chunk_s": 0.5, "seed": 0},
    "reverse":         {"what": "play backwards", "kind": "reverse"},
}


def source_factor(name):
    p = PERTURBATIONS[name]
    return p["rate"] if p["kind"] == "stretch" else 1.0


def _rms_match(y, x):
    ry, rx = np.sqrt(np.mean(y ** 2)) + 1e-9, np.sqrt(np.mean(x ** 2)) + 1e-9
    return y * (rx / ry)


def apply(x, sr, name, out_len=None):
    """Apply perturbation `name` to mono float32 x at sample rate sr. Output loudness matches the input (RMS)."""
    p = PERTURBATIONS[name]
    x = np.asarray(x, np.float32)
    k = p["kind"]
    if k == "hpss":
        h, pc = librosa.effects.hpss(x)
        y = h if p["keep"] == "harmonic" else pc
    elif k == "stretch":
        y = librosa.effects.time_stretch(x, rate=p["rate"])
        n = out_len or int(round(len(x) / p["rate"]))
        y = y[:n]
    elif k == "pitch":
        y = librosa.effects.pitch_shift(x, sr=sr, n_steps=p["steps"])
    elif k == "filter":
        sos = butter(4, p["hz"], btype=p["btype"], fs=sr, output="sos")
        y = sosfiltfilt(sos, x)
    elif k == "compress":
        f = max(1, int(p["frame_s"] * sr))
        env = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(f) / f, "same")) + 1e-4
        y = x / env
    elif k == "shuffle":
        c = int(p["chunk_s"] * sr)
        n = len(x) // c
        chunks = [x[i * c:(i + 1) * c] for i in range(n)]
        order = np.random.default_rng(p["seed"]).permutation(n)
        y = np.concatenate([chunks[i] for i in order] + [x[n * c:]])
    elif k == "reverse":
        y = x[::-1]
    else:
        raise ValueError(name)
    y = _rms_match(np.asarray(y, np.float64), x)
    return np.clip(y, -1, 1).astype(np.float32)
