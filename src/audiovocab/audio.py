"""Decoding: mono float32 at whatever rate a model needs (16 kHz Gemma/Whisper, 48 kHz CLAP)."""
import subprocess

import numpy as np


def probe_duration(file):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(file)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def decode(file, sr=16000, start=None, dur=None):
    """Mono float32 samples in [-1, 1]."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(file)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)
