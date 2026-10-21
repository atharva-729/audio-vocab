"""Phase 1 behaviour tests on ~30 of the sister project's podcast clips (read by path).

  length:    the same start point at 2, 5, 10, 20, 30, 40, 60 s; cosine to the 20 s and 30 s versions
  loudness:  -12, -6, 0, +6 dB gain (no clipping applied); cosine to 0 dB
  synthetic: silence, white noise, pink noise; how close they sit to real clips (is silence a hub?)
  wording:   prompt prefix on/off, single word vs phrase vs description; text-text and text-audio cosines
  timing:    seconds of compute per second of audio for 1-10 s clips

Writes reports/phase1_behaviour.json and .md. Embeddings are cached, so reruns are cheap.
"""
import json
import time

import numpy as np
import pandas as pd

from . import embed
from .audio import decode
from .config import path

N_CLIPS, SEED = 30, 0
LENGTHS = [2, 5, 10, 20, 30, 40, 60]
GAINS_DB = [-12, -6, 0, 6]
TIMING_LENGTHS = [1, 2, 5, 10]

# Wording test: concept -> (single word, short phrase, long description)
WORDING = {
    "laughter": ("laughter", "people laughing", "a group of friends laughing loudly together at a funny joke"),
    "rain": ("rain", "rain falling", "heavy rain falling steadily on a roof during a storm"),
    "dog": ("dog", "a dog barking", "a large dog barking repeatedly at someone outside the house"),
    "music": ("music", "piano music", "a slow and gentle solo piano piece played in a quiet room"),
    "speech": ("speech", "a man talking", "a man talking calmly and clearly into a microphone"),
    "crowd": ("crowd", "a cheering crowd", "a large stadium crowd cheering and clapping after a goal"),
    "whisper": ("whisper", "someone whispering", "a woman whispering softly and slowly very close to the microphone"),
    "anger": ("anger", "an angry voice", "a man shouting angrily during a heated argument"),
}
PROMPTS = {"none": None, "SearchQuery": "SearchQuery", "Document": "Document", "Classification": "Classification"}


def sister(*parts):
    return path("sister").joinpath(*parts)


def pick_clips():
    """One segment per show (first N_CLIPS shows after a seeded shuffle), each with >= 60 s of audio after it."""
    seg = pd.read_parquet(sister("index", "segments.parquet"))
    eps = pd.read_parquet(sister("data", "episodes.parquet")).set_index("episode_id")
    seg = seg[seg.start_s <= 900 - max(LENGTHS) - 1]
    pick = seg.sample(frac=1, random_state=SEED).drop_duplicates("show_id").head(N_CLIPS)
    pick = pick.assign(audio_file=[eps.loc[e, "audio_file"] for e in pick.episode_id])
    return pick.reset_index(drop=True)


_decoded = {}


def _episode(audio_file, sr):
    k = (audio_file, sr)
    if k not in _decoded:
        if len(_decoded) > 8:
            _decoded.clear()
        _decoded[k] =decode(sister("data", "audio_cache", "raw", audio_file), sr)
    return _decoded[k]


def _pink(n, rng):
    """1/f noise via spectral shaping of white noise."""
    f = np.fft.rfftfreq(n)
    spec = np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.maximum(f, f[1]))
    x = np.fft.irfft(spec, n)
    return x / np.abs(x).max()


def load_clip(key, sr):
    n = int(key["dur"] * sr)
    if key["kind"] == "podcast":
        x = _episode(key["audio_file"], sr)
        s = int(key["start"] * sr)
        x = x[s:s + n]
    elif key["kind"] == "silence":
        x = np.zeros(n, np.float32)
    elif key["kind"] == "white":
        x = np.random.default_rng(key["seed"]).uniform(-1, 1, n) * 0.3
    elif key["kind"] == "pink":
        x = _pink(n, np.random.default_rng(key["seed"])) * 0.3
    else:
        raise ValueError(key)
    return (x * 10 ** (key.get("gain_db", 0) / 20)).astype(np.float32)


def _key(row, dur, gain_db=0, start_offset=0.0):
    return {"dataset": "podcast", "kind": "podcast", "clip": row.segment_id, "audio_file": row.audio_file,
            "start": float(row.start_s + start_offset), "dur": float(dur), "gain_db": gain_db}


def _cos(a, b):
    return (a * b).sum(-1)


def _stats(x):
    x = np.asarray(x)
    return {"mean": round(float(x.mean()), 4), "min": round(float(x.min()), 4), "p10": round(float(np.percentile(x, 10)), 4)}


def run_model(name, clips):
    model = embed.load(name)
    res = {}

    # length
    by_len = {L: embed.embed_cached(model, [_key(r, L) for r in clips.itertuples()], load_clip) for L in LENGTHS}
    res["length"] = {L: {"cos_to_20s": _stats(_cos(by_len[L], by_len[20])),
                         "cos_to_30s": _stats(_cos(by_len[L], by_len[30]))} for L in LENGTHS}
    # Same start, adjacent non-overlapping 20 s windows: how different is "the next 20 s" (a baseline for drift)?
    nxt = embed.embed_cached(model, [_key(r, 20, start_offset=20) for r in clips.itertuples()], load_clip)
    res["length"]["next_20s_window_cos"] = _stats(_cos(nxt, by_len[20]))
    # Does audio beyond 30 s change anything? (truncation test)
    res["length"]["truncation_40_vs_30_max_abs_diff"] = float(np.abs(by_len[40] - by_len[30]).max())
    res["length"]["truncation_60_vs_30_max_abs_diff"] = float(np.abs(by_len[60] - by_len[30]).max())

    # loudness (10 s clips)
    by_gain = {g: embed.embed_cached(model, [_key(r, 10, gain_db=g) for r in clips.itertuples()], load_clip)
               for g in GAINS_DB}
    res["loudness"] = {g: _stats(_cos(by_gain[g], by_gain[0])) for g in GAINS_DB}

    # synthetic sounds vs real clips (10 s)
    real = by_gain[0]
    rr = real @ real.T
    real_pairs = rr[~np.eye(len(real), dtype=bool)]
    synth_keys = {k: {"dataset": "synthetic", "kind": k, "dur": 10.0, "seed": 0} for k in ("silence", "white", "pink")}
    synth = {k: embed.embed_cached(model, [v], load_clip)[0] for k, v in synth_keys.items()}
    res["synthetic"] = {"clip_to_clip_cos": _stats(real_pairs)}
    for k, v in synth.items():
        c = real @ v
        # hub test: for how many clips is this synthetic vector nearer than the clip's nearest real neighbour?
        nn = np.sort(rr - 2 * np.eye(len(real)), axis=1)[:, -1]
        res["synthetic"][k] = {"cos_to_clips": _stats(c), "closer_than_nearest_real_neighbour": int((c > nn).sum())}
    res["synthetic"]["silence_vs_white"] = round(float(synth["silence"] @ synth["white"]), 4)
    res["synthetic"]["silence_vs_pink"] = round(float(synth["silence"] @ synth["pink"]), 4)

    # wording
    words = [w for triple in WORDING.values() for w in triple]
    t = {}
    for pname, p in PROMPTS.items():
        if name == "clap" and pname != "none":
            continue
        t[pname] = model.text(words, prompt_name=p) if name == "gemma" else model.text(words)
    wres = {}
    if name == "gemma":
        wres["prefix_vs_none_same_text"] = {p: _stats(_cos(t[p], t["none"])) for p in PROMPTS if p != "none"}
        wres["searchquery_vs_document_same_text"] = _stats(_cos(t["SearchQuery"], t["Document"]))
    for pname, tv in t.items():
        tv3 = tv.reshape(len(WORDING), 3, -1)
        within = [float(tv3[i, a] @ tv3[i, b]) for i in range(len(WORDING)) for a, b in ((0, 1), (0, 2), (1, 2))]
        across = [float(tv3[i, f] @ tv3[j, f]) for f in range(3) for i in range(len(WORDING))
                  for j in range(len(WORDING)) if i < j]
        audio_cos = tv @ real.T
        wres[pname] = {
            "same_concept_diff_wording": _stats(within),
            "diff_concept_same_form": _stats(across),
            "text_to_audio_by_form": {form: _stats(audio_cos.reshape(len(WORDING), 3, -1)[:, k])
                                      for k, form in enumerate(("word", "phrase", "description"))},
        }
    res["wording"] = wres

    # timing per second of audio (fresh, uncached)
    timing = {}
    rng = np.random.default_rng(1)
    for L in TIMING_LENGTHS:
        batch = [rng.uniform(-0.1, 0.1, int(L * model.sr)).astype(np.float32) for _ in range(8)]
        model.audio(batch[:1])  # warm-up
        t0 = time.time()
        model.audio(batch)
        dt = time.time() - t0
        timing[L] = {"s_per_clip": round(dt / 8, 3), "s_per_audio_s": round(dt / 8 / L, 3)}
    res["timing"] = timing
    return res


def run(models=("gemma", "clap")):
    clips = pick_clips()
    out = {"clips": clips.segment_id.tolist(), "n_clips": len(clips)}
    for m in models:
        print(f"== {m}", flush=True)
        out[m] = run_model(m, clips)
    f = path("reports") / "phase1_behaviour.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
