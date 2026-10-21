"""Instrumental filter: faster-whisper small (int8) on a 20 s window around each music clip.

A word counts as "confident" when its segment has avg_logprob > -1.0 and no_speech_prob < 0.5, with Silero VAD on
(fixed rule). A track is flagged vocal when it has at least T confident words. T is calibrated on a held-out set:
100 FMA tracks tagged Instrumental vs 100 tagged with vocal subgenres (Rap, Singer-Songwriter, Jazz: Vocal,
Easy Listening: Vocal), from artists outside the analysis pools. Then T is applied unchanged to the analysis tracks.

Writes data/whisper.parquet (per item) and reports/instrumental_filter.json (calibration + funnel).
"""
import json
import time

import numpy as np
import pandas as pd

from .audio import decode
from .config import cfg, model_path, path

LOGPROB_MIN, NOSPEECH_MAX = -1.0, 0.5
T_GRID = list(range(1, 21))


def _items():
    c = pd.read_parquet(path("data") / "clips.parquet")
    m = c[c.dataset.isin(["fma", "mtg"])].drop_duplicates(["dataset", "item_id"])
    return m


def transcribe_all(threads=4):
    from faster_whisper import WhisperModel
    out_f = path("data") / "whisper.parquet"
    done = pd.read_parquet(out_f) if out_f.exists() else pd.DataFrame(columns=["key"])
    have = set(done.key)
    items = _items()
    todo = [r for r in items.to_dict("records") if f"{r['dataset']}:{r['item_id']}" not in have]
    print(f"whisper: {len(todo)} of {len(items)} items to do", flush=True)
    if not todo:
        return done
    m = WhisperModel(model_path("whisper"), device="cpu", compute_type=cfg()["models"]["whisper_compute"],
                     cpu_threads=threads)
    rows, t0 = [], time.time()
    for i, r in enumerate(todo, 1):
        start = max(0.0, float(r["start"]) - 5)
        try:
            x = decode(path("raw") / r["source"], 16000, start=start or None, dur=20.0)
            segs, info = m.transcribe(x, beam_size=1, vad_filter=True, condition_on_previous_text=False)
            segs = list(segs)
            conf = [s for s in segs if s.avg_logprob > LOGPROB_MIN and s.no_speech_prob < NOSPEECH_MAX]
            rows.append({"key": f"{r['dataset']}:{r['item_id']}", "dataset": r["dataset"], "item_id": r["item_id"],
                         "words_conf": sum(len(s.text.split()) for s in conf),
                         "words_all": sum(len(s.text.split()) for s in segs), "language": info.language,
                         "language_prob": round(float(info.language_probability), 3),
                         "text": " ".join(s.text.strip() for s in conf)[:300]})
        except Exception as e:
            rows.append({"key": f"{r['dataset']}:{r['item_id']}", "dataset": r["dataset"], "item_id": r["item_id"],
                         "words_conf": -1, "words_all": -1, "language": "", "language_prob": 0.0, "text": f"ERR {e}"})
        if i % 100 == 0 or i == len(todo):
            all_rows = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            all_rows.to_parquet(out_f, index=False)
            print(f"whisper: {i}/{len(todo)} ({(time.time() - t0) / i:.1f} s/item)", flush=True)
    return pd.read_parquet(out_f)


def calibrate_and_apply():
    w = pd.read_parquet(path("data") / "whisper.parquet").set_index("key")
    c = pd.read_parquet(path("data") / "clips.parquet")
    items = c[c.dataset.isin(["fma", "mtg"])].drop_duplicates(["dataset", "item_id"]).copy()
    items["key"] = items.dataset + ":" + items.item_id.astype(str)
    items = items[items.key.isin(w.index)]
    items["words"] = items.key.map(w.words_conf)
    cal = items[items.pool.isin(["calib_instrumental", "calib_vocal"])]
    pos = (cal.pool == "calib_vocal").values
    grid = []
    for T in T_GRID:
        flag = (cal.words >= T).values
        tpr, tnr = flag[pos].mean(), (~flag[~pos]).mean()
        grid.append({"T": T, "vocal_recall": round(float(tpr), 3), "instrumental_recall": round(float(tnr), 3),
                     "balanced_acc": round(float((tpr + tnr) / 2), 3)})
    best = max(grid, key=lambda g: (g["balanced_acc"], -g["T"]))
    T = best["T"]
    items["vocal"] = items.words >= T
    ana = items[~items.pool.isin(["calib_instrumental", "calib_vocal"])]
    funnel = {}
    fma = ana[ana.dataset == "fma"]
    for pool in ("genre", "region"):
        p = fma[fma.pool == pool]
        key = "genre" if pool == "genre" else "region"
        funnel[pool] = {k: {"candidates": int(len(v)), "metadata_instrumental_tag": int(v.instr_tag.astype(str).eq("True").sum()),
                            "whisper_instrumental": int((~v.vocal).sum()),
                            "whisper_vocal": int(v.vocal.sum()),
                            "instrumental_artists": int(v[~v.vocal].group_id.nunique())}
                        for k, v in p.groupby(key)}
    mtg = ana[ana.dataset == "mtg"]
    funnel["mtg"] = {"tracks": int(len(mtg)), "tag_instrumental": int(mtg.instr_tag.astype(str).eq("True").sum()),
                     "whisper_instrumental": int((~mtg.vocal).sum()),
                     "tag_and_whisper_instrumental": int(((~mtg.vocal) & mtg.instr_tag.astype(str).eq("True")).sum())}
    # agreement of whisper with metadata on tagged tracks (not used for calibration)
    tagged = fma[fma.instr_tag.astype(str) == "True"]
    funnel["fma_tagged_instrumental_flagged_vocal"] = int(tagged.vocal.sum())
    funnel["fma_tagged_instrumental_total"] = int(len(tagged))
    out = {"rule": {"logprob_min": LOGPROB_MIN, "no_speech_max": NOSPEECH_MAX, "vad": True, "window_s": 20},
           "calibration": {"n_instrumental": int((~pos).sum()), "n_vocal": int(pos.sum()), "grid": grid, "chosen": best},
           "funnel": funnel}
    (path("reports") / "instrumental_filter.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    flags = items[["dataset", "item_id", "words", "vocal"]].copy()
    flags.to_parquet(path("data") / "vocal_flags.parquet", index=False)
    return out
