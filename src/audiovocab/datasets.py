"""Phase 3: cut clips per the plan and write data/clips.parquet (one row per embedded window).

Windows (fixed before any embedding; Gemma truncates at 30 s and a 10 s window is within 0.91 cosine of 30 s):
  esc50       the whole 5 s clip
  fsd50k      first min(10 s, length)
  vocalsound  first min(10 s, length); 100 clips per class, one per speaker
  xenocanto   the 5 s window with the most energy (0.5 s steps), so the call itself is in the window
  cremad      the whole utterance (~2-3 s)
  fma         10-20 s of the 30 s clip (one window)
  musopen     two 10 s windows at 30% and 60% of the movement
  mtg         one 10 s window at 40% of the track

Columns: clip_id, dataset, item_id, window, source (path under data/raw), start, dur, label, family, groups,
group_id, split, licence, plus dataset-specific columns (genre, region, instr_tag, nd, sentence, instruments...).
"""
import io
import json
import tarfile
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from .audio import decode, probe_duration
from .config import ROOT, path

SEED = 0


def raw(*parts):
    return path("raw").joinpath(*parts)


def sel(name):
    return pd.read_parquet(path("data") / "selection" / f"{name}.parquet")


def tax():
    return yaml.safe_load((ROOT / "data" / "taxonomy.yaml").read_text(encoding="utf-8"))


def _groups(spec):
    return "|".join(spec.get("groups", []))


def _dur(p):
    try:
        return sf.info(str(p)).duration
    except Exception:
        return probe_duration(p)


def esc50(t):
    m = pd.read_csv(raw("meta", "esc50", "esc50.csv"))
    return pd.DataFrame({
        "dataset": "esc50", "item_id": m.filename.str[:-4], "window": 0,
        "source": "esc50/audio/" + m.filename, "start": 0.0, "dur": 5.0, "label": m.category,
        "family": m.category.map(lambda c: t["esc50"][c]["family"]),
        "groups": m.category.map(lambda c: _groups(t["esc50"][c])),
        "group_id": "esc_src_" + m.src_file.astype(str), "split": m.fold.astype(str),
        "licence": np.where(m.esc10, "CC BY 3.0 (ESC-10)", "CC BY-NC 3.0"),
    })


def fsd50k(t):
    s = sel("fsd50k")
    # clips embedded in Colab (Zenodo blocks the work network): durations come from the imported file
    cf = raw("fsd50k", "colab_durations.json")
    colab = json.loads(cf.read_text()) if cf.exists() else {}
    rows = []
    for r in s.itertuples():
        p = raw("fsd50k", "audio", f"{r.clip}.wav")
        if p.exists():
            dur = round(min(10.0, _dur(p)), 3)
        elif str(r.clip) in colab:
            dur = colab[str(r.clip)]
        else:
            continue
        rows.append({"dataset": "fsd50k", "item_id": r.clip, "window": 0, "source": f"fsd50k/audio/{r.clip}.wav",
                     "start": 0.0, "dur": dur, "label": r.label, "family": r.family,
                     "groups": _groups(t["fsd50k"][r.label]), "group_id": "fsd_up_" + r.group_id, "split": "eval",
                     "licence": r.licence})
    return pd.DataFrame(rows)


def vocalsound(t):
    z = zipfile.ZipFile(raw("vocalsound", "vs_release_16k.zip"))
    labels = {r.split(",")[1]: r.split(",")[2].strip('"').lower().replace(" ", "")
              for r in z.read("class_labels_indices_vs.csv").decode().splitlines()[1:]}
    recs = []
    for split in ("tr", "val", "te"):
        for d in json.loads(z.read(f"datafiles/{split}.json"))["data"]:
            recs.append({"file": d["wav"].rsplit("/", 1)[1], "speaker": d["spk_id"], "label": labels[d["labels"]],
                         "split": split})
    df = pd.DataFrame(recs).sample(frac=1, random_state=SEED)
    df = df.drop_duplicates("speaker").groupby("label").head(100)  # one clip per speaker, 100 per class
    out = raw("vocalsound", "audio")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.itertuples():
        p = out / r.file
        if not p.exists():
            p.write_bytes(z.read(f"audio_16k/{r.file}"))
        rows.append({"dataset": "vocalsound", "item_id": r.file[:-4], "window": 0, "source": f"vocalsound/audio/{r.file}",
                     "start": 0.0, "dur": round(min(10.0, _dur(p)), 3), "label": r.label,
                     "family": t["vocalsound"][r.label]["family"], "groups": _groups(t["vocalsound"][r.label]),
                     "group_id": "vs_spk_" + r.speaker, "split": r.split, "licence": "CC BY-SA 4.0"})
    return pd.DataFrame(rows)


def _loudest(x, sr, win=5.0, step=0.5):
    w, h = int(win * sr), int(step * sr)
    if len(x) <= w:
        return 0.0, len(x) / sr
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - w + 1, h)
    best = int(np.argmax(c[starts + w] - c[starts]))  # energy of each candidate window
    return round(starts[best] / sr, 3), win


def xenocanto(t):
    s = sel("xenocanto")
    rows = []
    for r in s.itertuples():
        p = raw("xenocanto", f"XC{r.clip}.mp3")
        if not p.exists():
            continue
        try:
            start, dur = _loudest(decode(p, 16000), 16000)
        except Exception:
            continue
        rows.append({"dataset": "xenocanto", "item_id": f"XC{r.clip}", "window": 0, "source": f"xenocanto/XC{r.clip}.mp3",
                     "start": start, "dur": dur, "label": r.label, "family": "animal",
                     "groups": _groups(t["xenocanto"][r.label]), "group_id": "xc_rec_" + r.group_id,
                     "split": "", "licence": r.licence, "quality": r.quality})
    return pd.DataFrame(rows)


def cremad(t):
    s = sel("cremad")
    rows = []
    for r in s.itertuples():
        p = raw("cremad", "AudioWAV", f"{r.clip}.wav")
        if not p.exists():
            continue
        rows.append({"dataset": "cremad", "item_id": r.clip, "window": 0, "source": f"cremad/AudioWAV/{r.clip}.wav",
                     "start": 0.0, "dur": round(min(10.0, _dur(p)), 3), "label": r.label, "family": "speech_emotion",
                     "groups": "emotions", "group_id": "cremad_actor_" + r.actor, "split": "",
                     "licence": "ODbL 1.0 / DbCL 1.0", "sentence": r.sentence, "level": r.level})
    return pd.DataFrame(rows)


def fma(t):
    s = pd.concat([sel("fma"), sel("fma_calibration")]).drop_duplicates("clip", keep="first")
    rows = []
    for r in s.itertuples():
        tid = f"{int(r.clip):06d}"
        p = raw("fma", f"{tid}.mp3")
        if not p.exists():
            continue
        rows.append({"dataset": "fma", "item_id": tid, "window": 0, "source": f"fma/{tid}.mp3", "start": 10.0,
                     "dur": 10.0, "label": r.genre if r.genre not in (None, "None") else "", "family": "music",
                     "groups": "", "group_id": "fma_artist_" + str(r.artist), "split": "", "licence": r.licence,
                     "genre": r.genre if r.genre not in (None, "None") else None,
                     "region": r.region if r.region not in (None, "None") else None,
                     "instr_tag": str(r.instr_tag) == "True", "nd": str(r.nd) == "True", "pool": r.pool})
    return pd.DataFrame(rows)


def musopen(t):
    s = sel("musopen")
    rows = []
    for r in s.itertuples():
        stem = r.clip.replace("/", "__")[:-5]
        for k in (0, 1):
            if not raw("musopen", f"{stem}_w{k}.flac").exists():
                continue
            rows.append({"dataset": "musopen", "item_id": stem, "window": k, "source": f"musopen/{stem}_w{k}.flac",
                         "start": 0.0, "dur": 10.0, "label": "classical", "family": "music", "groups": "",
                         "group_id": "musopen_work_" + r.work, "split": "", "licence": "Public Domain Mark 1.0",
                         "genre": "classical", "composer": r.composer, "nd": False})
    return pd.DataFrame(rows)


def mtg(t):
    s = sel("mtg")
    out = raw("mtg-jamendo", "audio")
    want = {r.path.replace(".mp3", ".low.mp3"): r for r in s.itertuples()}
    missing = [k for k in want if not (out / k).exists()]
    for shard in ("53", "86"):
        if not any(k.startswith(shard + "/") for k in missing):
            continue
        with tarfile.open(raw("mtg-jamendo", f"raw_30s_audio-low-{shard}.tar")) as tf:
            for m in tf:
                rel = m.name  # e.g. "53/1359953.low.mp3"
                if rel in want and not (out / rel).exists():
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    (out / rel).write_bytes(tf.extractfile(m).read())
    rows = []
    for rel, r in want.items():
        p = out / rel
        if not p.exists():
            continue
        dur = float(r.duration)
        rows.append({"dataset": "mtg", "item_id": r.track, "window": 0, "source": f"mtg-jamendo/audio/{rel}",
                     "start": round(max(0.0, 0.4 * dur - 5), 2), "dur": 10.0, "label": "", "family": "music",
                     "groups": "", "group_id": "mtg_artist_" + r.artist, "split": "",
                     "licence": "CC (per track); research-only (MTG-Jamendo)", "mtg_genres": r.genres,
                     "instruments": r.instruments, "instr_tag": str(r.tag_instrumental) == "True", "nd": True})
    return pd.DataFrame(rows)


BUILDERS = {"esc50": esc50, "fsd50k": fsd50k, "vocalsound": vocalsound, "xenocanto": xenocanto, "cremad": cremad,
            "fma": fma, "musopen": musopen, "mtg": mtg}


def build(only=None):
    t = tax()
    parts = []
    for name, fn in BUILDERS.items():
        if only and name not in only:
            continue
        df = fn(t)
        print(f"{name}: {len(df)} windows", flush=True)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["clip_id"] = df.dataset + ":" + df.item_id.astype(str) + ":" + df.window.astype(str)
    assert df.clip_id.is_unique
    cols = ["clip_id"] + [c for c in df.columns if c != "clip_id"]
    df = df[cols]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    f = path("data") / "clips.parquet"
    df.to_parquet(f, index=False)
    return df
