"""Phase 9 bridge: map the sister project's listening-test searches onto this study's findings.

Listening scores come from podcast-vibe-search/reports/study_results.json ("long", centring strength 0 = the raw
search that scored 33/48), read by path. The query -> class mapping below was fixed before any Phase 4 result was
computed. Verdict rules (fixed):
  probe < 0.70                      -> "weak area of the space"   (the audio embedding doesn't separate it well)
  probe >= 0.70 and zero-shot < 0.5 -> "knows but can't name"     (the information is there; words don't reach it)
  probe >= 0.70 and zero-shot >= 0.5 -> "space and words fine here: likely a podcast-corpus data gap"
Writes reports/vibe_search_bridge.json.
"""
import json

import numpy as np

from .config import path

MAP = {
    "two friends laughing about football, late-night energy": [("esc50", "laughing"), ("vocalsound", "laughter")],
    "calm, slow, one voice, like a bedtime story": [("cremad", "NEU"), ("cremad", "SAD")],
    "heated argument, people talking over each other": [("cremad", "ANG"), ("fsd50k", "Shout"), ("fsd50k", "Screaming")],
    "soft whispering, very quiet": [("fsd50k", "Whispering")],
    "excited fast talking about sports": [("cremad", "HAP")],
    "serious news voice reading the headlines": [("fsd50k", "Male_speech_and_man_speaking"), ("fsd50k", "Female_speech_and_woman_speaking")],
    "dramatic storytelling about a murder": [("cremad", "FEA")],
    "rain and nature sounds, no talking": [("esc50", "rain"), ("esc50", "wind"), ("fsd50k", "Rain"), ("fsd50k", "Stream")],
    "upbeat music and singing": [("fsd50k", "Singing")],
    "thoughtful slow interview about life": [("cremad", "NEU")],
    "kids story with funny voices": [("fsd50k", "Child_speech_and_kid_speaking")],
    "relaxed chat between comedians, lots of laughs": [("esc50", "laughing"), ("vocalsound", "laughter"), ("fsd50k", "Giggle")],
}


def _class_scores(model, ds, label, p4, p6):
    if ds == "cremad":
        e = p6.get(model, {})
        if not e:
            return None
        return {"probe": e["emotion_recall"].get(label), "zero_shot": e["zero_shot"]["raw"]["recall"].get(label)}
    c = p4.get(model, {}).get("datasets", {}).get(ds, {}).get("classes", {}).get(label)
    return {"probe": c["probe"], "zero_shot": c["zero_shot"]} if c else None


def run(model="gemma"):
    sister = json.loads((path("sister") / "reports" / "study_results.json").read_text(encoding="utf-8"))
    raw = next(r for r in sister["long"] if r["strength"] == 0.0)
    p4 = json.loads((path("reports") / "phase4_resolution.json").read_text())
    p6f = path("reports") / "phase6_emotion.json"
    p6 = json.loads(p6f.read_text()) if p6f.exists() else {}
    rows = []
    for q, classes in MAP.items():
        sc = [(ds, lab, _class_scores(model, ds, lab, p4, p6)) for ds, lab in classes]
        sc = [(d, l, s) for d, l, s in sc if s and s["probe"] is not None]
        probe = float(np.mean([s["probe"] for _, _, s in sc])) if sc else None
        zs = float(np.mean([s["zero_shot"] for _, _, s in sc])) if sc else None
        if probe is None:
            verdict = "not covered by this study's data"
        elif probe < 0.70:
            verdict = "weak area of the space"
        elif zs < 0.5:
            verdict = "knows but can't name"
        else:
            verdict = "space and words fine here: likely a podcast-corpus data gap"
        rows.append({"query": q, "listening_yes": raw["per_query"].get(q), "listening_of": 4,
                     "classes": [{"dataset": d, "label": l, **s} for d, l, s in sc],
                     "probe_mean": round(probe, 4) if probe is not None else None,
                     "zero_shot_mean": round(zs, 4) if zs is not None else None, "verdict": verdict})
    out = {"model": model, "source": "podcast-vibe-search/reports/study_results.json (long, strength 0.0)",
           "listening_total": [raw["yes"], raw["rated"]], "rows": rows}
    (path("reports") / "vibe_search_bridge.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
