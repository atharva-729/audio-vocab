"""Phase 9: reports/*.json -> docs/data.js. The page (docs/index.html) renders every number from this file;
no number is typed into the HTML. Sentences that contain numbers are built here from the reports."""
import json
import re

from .config import path

FAMILY_ORDER = ["human_vocal", "speech_emotion", "human_activity", "animal", "nature", "machine_domestic", "music"]
FAMILY_NAME = {"human_vocal": "Human vocal", "speech_emotion": "Speech emotion", "human_activity": "Human activity",
               "animal": "Animals", "nature": "Nature", "machine_domestic": "Machines & domestic", "music": "Music"}
DATASET_CARDS = [
    {"id": "esc50", "name": "ESC-50", "what": "50 everyday sound classes, 5 s clips, official folds", "licence": "CC BY-NC 3.0", "play": "ESC-10 clips only",
     "kind": "Everyday sounds", "plain": "2,000 five-second clips of 50 everyday sounds: dog, rain, chainsaw, sneeze, clock tick…",
     "why": "The standard baseline: can it hear everyday sounds at all?"},
    {"id": "fsd50k", "name": "FSD50K", "what": "AudioSet-ontology sound events incl. giggle vs chuckle", "licence": "per clip CC0/BY/BY-NC", "play": "CC0/BY clips only",
     "kind": "Fine sound differences", "plain": "Clips in 62 finer classes: giggle vs chuckle, scream vs shout, whisper vs singing, car vs truck…",
     "why": "The only free dataset with laughter <i>types</i> (giggle, chuckle), so we can test <b>kinds of laughter</b>."},
    {"id": "vocalsound", "name": "VocalSound", "what": "laughter, sigh, cough, throat-clear, sneeze, sniff; one clip per speaker", "licence": "CC BY-SA 4.0", "play": "yes",
     "kind": "Human non-speech sounds", "plain": "Laughter, sigh, cough, throat-clear, sneeze, sniff, each from a different person.",
     "why": "<b>Human sounds without words</b>, from thousands of people, so no single voice to memorise."},
    {"id": "xenocanto", "name": "xeno-canto", "what": "17 bird species: cuckoo, 6 owls, doves, hoopoe, songbirds, corvids", "licence": "per recording, no ND", "play": "CC0/BY/BY-SA only",
     "kind": "Bird species", "plain": "Recordings of 17 species: cuckoo, 6 owls, doves, hoopoe, songbirds, crows.",
     "why": "Answers <b>\"cuckoo vs owl\"</b> directly; the nature side of humans vs nature."},
    {"id": "cremad", "name": "CREMA-D", "what": "12 sentences x 6 emotions x 91 actors", "licence": "ODbL / DbCL", "play": "yes",
     "kind": "Emotion in voices", "plain": "91 actors each saying the same 12 sentences in 6 emotions (angry, happy, sad, afraid, disgusted, neutral).",
     "why": "<b>Same words</b> in every emotion: does it hear <i>how</i> it's said, or only <i>what</i>?"},
    {"id": "fma", "name": "FMA", "what": "30 s clips; genre + regional subgenres; artist ids", "licence": "per track CC (metadata CC BY)", "play": "CC BY/BY-SA/CC0 only",
     "kind": "Music by genre and region", "plain": "Thousands of 30-second tracks labelled with genre (jazz, rock, classical…) and region (Brazilian, Indian…).",
     "why": "Can a <b>genre or region word</b> find music? Sung tracks removed, so lyrics don't help."},
    {"id": "musopen", "name": "Musopen", "what": "canonical classical: 37 works, symphonies to sonatas", "licence": "Public Domain Mark", "play": "yes",
     "kind": "Real classical music", "plain": "Recordings of Beethoven, Mozart, Brahms, Schubert… (public domain).",
     "why": "FMA's \"classical\" is mostly amateur; this checks the <b>real thing</b>."},
    {"id": "mtg", "name": "MTG-Jamendo", "what": "genre + instrument tags on the same tracks (2 shards)", "licence": "research-only", "play": "no (labels only)",
     "kind": "Music with instrument labels", "plain": "Tracks tagged with both genre and instrument (piano, guitar…).",
     "why": "Holds the <b>instrument fixed</b>: is \"classical\" just \"piano\"?"},
]


def _load(name):
    f = path("reports") / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def family_table(p4):
    """One row per (model, dataset, family) with >= 2 classes."""
    rows = []
    for model in ("gemma", "clap"):
        for ds, r in (p4 or {}).get(model, {}).get("datasets", {}).items():
            for fam, f in r.get("families", {}).items():
                ev = f["effective_vocabulary"]
                rows.append({"model": model, "dataset": ds, "family": fam, "n_classes": len(f["classes"]),
                             "chance": f["chance"], "probe": f["probe"], "knn": f["knn"], "zero_shot": f["zero_shot"],
                             "zero_shot_centred": f["zero_shot_centred"], "vocab": ev["size"], "vocab_of": ev["of"],
                             "vocab_classes": ev["classes"], "pairs_below": ev["pairs_below"][:12],
                             "per_class": f["per_class"]})
    return rows


def class_points(p4):
    out = []
    for model in ("gemma", "clap"):
        for ds, r in (p4 or {}).get(model, {}).get("datasets", {}).items():
            for c, v in r["classes"].items():
                out.append({"model": model, "dataset": ds, "label": c, "family": v["family"], "probe": v["probe"],
                            "zero_shot": v["zero_shot"], "zero_shot_centred": v["zero_shot_centred"],
                            "tightness": v["tightness"], "nearest": v["nearest_class"], "n": v["n"]})
    return out


def build():
    D = {"families": FAMILY_ORDER, "family_names": FAMILY_NAME, "datasets": DATASET_CARDS}
    plan = _load("phase2_plan.json")
    p3 = _load("phase3_sanity.json")
    D["counts"] = p3 and {"clips": p3["clips_per_dataset"], "items": p3["items_per_dataset"], "embedded": p3["embedded"]}
    D["parity"] = _load("parity.json")
    D["p1"] = _load("phase1_behaviour.json")
    if D["p1"]:  # keep the summary only
        D["p1"] = {m: {k: D["p1"][m][k] for k in ("length", "loudness", "synthetic", "timing")} for m in ("gemma", "clap")}
    D["sanity"] = p3 and p3["sanity"]
    D["umap"] = _load("phase3_umap.json")
    p4 = _load("phase4_resolution.json")
    D["p4"] = p4 and {"protocol": p4["protocol"],
                      "models": {m: {"prefix": p4[m]["prefix"], "prefix_tune_scores": p4[m]["prefix_tune_scores"],
                                     "datasets": {ds: {k: r[k] for k in ("n_clips", "n_classes", "chance", "probe", "knn", "zero_shot", "zero_shot_centred")}
                                                  for ds, r in p4[m]["datasets"].items()},
                                     "siblings": {ds: r["sibling_groups"] for ds, r in p4[m]["datasets"].items()}}
                                 for m in ("gemma", "clap") if m in p4},
                      "families": family_table(p4), "classes": class_points(p4)}
    D["p5"] = _load("phase5_words.json")
    D["p6"] = _load("phase6_emotion.json")
    D["p7"] = _load("phase7_genre.json")
    D["p8"] = _load("phase8_whatmakes.json")
    D["bridge"] = _load("vibe_search_bridge.json")
    D["instrumental"] = _load("instrumental_filter.json")
    D["plan"] = plan and {k: plan[k] for k in ("esc50", "xenocanto", "cremad", "musopen") if k in plan}
    from .findings import findings
    D["text"] = findings(D)
    js = "window.DATA = " + json.dumps(D, separators=(",", ":"), default=str) + ";\n"
    (path("data").parent / "docs").mkdir(exist_ok=True)
    (path("data").parent / "docs" / "data.js").write_text(js, encoding="utf-8")
    return D


def hardcoded_numbers():
    """The Checkpoint 9 grep: numbers in the visible text of docs/index.html (scripts and styles removed)."""
    html = (path("data").parent / "docs" / "index.html").read_text(encoding="utf-8")
    html = re.sub(r"<script.*?</script>|<style.*?</style>|<div class=\"kicker\">.*?</div>", "", html, flags=re.S)  # kickers = section numbering
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"EmbeddingGemma 2|Apache 2\.0", "", text)  # names, not findings
    text = re.sub(r"Finding \d", "", text)  # section labels
    # "How to read" captions explain chart conventions (95% intervals, 0% = guessing, −100% = all gone), not results
    html_caps = re.findall(r'<p class="mut cap"[^>]*>(.*?)</p>', html, flags=re.S)
    for cap in html_caps:
        text = text.replace(re.sub(r"<[^>]+>", " ", cap), "")
    return re.findall(r"\b\d+(?:[.,]\d+)?%?", text)
