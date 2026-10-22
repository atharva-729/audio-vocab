"""Every sentence on the page that carries a number, built from the reports (via page.build's D).

Wording rule (README): findings are phrased "in these datasets, with this model". Interpretive sentences are
chosen by explicit conditions on the numbers, so a rerun with different numbers can't keep a stale claim.
"""

SIB_NAMES = {
    "laughter_types": "Laughter: giggle vs chuckle", "owls": "Owls: 6 species", "cuckoo_vs_owls": "Cuckoo vs owls",
    "hooting_calls": "Hooting calls: cuckoo, owls, doves, hoopoe", "songbirds": "Songbirds: 5 species",
    "corvids": "Crow vs magpie", "vocal_nonspeech": "VocalSound: laugh, sigh, cough, throat, sneeze, sniff",
    "dog_sounds": "Dog: bark vs growl", "cat_sounds": "Cat: meow, purr, hiss", "loud_voice": "Scream vs shout",
    "voice_kinds": "Whisper vs sing vs cry", "speakers": "Man vs woman vs child speaking",
    "fsd_body": "Body sounds (FSD50K)", "fsd_birds": "Bird calls (FSD50K)", "fsd_weather": "Rain vs thunder vs wind (FSD50K)",
    "fsd_water": "Water: stream, waves, drip, trickle", "road_vehicles": "Car vs motorcycle vs bus vs truck",
    "signals": "Siren, horn, doorbell, ringtone, bike bell", "tools": "Drill vs saw vs hammer", "kitchen": "Kitchen sounds",
    "esc_pets": "Dog vs cat (ESC-50)", "esc_farm": "Farm animals (ESC-50)", "esc_birds": "Birds (ESC-50)",
    "esc_water": "Water sounds (ESC-50)", "esc_weather": "Rain vs wind vs thunderstorm (ESC-50)",
    "esc_body": "Body sounds (ESC-50)", "esc_door": "Door knock vs creak", "esc_computer": "Mouse click vs typing",
    "esc_appliances": "Washing machine vs vacuum", "esc_clocks": "Clocks and bells", "esc_vehicles": "Vehicles (ESC-50)",
    "esc_tools": "Chainsaw vs hand saw",
}
SIB_ORDER = ["laughter_types", "vocal_nonspeech", "esc_body", "fsd_body", "loud_voice", "voice_kinds", "speakers",
             "cuckoo_vs_owls", "owls", "hooting_calls", "songbirds", "corvids", "esc_birds", "fsd_birds", "dog_sounds",
             "cat_sounds", "esc_pets", "esc_farm", "esc_weather", "fsd_weather", "esc_water", "fsd_water",
             "road_vehicles", "esc_vehicles", "signals", "tools", "esc_tools", "kitchen", "esc_appliances", "esc_door",
             "esc_computer", "esc_clocks"]


def pct(x):
    return "–" if x is None else f"{round(100 * x)}%"


def sibling_rows(D):
    p4 = D.get("p4") or {}
    models = p4.get("models", {})
    rows = {}
    for m, mm in models.items():
        for ds, groups in mm["siblings"].items():
            for g, r in groups.items():
                row = rows.setdefault(g, {"group": g, "name": SIB_NAMES.get(g, g), "chance": r["chance"], "dataset": ds,
                                          "classes": r["classes"]})
                row[m] = r["probe"]
                row[m + "_zs"] = r["zero_shot"]
    return [rows[g] for g in SIB_ORDER if g in rows]


def fam_rows(D, model):
    return [r for r in (D.get("p4") or {}).get("families", []) if r["model"] == model]


def static_text(D):
    p4 = (D.get("p4") or {}).get("protocol", {})
    return {
        "protocol": [
            "<b>Frozen embeddings, nothing fine-tuned.</b> EmbeddingGemma 2 (768-d, float32) and LAION-CLAP (512-d) embed every clip; "
            "text with Gemma's search-query prefix (chosen on the tuning fold) or no prefix for CLAP.",
            f"<b>Grouped splits everywhere:</b> ESC-50's official folds; elsewhere 5 folds by a hash of source recording, speaker, recordist, uploader or artist. "
            f"Fold {p4.get('tune_fold', '–')} tunes choices (prefix, centring strength, word form); every reported number uses the other folds.",
            f"<b>Probe</b> = logistic regression, fixed C = {p4.get('probe_C', '–')}. <b>kNN</b> = {p4.get('knn_k', '–')} neighbours, never from the clip's own source. "
            "<b>Zero-shot</b> = nearest class text (class name + 2 descriptions, averaged).",
            f"<b>Uncertainty:</b> 95% cluster-bootstrap intervals over groups ({p4.get('bootstrap', '–')} resamples). "
            f"\"Told apart\" = pairwise probe accuracy ≥ {pct(p4.get('pairwise_threshold'))}; effective vocabulary = the largest set of classes all told apart (a maximum clique).",
            "<b>Windows:</b> Gemma silently truncates audio after 30 s, so sounds use ≤ 10 s and music uses 10 s windows (Phase 1: a 10 s window keeps most of the 30 s vector).",
        ],
        "limits": [
            "<b>Two models only</b>; CLAP may have seen ESC-50-like data in training, and Gemma's training data is not documented.",
            "<b>Dataset label noise</b>, and FSD50K labels propagate up the AudioSet tree (we only use clips without a rival sibling label).",
            "<b>FMA is indie/amateur music</b>; its \"classical\" isn't canonical (Musopen is the check).",
            "<b>The instrumental filter is imperfect</b> (Whisper on a 20 s window, threshold calibrated on held-out tracks).",
            "<b>Genre and region words are culturally loaded</b>, and FMA's regional tags are sparse once vocals are removed.",
            "<b>Correlation isn't cause</b>: the perturbations are the causal part of the genre analysis.",
            "<b>English words only</b>; one listener in the Vibe Search test being bridged.",
            "<b>Laughter subtypes</b>: only giggle and chuckle exist as labels in licence-clear data.",
        ],
        "repro": [
            "<code>vocab plan</code> → selection lists from metadata; <code>vocab fetch all</code> + <code>scripts/import_bundle.py</code> for the proxy-blocked sets.",
            "<code>vocab clips</code> → <code>data/clips.parquet</code>; <code>vocab embed</code> → <code>index/gemma.npy</code>, <code>index/clap.npy</code> (cached per clip).",
            "<code>vocab analyse</code> → <code>reports/phase3…phase8 *.json</code>; <code>vocab page</code> → <code>docs/data.js</code>.",
            "Every number on this page is read from <code>reports/*.json</code>; decisions are logged in <code>reports/decisions_log.md</code>.",
        ],
        "attributions": [
            "ESC-50: K. J. Piczak, ACM MM 2015, CC BY-NC 3.0 (clips from Freesound.org).",
            "FSD50K: E. Fonseca et al., IEEE/ACM TASLP 2022, CC BY 4.0; clips under their own CC licences (Freesound.org).",
            "VocalSound: Y. Gong, J. Yu, J. Glass, ICASSP 2022, CC BY-SA 4.0.",
            "xeno-canto: recordings by their recordists, each under its CC licence (xeno-canto.org).",
            "CREMA-D: H. Cao et al., IEEE Trans. Affective Computing 2014, ODbL 1.0 / DbCL 1.0.",
            "FMA: M. Defferrard et al., ISMIR 2017; metadata CC BY 4.0, audio under each artist's licence.",
            "MTG-Jamendo: D. Bogdanov et al., ICML 2019 ML4MD; non-commercial research use.",
            "Musopen: recordings by Musopen, Public Domain Mark, via the Internet Archive.",
        ],
    }


def findings(D):
    T = static_text(D)
    T["sibling_rows"] = sibling_rows(D)
    from .findings_text import narrative
    T.update(narrative(D, T))
    return T
