"""Phase 3 step 4: the text side. One table of every text the analyses use, embedded by both models.

index/text.parquet columns: text, kind (class_name | class_desc | genre | region | synonym | emotion | descriptor |
generic), dataset, target (the label/word it stands for), form (bare | phrase | sentence | desc), concept.
Gemma is embedded twice: with the SearchQuery prefix (index/text_gemma.npy) and without any prefix
(index/text_gemma_none.npy); Phase 4 picks one on the tuning fold. CLAP has no prefixes (index/text_clap.npy).
"""
import json

import numpy as np
import pandas as pd
import yaml

from .centering import GENERIC_PHRASES
from .config import ROOT, cfg, path


def clean(label):
    return label.replace("_and_", " and ").replace("_", " ").replace("(", "").replace(")", "").strip().lower()


def build_table():
    tax = yaml.safe_load((ROOT / "data" / "taxonomy.yaml").read_text(encoding="utf-8"))
    rows = []
    for ds in ("esc50", "fsd50k", "vocalsound", "xenocanto", "cremad"):
        for label, spec in tax[ds].items():
            name = spec.get("word") or clean(label)
            rows.append({"text": name, "kind": "class_name", "dataset": ds, "target": label, "form": "bare"})
            for d in spec.get("text", []):
                rows.append({"text": d, "kind": "class_desc", "dataset": ds, "target": label, "form": "desc"})
    forms = dict(zip(("bare", "phrase", "sentence"), tax["music"]["word_forms"]))
    for kind, mapping in (("genre", tax["music"]["genres"]), ("region", tax["music"]["regions"])):
        for word in sorted(set(mapping.values())):
            for form, tpl in forms.items():
                rows.append({"text": tpl.format(w=word), "kind": kind, "dataset": "music", "target": word, "form": form})
    for concept, spec in cfg()["synonyms"].items():
        for w in spec["words"]:
            rows.append({"text": w, "kind": "synonym", "dataset": "", "target": concept, "form": "bare"})
    for code, prompts in cfg()["emotion_prompts"].items():
        for p in prompts:
            rows.append({"text": p, "kind": "emotion", "dataset": "cremad", "target": code, "form": "desc"})
    for group, words in cfg()["descriptors"].items():
        for w in words:
            rows.append({"text": w, "kind": "descriptor", "dataset": "music", "target": group, "form": "bare"})
    for p in GENERIC_PHRASES:
        rows.append({"text": p, "kind": "generic", "dataset": "", "target": "", "form": "desc"})
    return pd.DataFrame(rows)


def run():
    from . import embed
    t = build_table()
    path("index").mkdir(exist_ok=True)
    t.to_parquet(path("index") / "text.parquet", index=False)
    texts = t.text.tolist()
    g = embed.load("gemma", text_only=True)
    np.save(path("index") / "text_gemma.npy", g.text(texts, prompt_name="SearchQuery"))
    np.save(path("index") / "text_gemma_none.npy", g.text(texts, prompt_name=None))
    del g
    c = embed.load("clap")
    np.save(path("index") / "text_clap.npy", c.text(texts))
    return t
