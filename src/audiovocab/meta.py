"""Phase 2: fetch dataset metadata only (labels, licences, folds, sizes); no audio.

Everything lands in data/raw/meta/ (gitignored). Safe to rerun: finished files are skipped.
"""
import json

import requests
import truststore

from .config import env, path

truststore.inject_into_ssl()  # corporate TLS inspection

UA = {"User-Agent": "Mozilla/5.0 audio-vocab-research"}

FILES = {
    "esc50/esc50.csv": "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv",
    "esc50/LICENSE": "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/LICENSE",
    "esc50/README.md": "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/README.md",
    "fsd50k/FSD50K.ground_truth.zip": "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1",
    "fsd50k/FSD50K.metadata.zip": "https://zenodo.org/records/4060432/files/FSD50K.metadata.zip?download=1",
    "fsd50k/zenodo.json": "https://zenodo.org/api/records/4060432",
    "vocalsound/README.md": "https://raw.githubusercontent.com/YuanGongND/vocalsound/main/README.md",
    "cremad/README.md": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/README.md",
    "cremad/LICENSE.txt": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/LICENSE.txt",
    "cremad/VideoDemographics.csv": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv",
    "cremad/SentenceFilenames.csv": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/SentenceFilenames.csv",
    "cremad/tree.json": "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1",
    "ravdess/zenodo.json": "https://zenodo.org/api/records/1188976",
    "fma/fma_metadata.zip": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
    "fma/README.md": "https://raw.githubusercontent.com/mdeff/fma/master/README.md",
    "mtg/raw_30s_cleantags.tsv": "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags.tsv",
    "mtg/autotagging.tsv": "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags_50artists.tsv",
    "mtg/README.md": "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/README.md",
    "mtg/LICENSE": "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/LICENSE",
    "musopen/MusopenCollectionAsFlac.json": "https://archive.org/metadata/MusopenCollectionAsFlac",
    "songdescriber/zenodo.json": "https://zenodo.org/api/records/10072001",
}


def fetch(rel, url):
    f = path("raw") / "meta" / rel
    if f.exists() and f.stat().st_size:
        return f, "cached"
    f.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=UA, timeout=600, stream=True) as r:
        r.raise_for_status()
        tmp = f.with_suffix(f.suffix + ".part")
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
    tmp.replace(f)
    return f, "ok"


def xeno_canto(query, page=1):
    """One page of xeno-canto API v3 results (cached on disk by query + page)."""
    f = path("raw") / "meta" / "xenocanto" / f"{query.replace(':', '_').replace(' ', '+')}_p{page}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    r = requests.get("https://xeno-canto.org/api/3/recordings", headers=UA, timeout=60,
                     params={"query": query, "key": env("xeno_canto_api_key"), "page": page, "per_page": 500})
    r.raise_for_status()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    return r.json()


def run():
    status = {}
    for rel, url in FILES.items():
        try:
            status[rel] = fetch(rel, url)[1]
        except Exception as e:
            status[rel] = f"failed: {type(e).__name__}: {e}"
        print(f"{rel}: {status[rel]}", flush=True)
    return status
