"""Build the offline dataset bundle on a machine with open internet (the work proxy blocks these hosts).

Contents (and nothing else):
  bundle/vocalsound/   vs_release_16k.zip (as downloaded), README.md, LICENSE
  bundle/mtg-jamendo/  raw_30s audio-low shards for track folders 53 and 86 (as downloaded), checksum file
  bundle/manifest.json {relative path: {bytes, sha256}}
then bundle.zip (stored, zip64). Safe to rerun: verified files are skipped.

Needs Python 3.10+ and `pip install requests`. Usage:  python fetch_bundle.py [--out bundle] [--no-zip]
Keep bundle.zip PRIVATE on Google Drive.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

VOCALSOUND_ZIP = "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1"
VOCALSOUND_RAW = "https://raw.githubusercontent.com/YuanGongND/vocalsound/main/"
MTG_RAW = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
MTG_SHARDS = ["53", "86"]
EXPECTED_SHA = {  # from data/download/raw_30s_audio-low_sha256_tars.txt in the MTG repo, 2026-10-22
    "raw_30s_audio-low-53.tar": "9913b3105212b75f4779b375c6659da98f54f31dbf526fc8a8191ad9be027f54",
    "raw_30s_audio-low-86.tar": "9fda026a125fb878fc20e23ba543d6d16084d842fbcd5c68f53e6196796e8da8",
}
UA = {"User-Agent": "Mozilla/5.0 audio-vocab-bundle"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expect_sha=None):
    if dest.exists() and (expect_sha is None or sha256(dest) == expect_sha):
        log(f"{dest.name}: already done")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"{dest.name}: downloading {url}")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if expect_sha and sha256(tmp) != expect_sha:
        tmp.unlink()
        sys.exit(f"{dest.name}: checksum mismatch")
    tmp.replace(dest)
    log(f"{dest.name}: {dest.stat().st_size / 1e9:.2f} GB")


def mtg_shard_urls():
    """Mirrors and tar checksums, as used by the official scripts/download/download.py (mtg-fast, then mtg)."""
    script = requests.get(MTG_RAW + "scripts/download/download.py", headers=UA, timeout=60).text
    bases = re.findall(r"['\"](https://(?:cdn\.freesound\.org|essentia\.upf\.edu)/[^'\"]*mtg-jamendo/)['\"]", script)
    bases = list(dict.fromkeys(b.rstrip("/") for b in bases)) or [
        "https://cdn.freesound.org/mtg-jamendo", "https://essentia.upf.edu/documentation/datasets/mtg-jamendo"]
    sums_txt = requests.get(MTG_RAW + "data/download/raw_30s_audio-low_sha256_tars.txt", headers=UA, timeout=60)
    sums_txt.raise_for_status()
    sums = dict(line.split()[::-1] for line in sums_txt.text.splitlines() if line.strip())
    return bases, sums, sums_txt.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()

    vs = out / "vocalsound"
    download(VOCALSOUND_ZIP, vs / "vs_release_16k.zip")
    for name in ("README.md", "LICENSE"):
        r = requests.get(VOCALSOUND_RAW + name, headers=UA, timeout=60)
        if r.ok:
            (vs / name).write_bytes(r.content)
    with zipfile.ZipFile(vs / "vs_release_16k.zip") as z:
        wavs = [n for n in z.namelist() if n.endswith(".wav")]
        log(f"vocalsound: zip opens, {len(wavs)} wav files")

    bases, sums, sums_txt = mtg_shard_urls()
    mt = out / "mtg-jamendo"
    mt.mkdir(parents=True, exist_ok=True)
    (mt / "raw_30s_audio-low_sha256_tars.txt").write_text(sums_txt, encoding="utf-8")
    for shard in MTG_SHARDS:
        name = f"raw_30s_audio-low-{shard}.tar"
        assert sums.get(name) == EXPECTED_SHA[name], f"{name}: checksum list changed upstream"
        for i, base in enumerate(bases):
            try:
                download(f"{base}/raw_30s/audio-low/{name}", mt / name, sums[name])
                break
            except requests.RequestException as e:
                log(f"{name}: {base} failed ({e})" + ("; trying next mirror" if i + 1 < len(bases) else ""))
        else:
            sys.exit(f"{name}: all mirrors failed")

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".part"))
    manifest = {str(p.relative_to(out)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"manifest: {len(manifest)} files, {sum(v['bytes'] for v in manifest.values()) / 1e9:.2f} GB")

    if not args.no_zip:
        zp = out.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in files + [out / "manifest.json"]:
                z.write(p, Path("bundle") / p.relative_to(out))
        log(f"zip ready: {zp} ({zp.stat().st_size / 1e9:.2f} GB). Upload to Google Drive (private).")


if __name__ == "__main__":
    main()
