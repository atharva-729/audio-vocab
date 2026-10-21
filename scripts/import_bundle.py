"""Unpack bundle.zip (VocalSound + MTG-Jamendo shards), verify every checksum, and move files into data/raw/.

  data/raw/vocalsound/    vs_release_16k.zip, README.md, LICENSE
  data/raw/mtg-jamendo/   raw_30s audio-low shard tars + checksum file

Usage:  python scripts/import_bundle.py [path/to/bundle.zip]
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(zip_path=ROOT / "bundle.zip"):
    staging = ROOT / "bundle"
    if not staging.exists():
        print(f"unzipping {zip_path} ...", flush=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ROOT)

    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for rel, meta in manifest.items():
        p = staging / rel
        if not p.exists() or p.stat().st_size != meta["bytes"] or sha256(p) != meta["sha256"]:
            bad.append(rel)
    print(f"verified {len(manifest) - len(bad)}/{len(manifest)} files")
    if bad:
        sys.exit("checksum mismatch:\n  " + "\n  ".join(bad))

    expected = ["vocalsound/vs_release_16k.zip", "mtg-jamendo/raw_30s_audio-low-53.tar",
                "mtg-jamendo/raw_30s_audio-low-86.tar"]
    missing = [e for e in expected if e not in manifest]
    if missing:
        print("WARNING: expected files not in bundle (names may differ; check):\n  " + "\n  ".join(missing))

    for sub in ("vocalsound", "mtg-jamendo"):
        src, dst = staging / sub, ROOT / "data" / "raw" / sub
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), target)
    shutil.rmtree(staging)
    print("imported into data/raw/; bundle.zip can be deleted (and removed from Drive)")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
