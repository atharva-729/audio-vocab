"""Unpack bundle.zip (built by fetch_bundle.py), verify checksums, and move files into place.

  models/<org>--<name>/      model weights
  data/audio_cache/raw/      trimmed episode audio (first 15 min, original codec)
  data/rss/                  raw feed XML
  data/bundle_episodes.json  feed metadata + picked episodes
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

    moves = {
        staging / "models": ROOT / "models",
        staging / "audio": ROOT / "data" / "audio_cache" / "raw",
        staging / "feeds": ROOT / "data" / "rss",
    }
    for src, dst in moves.items():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), target)
    shutil.copy(staging / "episodes.json", ROOT / "data" / "bundle_episodes.json")
    shutil.copy(staging / "failures.txt", ROOT / "runs" / "bundle_failures.txt")
    shutil.rmtree(staging)
    print("imported; bundle.zip can be deleted")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
