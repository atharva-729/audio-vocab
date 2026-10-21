"""Build the offline bundle (models + feeds + trimmed audio) on a machine with open internet.

Run on the personal laptop, then copy bundle.zip to the work machine. Needs:
  Python 3.10+, ffmpeg/ffprobe on PATH, and `pip install huggingface_hub`.

  python fetch_bundle.py                 # everything, then bundle.zip
  python fetch_bundle.py --skip-models   # feeds + audio only

Safe to rerun: finished models/episodes are skipped.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path

import yaml

MODELS = ["google/embeddinggemma-2", "laion/clap-htsat-unfused", "Systran/faster-whisper-small"]
EPISODES_PER_SHOW = 3
MIN_EPISODE_MIN = 10
TRIM_S = 900  # first 15 minutes
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) vibe-search-prototype"
IT = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- models ----------

def fetch_models(out):
    from huggingface_hub import snapshot_download

    for repo in MODELS:
        dest = out / "models" / repo.replace("/", "--")
        if (dest / ".done").exists():
            log(f"model {repo}: already done")
            continue
        log(f"model {repo}: downloading ...")
        snapshot_download(repo, local_dir=dest)
        (dest / ".done").write_text("ok")
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
        log(f"model {repo}: {size:.0f} MB")


# ---------- feeds ----------

def duration_s(text):
    if not text:
        return None
    try:
        s = 0.0
        for part in text.strip().split(":"):
            s = s * 60 + float(part)
        return s
    except ValueError:
        return None


def parse_feed(show, xml_bytes):
    ch = ET.fromstring(xml_bytes).find("channel")
    cats = []
    for c in ch.findall(IT + "category"):
        cats.append(c.get("text"))
        cats += [f"{c.get('text')} > {sub.get('text')}" for sub in c.findall(IT + "category")]
    eps = []
    for it in ch.findall("item"):
        enc = it.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate")).isoformat()
        except Exception:
            pub = ""
        eps.append({
            "guid": (it.findtext("guid") or enc.get("url")).strip(),
            "title": (it.findtext("title") or "").strip(),
            "pub": pub,
            "duration_s": duration_s(it.findtext(IT + "duration")),
            "type": (it.findtext(IT + "episodeType") or "full").strip().lower(),
            "audio_url": enc.get("url"),
            "mime": enc.get("type", ""),
        })
    eps.sort(key=lambda e: e["pub"], reverse=True)
    full = [e for e in eps if e["type"] == "full"]
    picked = [e for e in full if (e["duration_s"] or 0) >= MIN_EPISODE_MIN * 60][:EPISODES_PER_SHOW]
    if len(picked) < EPISODES_PER_SHOW:  # some feeds omit itunes:duration
        picked += [e for e in full if e["duration_s"] is None][:EPISODES_PER_SHOW - len(picked)]
    for e in picked:
        e["id"] = f"{show['id']}_{hashlib.sha1(e['guid'].encode()).hexdigest()[:8]}"
    return {
        "show_id": show["id"], "show": (ch.findtext("title") or show["name"]).strip(), "vibe": show["vibe"],
        "spare": bool(show.get("spare")), "feed": show["feed"], "link": (ch.findtext("link") or "").strip(),
        "itunes_categories": cats, "n_items": len(eps), "episodes": picked,
    }


def fetch_feed(show, out):
    path = out / "feeds" / f"{show['id']}.xml"
    if not path.exists():
        req = urllib.request.Request(show["feed"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    try:
        return parse_feed(show, path.read_bytes())
    except ET.ParseError:
        path.unlink()  # don't cache an error page
        raise


# ---------- audio ----------

def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def fetch_audio(ep, out):
    """Copy the first TRIM_S seconds of the stream without re-encoding; fall back to FLAC."""
    audio_dir = out / "audio"
    for f in audio_dir.glob(ep["id"] + ".*"):
        if f.suffix != ".part":
            return {"file": f.name, "status": "cached"}
    is_mp3 = "mpeg" in ep["mime"] or ep["audio_url"].split("?")[0].lower().endswith(".mp3")
    attempts = [(".mp3" if is_mp3 else ".m4a", ["-c:a", "copy"]),
                (".flac", ["-c:a", "flac", "-ac", "1", "-ar", "48000"])]
    err = ""
    for ext, codec in attempts:
        tmp = audio_dir / f"{ep['id']}.part{ext}"
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-user_agent", UA, "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
               "-i", ep["audio_url"], "-t", str(TRIM_S), "-map", "0:a:0", "-vn", *codec, str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode == 0 and tmp.exists() and probe_duration(tmp) > 60:
            final = audio_dir / f"{ep['id']}{ext}"
            tmp.replace(final)
            return {"file": final.name, "status": "ok"}
        err = (r.stderr or "no output").strip().splitlines()[-1:] or ["?"]
        err = err[0][:200]
        tmp.unlink(missing_ok=True)
    return {"file": None, "status": f"failed: {err}"}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", default=str(Path(__file__).with_name("feeds.yaml")))
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. Install with:  winget install Gyan.FFmpeg  (then open a new terminal)")

    out = Path(args.out).resolve()
    for d in ("models", "feeds", "audio"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shows = yaml.safe_load(Path(args.feeds).read_text(encoding="utf-8"))["shows"]
    shutil.copy(args.feeds, out / "feeds.yaml")

    if not args.skip_models:
        fetch_models(out)

    failures, results = [], []
    for show in shows:
        try:
            res = fetch_feed(show, out)
            results.append(res)
            log(f"feed {show['id']}: {len(res['episodes'])} episodes, categories={res['itunes_categories'][:2]}")
            if len(res["episodes"]) < EPISODES_PER_SHOW:
                failures.append(f"{show['id']}: only {len(res['episodes'])} usable episodes")
        except Exception as e:
            failures.append(f"{show['id']}: feed failed: {type(e).__name__}: {e}")
            log(f"feed {show['id']}: FAILED {e}")

    jobs = [ep for res in results for ep in res["episodes"]]
    log(f"audio: {len(jobs)} episodes, first {TRIM_S // 60} min each ...")

    def run(ep):
        try:
            r = fetch_audio(ep, out)
        except Exception as e:
            r = {"file": None, "status": f"failed: {type(e).__name__}: {e}"}
        ep.update(r)
        log(f"audio {ep['id']}: {r['status']}")
        return ep

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, jobs))
    failures += [f"{ep['id']} ({ep['title'][:50]}): {ep['status']}" for ep in jobs if not ep["file"]]

    (out / "episodes.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")

    log("checksums ...")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts)
    manifest = {str(p.relative_to(out)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    ok = sum(1 for ep in jobs if ep["file"])
    audio_gb = sum(v["bytes"] for k, v in manifest.items() if k.startswith("audio/")) / 1e9
    total_gb = sum(v["bytes"] for v in manifest.values()) / 1e9
    log(f"done: {len(results)}/{len(shows)} feeds, {ok}/{len(jobs)} episodes, audio {audio_gb:.2f} GB, total {total_gb:.2f} GB")
    log(f"{len(failures)} problems listed in {out / 'failures.txt'}")

    if not args.no_zip:
        zip_path = out.with_suffix(".zip")
        log(f"zipping to {zip_path} ...")
        # Audio and weights are already compressed; store keeps zipping fast.
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in files + [out / "manifest.json"]:
                z.write(p, Path("bundle") / p.relative_to(out))
        log(f"zip ready: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB). Upload this to Drive.")


if __name__ == "__main__":
    main()
