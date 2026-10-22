"""Phase 3 downloads (reachable hosts only; VocalSound and MTG-Jamendo come from the bundle).

Every fetcher is resumable: finished files are skipped, partial files end in .part. Audio lands in data/raw/<dataset>/.

  vocab fetch esc50 | fsd50k | cremad | xenocanto | fma | musopen | all
"""
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import truststore

from .config import path

truststore.inject_into_ssl()  # corporate TLS inspection

UA = {"User-Agent": "Mozilla/5.0 audio-vocab-research"}


def raw(*parts):
    return path("raw").joinpath(*parts)


def log(name, msg):
    line = f"{time.strftime('%H:%M:%S')} {name}: {msg}"
    print(line, flush=True)
    f = path("runs") / f"fetch_{name}.log"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def download(url, dest, session=None, timeout=120, retries=3):
    if dest.exists() and dest.stat().st_size:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    s = session or requests
    for attempt in range(retries):
        try:
            start = tmp.stat().st_size if tmp.exists() else 0
            h = {**UA, **({"Range": f"bytes={start}-"} if start else {})}
            with s.get(url, headers=h, stream=True, timeout=timeout) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if start and r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    tmp.replace(dest)
    return "ok"


def download_ranged(url, dest, name, chunk=64 << 20, retries=30):
    """Large files as a sequence of fixed-size range requests. The proxy holds back long single responses
    (a 3 GB stream delivered 0 bytes in 8 min), but completes bounded ranges."""
    if dest.exists() and dest.stat().st_size:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    s = requests.Session()
    r = s.get(url, headers={**UA, "Range": "bytes=0-0"}, timeout=120)
    r.raise_for_status()
    size = int(r.headers["content-range"].rsplit("/", 1)[1])
    pos, fails, t0 = (tmp.stat().st_size if tmp.exists() else 0), 0, time.time()
    with open(tmp, "ab") as f:
        while pos < size:
            end = min(pos + chunk, size) - 1
            try:
                r = s.get(url, headers={**UA, "Range": f"bytes={pos}-{end}"}, timeout=600)
                if r.status_code != 206 or len(r.content) != end - pos + 1:
                    raise IOError(f"HTTP {r.status_code}, {len(r.content)} bytes")
                f.write(r.content)
                f.flush()
                pos = end + 1
                fails = 0
                if (pos // chunk) % 8 == 0 or pos == size:
                    log(name, f"{dest.name}: {pos / 1e9:.2f}/{size / 1e9:.2f} GB "
                              f"({pos / 1e6 / max(time.time() - t0, 1):.1f} MB/s this session)")
            except (requests.RequestException, IOError) as e:
                fails += 1
                if fails > retries:
                    raise
                log(name, f"{dest.name}: retry {fails} at {pos}: {e}")
                time.sleep(min(60, 5 * fails))
    tmp.replace(dest)
    return "ok"


def _pool(name, items, fn, workers):
    done = fail = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(fn, items), 1):
            if res is True:
                done += 1
            else:
                fail += 1
                log(name, f"FAILED {res}")
            if i % 200 == 0 or i == len(items):
                log(name, f"{i}/{len(items)} ({fail} failed)")
    return done, fail


def sel(name):
    return pd.read_parquet(path("data") / "selection" / f"{name}.parquet")


# ---------- per dataset ----------

def esc50():
    z = raw("esc50", "ESC-50-master.zip")
    log("esc50", download("https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master", z, timeout=600))
    out = raw("esc50", "audio")
    if not out.exists() or len(list(out.glob("*.wav"))) < 2000:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.startswith("ESC-50-master/audio/") and m.endswith(".wav"):
                    t = out / m.rsplit("/", 1)[1]
                    if not t.exists():
                        t.parent.mkdir(parents=True, exist_ok=True)
                        t.write_bytes(zf.read(m))
    log("esc50", f"{len(list(out.glob('*.wav')))} wav files")


def fsd50k():
    base = "https://zenodo.org/records/4060432/files/"
    for part in ("FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip", "FSD50K.doc.zip"):
        log("fsd50k", f"{part}: {download_ranged(base + part + '?download=1', raw('fsd50k', part), 'fsd50k')}")
    extract_fsd50k()


def extract_fsd50k():
    """Extract the selected clips from the two-part split zip (.z01 + .zip) without 7-Zip.

    Python's zipfile can't open split archives, so the parts are read through a virtual concatenation and each
    member's local-header offset is corrected by the size of the disks before its own."""
    from .splitzip import SplitZip
    want = set(sel("fsd50k")["clip"].astype(str))
    out = raw("fsd50k", "audio")
    out.mkdir(parents=True, exist_ok=True)
    todo = [w for w in want if not (out / f"{w}.wav").exists()]
    if not todo:
        log("fsd50k", f"{len(want)} clips already extracted")
        return
    with SplitZip([raw("fsd50k", "FSD50K.eval_audio.z01"), raw("fsd50k", "FSD50K.eval_audio.zip")]) as z:
        names = {n.rsplit("/", 1)[-1][:-4]: n for n in z.namelist() if n.endswith(".wav")}
        for w in todo:
            (out / f"{w}.wav").write_bytes(z.read(names[w]))
    log("fsd50k", f"extracted {len(todo)} clips ({len(want)} selected)")


def cremad():
    s = requests.Session()
    clips = sel("cremad")["clip"].tolist()
    url = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/{}.wav"

    def one(c):
        try:
            dest = raw("cremad", "AudioWAV", f"{c}.wav")
            download(url.format(c), dest, session=s)
            if dest.stat().st_size < 1000:  # an LFS pointer or error page, not audio
                dest.unlink()
                return f"{c}: tiny file"
            return True
        except Exception as e:
            return f"{c}: {e}"
    log("cremad", f"done={_pool('cremad', clips, one, 8)}")


def xenocanto():
    s = requests.Session()
    df = sel("xenocanto")

    def one(row):
        try:
            download(row.url, raw("xenocanto", f"XC{row.clip}.mp3"), session=s)
            return True
        except Exception as e:
            return f"XC{row.clip}: {e}"
    log("xenocanto", f"done={_pool('xenocanto', list(df.itertuples()), one, 4)}")


def fma():
    from .remotezip import open_remote_zip
    ids = pd.concat([sel("fma"), sel("fma_calibration")])["clip"].astype(str).drop_duplicates().tolist()
    url = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
    todo = [i for i in ids if not raw("fma", f"{int(i):06d}.mp3").exists()]
    log("fma", f"{len(ids)} tracks selected, {len(todo)} to fetch")
    if not todo:
        return
    import threading
    local = threading.local()

    def one(i):
        try:
            if not hasattr(local, "z"):
                local.z = open_remote_zip(url)
            tid = f"{int(i):06d}"
            data = local.z.read(f"fma_large/{tid[:3]}/{tid}.mp3")
            dest = raw("fma", f"{tid}.mp3")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except KeyError:
            return f"{i}: not in fma_large.zip"
        except Exception as e:
            if hasattr(local, "z"):
                del local.z
            return f"{i}: {type(e).__name__}: {e}"
    _pool("fma", todo, one, 6)


def musopen():
    """Full FLAC per selected movement (Python download: ffmpeg's own TLS rejects the proxy's certificate), then two
    10 s windows at 30% and 60% of its length, cut locally to mono FLAC (lossless). The full file is then deleted."""
    df = sel("musopen")
    base = "https://archive.org/download/MusopenCollectionAsFlac/"

    def one(row):
        try:
            stem = row.clip.replace("/", "__")[:-5]
            wins = [raw("musopen", f"{stem}_w{k}.flac") for k in (0, 1)]
            if all(w.exists() for w in wins):
                return True
            full = raw("musopen", "full", f"{stem}.flac")
            download_ranged(base + requests.utils.quote(row.clip), full, "musopen", chunk=32 << 20)
            for w, frac in zip(wins, (0.3, 0.6)):
                tmp = w.with_name(w.stem + ".part.flac")
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{row.length * frac:.2f}",
                                "-i", str(full), "-t", "10", "-ac", "1", "-c:a", "flac", str(tmp)],
                               check=True, capture_output=True, timeout=600)
                tmp.replace(w)
            full.unlink()
            return True
        except Exception as e:
            return f"{row.clip}: {e}"
    log("musopen", f"done={_pool('musopen', list(df.itertuples()), one, 3)}")


def import_colab_fsd(npz_path):
    """Fold fsd50k_colab.npz (from colab/fsd50k_embed.ipynb) into the project: durations for clips.parquet, and
    every vector into the per-clip embedding cache under the same key a local run would use."""
    import json

    import numpy as np

    from . import cache, datasets, embed
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    log("fsd50k", f"colab file: {len(z['ids'])} clips, parity {meta['parity']}, {meta['device']} {meta.get('gpu', '')}")
    if min(min(v) for v in meta["parity"].values()) < 0.999:
        raise SystemExit("parity below 0.999: the Colab vectors don't match the local pipeline; not importing")
    durs = {str(i): float(d) for i, d in zip(z["ids"], z["dur"])}
    raw("fsd50k").mkdir(parents=True, exist_ok=True)
    raw("fsd50k", "colab_durations.json").write_text(json.dumps(durs))
    df = datasets.build()
    fsd = df[df.dataset == "fsd50k"]
    idx = {str(i): k for k, i in enumerate(z["ids"])}
    for r in fsd.to_dict("records"):
        k = embed.clip_key(r)
        cache.put_array("emb_gemma", k, z["gemma"][idx[str(r["item_id"])]].astype(np.float32))
        cache.put_array("emb_clap", k, z["clap"][idx[str(r["item_id"])]].astype(np.float32))
    log("fsd50k", f"imported {len(fsd)} FSD50K clips into clips.parquet and the embedding cache")
    return len(fsd)


FETCHERS = {"esc50": esc50, "cremad": cremad, "xenocanto": xenocanto, "fma": fma, "musopen": musopen,
            "fsd50k": fsd50k}


def run(name):
    if name == "all":
        with ThreadPoolExecutor(len(FETCHERS)) as ex:
            list(ex.map(lambda f: f(), FETCHERS.values()))
    else:
        FETCHERS[name]()
