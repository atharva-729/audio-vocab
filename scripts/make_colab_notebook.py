"""Generate colab/fsd50k_embed.ipynb: a self-contained Colab notebook that downloads FSD50K eval from Zenodo
(blocked on the work network), extracts our 2,399 selected clips, embeds them with the same models and the same
preprocessing as src/audiovocab/embed.py, and writes one small file to bring back: fsd50k_colab.npz.

Inlined: the clip selection, the local weights' SHA-256, and reference vectors for 3 ESC-50 clips (a parity check).
Run:  .venv/Scripts/python.exe scripts/make_colab_notebook.py
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audiovocab import cache, embed, splitzip  # noqa: E402

WEIGHT_SHA = {
    "google/embeddinggemma-2": ("model.safetensors", "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"),
    "laion/clap-htsat-unfused": ("pytorch_model.bin", "1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb"),
}


def reference_vectors():
    clips = pd.read_parquet(ROOT / "data" / "clips.parquet")
    esc = clips[clips.dataset == "esc50"].sort_values("clip_id").iloc[[0, 700, 1400]]
    refs = []
    for r in esc.to_dict("records"):
        k = embed.clip_key(r)
        g, c = cache.get_array("emb_gemma", k), cache.get_array("emb_clap", k)
        refs.append({"file": r["source"].rsplit("/", 1)[1], "gemma": np.round(g, 7).tolist(), "clap": np.round(c, 7).tolist()})
    return refs


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main():
    sel = pd.read_parquet(ROOT / "data" / "selection" / "fsd50k.parquet")
    selection = [[str(c), str(l)] for c, l in zip(sel["clip"], sel["label"])]
    refs = reference_vectors()
    split_src = inspect.getsource(splitzip).split('"""', 2)[2]  # module body without its docstring

    cells = [
        cell("""
# FSD50K → embeddings (audio-vocab)

Zenodo blocks the work network, so this notebook does the FSD50K part of audio-vocab in Colab:
1. downloads FSD50K **eval** audio from Zenodo (6.3 GB);
2. extracts the **2,399 clips** selected in `data/selection/fsd50k.parquet` (inlined below);
3. embeds them with **EmbeddingGemma 2** and **CLAP**, using exactly the preprocessing of `src/audiovocab/embed.py`;
4. writes **`fsd50k_colab.npz`** (a few MB). Put that file in the repo root and tell Claude.

**How to run:** Runtime → Change runtime type → **T4 GPU** → Runtime → **Run all**. Allow Google Drive access when asked.

**Timeouts:** everything is saved to Google Drive as it goes (`MyDrive/audio-vocab-fsd50k/`). If Colab disconnects,
just **Run all** again: the download resumes and finished clips are skipped. Keep the tab open.

Checks built in: the downloaded model weights are compared with the SHA-256 of the files used locally, and 3 ESC-50
clips are re-embedded and compared with the vectors computed on the work laptop.
""", "markdown"),
        cell("""
USE_DRIVE = True   # set False to work in /content only (no resume after a disconnect)
import os, subprocess, json, time, hashlib
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/audio-vocab-fsd50k'
else:
    WORK = '/content/audio-vocab-fsd50k'
os.makedirs(WORK, exist_ok=True)
LOCAL = '/content/fsd_local'   # big downloads go to the VM disk first, then are copied to Drive in parts
os.makedirs(LOCAL, exist_ok=True)
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'NO GPU: switch the runtime to T4 GPU')
"""),
        cell("""
!pip -q install "transformers==5.20.2" "sentence-transformers==6.1.0" soundfile
import torch, transformers, sentence_transformers
print('torch', torch.__version__, '| transformers', transformers.__version__, '| sentence-transformers', sentence_transformers.__version__, '| cuda', torch.cuda.is_available())
"""),
        cell("SELECTION = " + json.dumps(selection) + "\nREFS = " + json.dumps(refs) + "\nWEIGHT_SHA = " + json.dumps(WEIGHT_SHA) +
             "\nprint(len(SELECTION), 'clips selected;', len(REFS), 'reference clips')"),
        cell("""
# Models (same repos as the work laptop); weights checked against the local files' SHA-256
from huggingface_hub import snapshot_download
MODELS = {}
for repo, (fname, sha) in WEIGHT_SHA.items():
    d = snapshot_download(repo, local_dir=f'{WORK}/models/{repo.replace("/", "--")}')
    h = hashlib.sha256()
    with open(f'{d}/{fname}', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    ok = h.hexdigest() == sha
    print(repo, 'weights match the work laptop:', ok)
    if not ok:
        print('  WARNING: different weights than the local copy; the parity check below will show whether it matters')
    MODELS[repo] = d
"""),
        cell("""
# FSD50K eval audio: resumable download in 64 MB ranges, then copied to Drive
import requests
BASE = 'https://zenodo.org/records/4060432/files/'
def fetch(name, chunk=64 << 20):
    dst_drive = f'{WORK}/{name}'
    if os.path.exists(dst_drive + '.done'):
        return dst_drive
    dst = f'{LOCAL}/{name}'
    url = BASE + name + '?download=1'
    size = int(requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=60).headers['content-range'].split('/')[-1])
    pos = os.path.getsize(dst) if os.path.exists(dst) else 0
    t0 = time.time()
    with open(dst, 'ab') as f:
        while pos < size:
            end = min(pos + chunk, size) - 1
            for attempt in range(20):
                try:
                    r = requests.get(url, headers={'Range': f'bytes={pos}-{end}'}, timeout=300)
                    if r.status_code == 206 and len(r.content) == end - pos + 1:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f'download failed at {pos}')
            f.write(r.content); pos = end + 1
            print(f'\\r{name}: {pos/1e9:.2f}/{size/1e9:.2f} GB ({pos/1e6/(time.time()-t0+1e-9):.0f} MB/s)', end='')
    print()
    subprocess.run(['cp', dst, dst_drive], check=True)
    open(dst_drive + '.done', 'w').close()
    return dst_drive
PARTS = [fetch('FSD50K.eval_audio.z01'), fetch('FSD50K.eval_audio.zip')]
"""),
        cell("# Split-zip reader (copied from src/audiovocab/splitzip.py)\nimport io, zipfile\n" + split_src),
        cell("""
# Extract only the selected clips
AUD = f'{WORK}/audio'
os.makedirs(AUD, exist_ok=True)
want = {c for c, _ in SELECTION}
todo = [c for c in want if not os.path.exists(f'{AUD}/{c}.wav')]
if todo:
    local_parts = []
    for p in PARTS:   # read from the VM disk if it's still there (much faster than Drive)
        lp = f'{LOCAL}/{os.path.basename(p)}'
        local_parts.append(lp if os.path.exists(lp) else p)
    with SplitZip(local_parts) as z:
        names = {n.rsplit('/', 1)[-1][:-4]: n for n in z.namelist() if n.endswith('.wav')}
        for c in todo:
            with open(f'{AUD}/{c}.wav', 'wb') as f:
                f.write(z.read(names[c]))
print(sum(os.path.exists(f'{AUD}/{c}.wav') for c in want), 'of', len(want), 'clips extracted')
"""),
        cell("""
# The embedding code of src/audiovocab/embed.py, on GPU, float32 (never fp16)
import numpy as np, soundfile as sf
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
def decode(file, sr, start=None, dur=None):
    cmd = ['ffmpeg', '-nostdin', '-v', 'error']
    if start is not None:
        cmd += ['-ss', str(start)]
    cmd += ['-i', str(file)]
    if dur is not None:
        cmd += ['-t', str(dur)]
    cmd += ['-ac', '1', '-ar', str(sr), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)
def _l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)
class Gemma:
    name, sr = 'gemma', 16000
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(MODELS['google/embeddinggemma-2'], device=DEV, config_kwargs={'vision_config': None},
                                     model_kwargs={'dtype': torch.float32})
    def audio(self, clips):
        return _l2(self.m.encode([{'array': c, 'sampling_rate': self.sr} for c in clips], batch_size=4))
class Clap:
    name, sr, window_s = 'clap', 48000, 10
    def __init__(self):
        from transformers import ClapModel, ClapProcessor
        self.m = ClapModel.from_pretrained(MODELS['laion/clap-htsat-unfused']).eval().to(DEV)
        self.p = ClapProcessor.from_pretrained(MODELS['laion/clap-htsat-unfused'])
    @torch.inference_mode()
    def audio(self, clips):
        w = self.window_s * self.sr
        wins = [c[i:i + w] for c in clips for i in range(0, max(len(c) - w // 2, 1), w)]
        counts = [len(range(0, max(len(c) - w // 2, 1), w)) for c in clips]
        inp = {k: v.to(DEV) for k, v in self.p(audio=wins, sampling_rate=self.sr, return_tensors='pt').items()}
        o = self.m.get_audio_features(**inp)
        o = o.pooler_output if hasattr(o, 'pooler_output') else o
        e = _l2(o.cpu().numpy())
        out, i = [], 0
        for n in counts:
            out.append(e[i:i + n].mean(0)); i += n
        return _l2(out)
"""),
        cell("""
# Parity: 3 ESC-50 clips embedded here vs the work laptop's vectors (GPU float32 vs CPU float32)
PAR = {}
for M in (Gemma, Clap):
    m = M()
    for r in REFS:
        url = 'https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/' + r['file']
        p = f'{LOCAL}/{r["file"]}'
        if not os.path.exists(p):
            open(p, 'wb').write(requests.get(url, timeout=60).content)
        v = m.audio([decode(p, m.sr, None, 5.0)])[0]
        PAR.setdefault(m.name, []).append(float(v @ np.asarray(r[m.name], np.float32)))
    del m; torch.cuda.empty_cache()
print('cosine to the work laptop vectors:', {k: [round(x, 5) for x in v] for k, v in PAR.items()})
print('PARITY OK' if min(min(v) for v in PAR.values()) > 0.999 else 'PARITY WARNING: tell Claude these numbers')
"""),
        cell("""
# Embed the selected clips (window: first min(10 s, clip length)); saved per clip, so a rerun resumes
DUR = {c: round(min(10.0, sf.info(f'{AUD}/{c}.wav').duration), 3) for c, _ in SELECTION}
for M in (Gemma, Clap):
    m = M()
    od = f'{WORK}/emb_{m.name}'; os.makedirs(od, exist_ok=True)
    todo = [c for c, _ in SELECTION if not os.path.exists(f'{od}/{c}.npy')]
    t0 = time.time()
    for j in range(0, len(todo), 16):
        batch = todo[j:j + 16]
        vecs = m.audio([decode(f'{AUD}/{c}.wav', m.sr, None, DUR[c]) for c in batch])
        for c, v in zip(batch, vecs):
            np.save(f'{od}/{c}.npy', v)
        print(f'\\r{m.name}: {j + len(batch)}/{len(todo)} ({time.time() - t0:.0f} s)', end='')
    print(); del m; torch.cuda.empty_cache()
"""),
        cell("""
# The one file to bring back: put fsd50k_colab.npz in the audio-vocab repo root
ids = [c for c, _ in SELECTION]
np.savez(f'{WORK}/fsd50k_colab.npz', ids=np.array(ids), labels=np.array([l for _, l in SELECTION]),
         dur=np.array([DUR[c] for c in ids]),
         gemma=np.stack([np.load(f'{WORK}/emb_gemma/{c}.npy') for c in ids]),
         clap=np.stack([np.load(f'{WORK}/emb_clap/{c}.npy') for c in ids]),
         meta=json.dumps({'parity': PAR, 'torch': torch.__version__, 'transformers': transformers.__version__,
                          'sentence_transformers': sentence_transformers.__version__, 'device': DEV,
                          'gpu': subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip()}))
print('saved', f'{WORK}/fsd50k_colab.npz', round(os.path.getsize(f'{WORK}/fsd50k_colab.npz') / 1e6, 1), 'MB')
from google.colab import files
files.download(f'{WORK}/fsd50k_colab.npz')
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "colab" / "fsd50k_embed.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(out, round(out.stat().st_size / 1e3), "KB")


if __name__ == "__main__":
    main()
