# Environment and gotchas

Copied from section 4 of `podcast-vibe-search/PROJECT_STATUS.md` (commit `4facfd4`). Additions for this project are
marked **(audio-vocab)**.

- **Machine:** Windows 11 work laptop, 12-core CPU, no GPU. Shells: PowerShell and Git Bash.
- **Python 3.14** in `.venv`. All dependencies installed fine, including torch 2.14 CPU, transformers 5.20,
  sentence-transformers 6.1, faster-whisper 1.2.1 and librosa 1.0. `pip install -e .` provides the CLI (`vocab`
  here, `vibe` in the sister).
- **Corporate TLS inspection:** Python 3.13+ rejects the proxy's CA (VERIFY_X509_STRICT). The code uses
  `truststore.inject_into_ssl()` (in `llm.py` and any downloader). pip works as-is.
- **Extra dependencies:** `pillow` and `torchvision`, because EmbeddingGemma 2's processor loads its image and video
  parts even when unused. `truststore` is also added. **(audio-vocab)** `plotly` added (named in this README);
  `faster-whisper` kept for the instrumental filter.
- **EmbeddingGemma 2 must run in float32 on CPU.** It loads in bfloat16 by default, which is **12× slower** here. Never
  use fp16 (NaNs, per the model card). Load with `config_kwargs={"vision_config": None}` (audio and text) or also
  `"audio_config": None` (text only, for queries).
- **Models live in the sister's `models/`** (not copied; absolute paths in `config.yaml`):
  `google--embeddinggemma-2`, `laion--clap-htsat-unfused` and `Systran--faster-whisper-small`. Run with
  `HF_HUB_OFFLINE=1` (`config.py` sets it by default).
- **LLM:** LLM Foundry at **`https://llmfoundry.straivedemo.com/openai/v1`** (the `.straive.com` host returns a
  "down" page), model **`gpt-5-mini`** with `reasoning_effort: minimal`, and the token in `LLMFOUNDRY_TOKEN` (sent as
  `Bearer <token>:audio-vocab`). The spend guard is $0.80/day, logged in `runs/spend.json`. Calls are cached in
  `runs/cache/`. Not expected to be needed.
- **Long jobs:** run them detached (`Start-Process`). The sister's overnight run used `runs/overnight.sh`, which also
  blocks sleep via `SetThreadExecutionState`.
- **Headless screenshots:** Edge headless needs its own `--user-data-dir` while the user has Edge open, and any local
  server must be running.
- **Voice:** the user dictates with Windows `Win+H`. Expect rambling, dictated messages and interpret them charitably.

## Network: what the proxy blocks (audio-vocab, probed 2026-10-22)

Probed with ranged GETs from the work laptop. Never try to bypass the proxy; blocked items go through the
personal-laptop bundle (`scripts/fetch_bundle.py` → Google Drive → `scripts/import_bundle.py`).

| Host | Status | Used for |
|---|---|---|
| github.com, codeload, raw.githubusercontent, media.githubusercontent (LFS) | open, ~10 MB/s | ESC-50, CREMA-D, MTG-Jamendo metadata, VocalSound metadata |
| zenodo.org | **blocked for large downloads since 2026-10-22** (403 "unusual traffic from your network"); small files still work | FSD50K (done in Colab instead) |
| os.unil.cloud.switch.ch | open, ~2 MB/s | FMA |
| archive.org | open, ~1.6 MB/s | Musopen recordings |
| xeno-canto.org | open; API v3 needs a key (`xeno_canto_api_key` in `.env`) | bird calls |
| pypi.org, files.pythonhosted.org, nltk_data | open | packages |
| huggingface.co API | open, **but the weight/data CDNs are blocked** | — |
| cdn.freesound.org, freesound.org API | **blocked** (403 proxy page) | MTG-Jamendo audio |
| api.jamendo.com, *.storage.jamendo.com | **blocked** | MTG-Jamendo audio |
| www.dropbox.com | **blocked** (TLS reset) | VocalSound audio |
| us.aws.cdn.hf.co, *.xethub.hf.co | **blocked** | HF downloads |
| musopen.org | Cloudflare bot check | use archive.org instead |
