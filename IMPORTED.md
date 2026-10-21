# Imported from podcast-vibe-search

Source repo: `C:/Users/e416851/Documents/GitHub/podcast-vibe-search` (read-only), commit
`4facfd4d4f31e5b0435d255edb367acfef1615bd`, copied 2026-10-22. The SHA-256 prefix is of the source file at that commit.

| This repo | Source | Source sha256 | Changes |
|---|---|---|---|
| `src/audiovocab/embed.py` | `src/vibe/embed.py` | `f4236c029352` | Kept `Gemma` and `Clap` classes unchanged (float32, `config_kwargs`, fixed 10 s CLAP windows). Model keys renamed `embed_a/embed_b` → `gemma/clap`. Dropped episode/segment functions; added `embed_cached` (per-clip cache keyed by dataset + clip + model + preprocessing). Env var `VIBE_THREADS` → `VOCAB_THREADS`. |
| `src/audiovocab/cache.py` | `src/vibe/cache.py` | `28fa571609ab` | JSON cache unchanged; added `.npy` array get/put. |
| `src/audiovocab/centering.py` | `src/vibe/centering.py` | `d63d1841bbd7` | 77 `GENERIC_PHRASES` unchanged. Means are passed in explicitly (`center(x, mean, strength)`) instead of loaded from one global `centering.npz`, so each analysis can compute its audio mean on a tuning split. |
| `src/audiovocab/llm.py` | `src/vibe/llm.py` | `fc46038817b2` | Unchanged except the token is read via `config.env` (environment or `.env`) and the tag is `:audio-vocab`. |
| `src/audiovocab/config.py` | `src/vibe/config.py` | `0d9d23ed3198` | Same shape; adds `HF_HUB_OFFLINE=1` default and an `env()` helper for `.env`. |
| `src/audiovocab/audio.py` | `src/vibe/audio.py` | `40273be800bf` | Kept `decode` and `probe_duration`; dropped podcast segmenting. |
| `scripts/fetch_bundle.py` | `scripts/fetch_bundle.py` | `394eaa8e28e5` | Template; rewritten for the dataset bundle in Phase 2. |
| `scripts/import_bundle.py` | `scripts/import_bundle.py` | `977ce1cec5a8` | Template; rewritten for the dataset bundle in Phase 2. |
| `CLAUDE.md` | `CLAUDE.md` | `2965cf4663a5` | Unchanged. |
| `ENVIRONMENT.md` | `PROJECT_STATUS.md` §4 | `da7ed527983d` | Section copied; additions marked "(audio-vocab)", plus a proxy reachability table. |
| `requirements.txt` | `.venv` `pip freeze` + `requirements.txt` | `7b971a34abc1` | Same packages pinned to the sister venv's versions; adds `plotly`, `scikit-learn`, `scipy` pins. |

**Not copied (referenced by absolute path):** model weights in `podcast-vibe-search/models/`
(`google--embeddinggemma-2`, `laion--clap-htsat-unfused`, `Systran--faster-whisper-small`). To make the repos
independent, copy that folder and change the three paths in `config.yaml`.

**Read by path, never written:** the sister's `index/` (parity check), `data/` and `data/audio_cache/raw/` (Phase 1
podcast clips), `runs/cache/phase1/` (CLAP parity rows).
