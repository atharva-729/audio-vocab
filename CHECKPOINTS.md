# Checkpoints

The user asked for the phases to run without stopping, with each checkpoint written here. Numbers come from the
named `reports/*.json` files.

## Checkpoint 0: set up from the sister project

- **Sister repo:** `C:/Users/e416851/Documents/GitHub/podcast-vibe-search`, commit `4facfd4`, read-only. The files
  copied, their sources and hashes are listed in `IMPORTED.md`.
- **Model paths** (`config.yaml`, absolute, not copied): `podcast-vibe-search/models/google--embeddinggemma-2`,
  `.../laion--clap-htsat-unfused`, `.../Systran--faster-whisper-small`. `HF_HUB_OFFLINE=1` is set by default.
- **Parity** (`reports/parity.json`): **pass.** Gemma audio, 10 sister clips: max |diff| 1.6e-7. Gemma text, 5
  transcripts: max |diff| 1.1e-7. CLAP, 5 clips: identical.
- **Time per 20 s clip:** Gemma 4.0–4.2 s, CLAP 0.54–0.58 s (CPU: Intel Core 5 120U, 10 cores / 12 threads).

## Checkpoint 1: read the model

- `MODEL_NOTES.md` has every claim sourced to a file and line or a model-card section.
- **Key facts:**
  - **Gemma truncates audio at 30 s silently.** The model card's "~327 s" doesn't apply in the transformers path.
  - The processor's `audio_seq_length: 280` (11.2 s) is **not** a limit.
  - 25 audio tokens per second.
  - Audio goes into the *same* text transformer as soft tokens, with mean pooling over all positions.
  - Text queries should use the `SearchQuery` prefix; audio takes no prefix.
- **Vibe Search cross-check** (`reports/vibe_search_crosscheck.md`): **no problems.** 20 s clips aren't truncated,
  and queries, transcripts and centring phrases all use the right formats.
- **Behaviour tests** (`reports/phase1_behaviour.json`; tables at the end of `MODEL_NOTES.md`):
  - **Length:** a 10 s clip has cosine 0.91 to the same clip at 30 s, and 20 s gives 0.97. Content drift (the
    next 20 s) gives 0.81, so content moves the vector more than length does.
  - **Loudness:** Gemma is near-invariant (≥ 0.98 from −12 to +6 dB). CLAP is less so (0.92 at −12 dB).
  - **Silence/noise:** not hubs in either model. Gemma's space is a narrow cone (clip↔clip ≈ 0.71, silence↔speech
    ≈ 0.67).
  - **Wording:** the `SearchQuery` prefix moves a text vector to cosine 0.89 with the bare text. Prefixes push text
    away from audio in raw cosine, so the prefix is chosen on the tune fold in Phase 4.
- **Cost per clip** (measured while downloads ran, so upper bounds): Gemma 0.7 s (1 s clip) to 3.9 s (10 s clip);
  CLAP 0.24–0.37 s.

## Checkpoint 2: datasets, licences, download plan

- **Licences** (`data/LICENCES.md`, each verified from the source text):
  - ESC-50: CC BY-NC 3.0.
  - FSD50K: per clip (CC0 / BY / BY-NC / Sampling+).
  - VocalSound: CC BY-SA 4.0.
  - xeno-canto: per recording, ND excluded.
  - CREMA-D: ODbL / DbCL.
  - FMA: metadata CC BY 4.0, audio per track.
  - MTG-Jamendo: research-only.
  - Musopen: Public Domain Mark.
  - Excluded: GTZAN (licence unclear); AudioSet and MusicCaps (YouTube). Not used: RAVDESS (backup only) and Song
    Describer (optional).
- **Reachability** (`ENVIRONMENT.md`): VocalSound (Dropbox) and MTG-Jamendo audio (Freesound CDN) were blocked
  and came via the bundle; it arrived and all checksums verified. **Zenodo later blocked this network** ("unusual
  traffic") partway through the FSD50K download (see Phase 3).
- **Selection** (`reports/phase2_plan.json`, `data/selection/*.parquet`):
  - ESC-50: all 2,000 clips, 50 classes.
  - FSD50K eval: 2,399 clips, 62 classes, ≤ 40 per class. This includes **Giggle (40) and Chuckle/chortle (30)**,
    the only laughter subtypes in licence-clear data.
  - VocalSound: 600 clips, 6 classes, one per speaker.
  - xeno-canto: 17 species × 40 recordings (cuckoo, 6 owls, doves, hoopoe, songbirds, corvids).
  - CREMA-D: all 7,442 clips.
  - FMA: 4,198 tracks (genre and region pools plus a 200-track Whisper calibration set).
  - Musopen: 116 movements from 37 works.
  - MTG-Jamendo: shards 53 and 86, 1,135 tracks.
- **Taxonomy:** `data/taxonomy.yaml`. 11 genre words and 11 region words; dropped labels are listed with reasons.
- **Decisions:** `reports/decisions_log.md`.
