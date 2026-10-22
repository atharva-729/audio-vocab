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

## Checkpoint 3: embed and first look

- **Clips** (`data/clips.parquet`): ESC-50 2,000 · VocalSound 600 · xeno-canto 680 · CREMA-D 7,442 · FMA 4,192 ·
  MTG 1,135 · Musopen 232 windows (116 movements). **FSD50K is not in yet:** Zenodo blocked the work network, and
  `colab/fsd50k_embed.ipynb` does it in Colab.
- **Embeddings:** all 16,281 windows with both models. 30 FMA files (15 per model) were corrupt and are skipped,
  logged in `runs/undecodable.jsonl`. Gemma took about 5.5 h on this CPU; CLAP took about 1.5 h.
- **Sanity checks** (`reports/phase3_sanity.json`): **all pass, both models.**
  - Every ESC-50 class is tighter than random clips.
  - ESC-50 probe on the official folds: Gemma 0.82 (CI 0.80–0.84), CLAP 0.97 (chance 0.02).
  - "dog barking" retrieves 17 dog clips (Gemma) or 25 (CLAP) in its top 25, against 0.5 by chance.
  - The music genre probe check is in Phase 7.
- **UMAPs** (`reports/phase3_umap.json`, drawn on the page): on the joint map only 11% of a word's 10 nearest
  neighbours are clips. Mean cosine (Gemma): clip↔clip 0.76, text↔text 0.67, text↔clip 0.60. That's the same
  modality gap the sister project found, now on clean data.
- **Method fix** (decision 17): probes standardise features on the training folds. Without it, Gemma's narrow-cone
  vectors made the fixed C underfit, and the probe scored below kNN.

## Checkpoint 4: resolution (Q1) and knows vs can name (Q2)

`reports/phase4_resolution.json`, grouped splits, reported on non-tune folds. Gemma's prefix (chosen on the tune
fold) is `SearchQuery` (0.240 vs 0.211 zero-shot on tune).

| Set | Gemma probe | Gemma zero-shot | CLAP probe | CLAP zero-shot | chance |
|---|---|---|---|---|---|
| ESC-50, 50 classes | 0.82 | 0.25 | 0.97 | 0.90 | 0.02 |
| VocalSound (laugh, sigh, cough, throat, sneeze, sniff) | 0.83 | 0.34 | 0.81 | 0.76 | 0.17 |
| xeno-canto, 17 bird species | 0.66 | 0.17 | 0.67 | 0.25 | 0.06 |
| Owls, 6 species | 0.73 | 0.22 | 0.71 | 0.21 | 0.17 |
| Cuckoo vs 6 owls | 0.72 | 0.22 | 0.74 | 0.27 | 0.14 |
| Songbirds, 5 species | 0.63 | 0.20 | 0.69 | 0.38 | 0.20 |
| Crow vs magpie | 0.97 | 0.93 | 0.97 | 0.90 | 0.50 |

- **Resolution:**
  - The space is fine-grained for everyday sounds and human vocal sounds: ESC-50 families reach 0.84–0.98 (Gemma)
    and ≥ 0.97 (CLAP).
  - It is coarser for bird species: 0.63–0.73 within owls or songbirds, in both models.
  - "Cuckoo vs owl" isn't easy: 0.72–0.74 over 7 classes.
- **Effective vocabulary:** every class in every family clears 70% pairwise separability, so the measure is at its
  ceiling here. The differences show in the sibling sets and the bird species.
- **Knows vs can name:**
  - **CLAP** names everyday sounds almost as well as it separates them (ESC-50 0.97 vs 0.90). The gap opens on
    bird species (0.67 vs 0.25).
  - **Gemma** knows far more than words can reach everywhere: ESC-50 0.82 vs 0.25, VocalSound 0.83 vs 0.34.
- **Laughter types** (giggle vs chuckle) are pending FSD50K.

## Checkpoint 5: words vs sounds (Q3)

`reports/phase5_words.json` (pooled ESC-50, VocalSound and xeno-canto clips; FSD50K pending).

- **Synonym clouds:**
  - Synonyms sit much more loosely in Gemma's text space than in CLAP's (cohesion margin 0.03–0.16 vs 0.11–0.56).
  - Synonyms retrieve largely *different* clips: top-25 overlap 0.06–0.48, with only "applause" and "siren" above
    0.3. So the text side is finer than the audio side: the word changes, the retrieved sounds change too, but
    rarely to a better match.
  - "scream" and "whisper" have no matching clips without FSD50K.
- **Structure match:**
  - Audio-centroid and text similarity line up significantly for ESC-50 (Gemma ρ 0.24, CLAP 0.55; p < .001) and
    birds (0.32 / 0.54).
  - The match is strongest for human vocal sounds (ESC-50: 0.75 / 0.74) and absent for nature (−0.19 / 0.10).
- **Gap per family:**
  - Gemma: a clip's own class text beats other class texts by only 0.02–0.045 cosine. Clips sit about 0.86 from
    each other and about 0.70 from text.
  - CLAP: the margin is 0.28–0.52.
  - So Gemma's audio–text alignment for sound events is thin.
- **Centring sweep:**
  - Tuned on the tune fold: α = 0.75 (Gemma), 0.5 (CLAP).
  - On the report folds, centring helps both models overall (Gemma 0.25 → 0.28, CLAP 0.74 → 0.77).
  - For Gemma it *hurts* animals (0.21 → 0.15 at full strength) and nature, and helps human vocal (0.37 → 0.44)
    and machines.

## Checkpoint 6: emotion (Q4)

`reports/phase6_emotion.json`: CREMA-D, 7,442 clips, 91 actors, 12 sentences, 6 emotions.

| Probe (skill above chance) | how (emotion) | what (sentence) | who (speaker) |
|---|---|---|---|
| Gemma | 0.55 (0.46) | **1.00** (1.00) | 0.04 (0.03) |
| CLAP | 0.56 (0.47) | 0.88 (0.87) | 0.48 (0.47) |
| Prosody features | 0.44 (0.33) | 0.38 (0.33) | 0.16 (0.15) |
| Prosody + MFCC | 0.47 (0.36) | 0.72 (0.70) | 0.49 (0.48) |

- **Zero-shot emotion** ("an angry voice"…): Gemma 0.15 raw / 0.19 centred; CLAP 0.16 / 0.36 (chance 0.17). Raw,
  both models call almost everything "neutral".
- **Anger:** the probe finds it best of all emotions (recall 0.72 Gemma, 0.73 CLAP), but the words "an angry voice"
  find 2% (Gemma) or 0% (CLAP).
- **Verdict:**
  - The embeddings carry emotion, a bit better than loudness/pitch features, but **words can't reach it** ("knows
    but can't name").
  - Gemma encodes *what* is said almost perfectly and *who* says it barely at all; CLAP keeps the speaker.
  - The sister's "heated argument" failure fits: the hostility signal exists in the audio, but the query words
    don't find it.
