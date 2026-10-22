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

- **Clips** (`data/clips.parquet`, 18,564 items):
  - ESC-50 2,000; FSD50K 2,399; VocalSound 600; xeno-canto 680; CREMA-D 7,442.
  - Music: FMA 4,192; Musopen 116 movements (232 windows); MTG-Jamendo 1,135.
  - FSD50K was embedded in Colab (`colab/fsd50k_embed.ipynb`), because Zenodo blocked the work network. Parity
    with this laptop's pipeline is 1.0000 for both models on 3 reference clips, and the weights' SHA-256 match.
- **Embeddings:** both models, L2-normalised. 15 corrupt FMA files per model are skipped and logged. Gemma took
  about 5.5 h on this CPU; CLAP about 1.5 h.
- **Instrumental filter** (`reports/instrumental_filter.json`, decisions 18–19):
  - Whisper small on a 20 s window. The rule (all transcribed words, threshold T = 1) was chosen on 200 held-out
    calibration tracks: vocal recall 0.61, instrumental recall 0.92.
  - Result: 3,050 FMA tracks instrumental, 927 vocal.
  - MTG: only the 186 piano tracks were checked, and 177 are instrumental.
- **Sanity checks** (`reports/phase3_sanity.json`): **all pass, both models.**
  - ESC-50 classes are tighter than random.
  - ESC-50 probe on the official folds: Gemma 0.83, CLAP 0.97 (chance 0.02).
  - "dog barking" retrieves 17 (Gemma) or 25 (CLAP) dog clips in its top 25, against 0.5 by chance.
  - Music genre probe with artist-grouped splits: 0.41 / 0.49 (chance 0.09).
- **UMAPs** (`reports/phase3_umap.json`): on the joint map only 6% of a word's 10 nearest neighbours are clips.
  Mean cosine: clip↔clip 0.76, text↔text 0.67, text↔clip 0.60. That's the modality gap on clean data.
- **Method fixes** (decisions 17, 21):
  - Probes standardise features.
  - C is chosen per fold by cross-validation inside the training folds. A fixed C underfit Gemma's narrow-cone
    vectors and overfit the small music sets.

## Checkpoint 4: resolution (Q1) and knows vs can name (Q2)

`reports/phase4_resolution.json`: grouped splits, reported on the non-tune folds, 95% cluster-bootstrap CIs.
Gemma's prefix, chosen on the tune fold, is `SearchQuery`.

| Set | Gemma probe | Gemma zero-shot | CLAP probe | CLAP zero-shot | chance |
|---|---|---|---|---|---|
| ESC-50, 50 classes | 0.83 | 0.25 | 0.97 | 0.90 | 0.02 |
| FSD50K, 62 classes | 0.48 | 0.19 | 0.75 | 0.67 | 0.02 |
| VocalSound, 6 vocal sounds | 0.84 | 0.34 | 0.82 | 0.76 | 0.17 |
| xeno-canto, 17 bird species | 0.67 | 0.17 | 0.66 | 0.25 | 0.06 |
| **Giggle vs chuckle** | **0.69** (0.56–0.82) | 0.68 | **0.69** (0.57–0.80) | 0.56 | 0.50 |
| Cuckoo vs 6 owls | 0.73 (0.67–0.80) | 0.22 | 0.74 (0.69–0.80) | 0.27 | 0.14 |
| Owls, 6 species | 0.72 | 0.22 | 0.71 | 0.21 | 0.17 |
| Songbirds, 5 species | 0.63 | 0.20 | 0.69 | 0.38 | 0.20 |
| Scream vs shout | 0.84 | 0.84 | 0.89 | 0.87 | 0.50 |
| Whisper vs sing vs cry | 0.81 | 0.78 | 0.91 | 0.91 | 0.33 |
| Car vs motorcycle vs bus vs truck | 0.44 | 0.32 | 0.59 | 0.52 | 0.25 |

**Anand's question:**
- **Giggle vs chuckle is told apart, but coarsely.** The probe gets 69% (chance 50%), on only 59 test clips. These
  are the only two laughter subtypes in licence-clear labelled data.
- **Relative to chance, bird species are separated at least as well.** Cuckoo vs owls skill is about 0.69, against
  0.39 for laughter. So the space is not concentrated on human expression over wildlife.
- **Words reach the two very differently.** "giggle" finds giggles 90% (Gemma) and 97% (CLAP) of the time, while
  bird-species names barely beat chance.

**Knows vs can name:**
- **CLAP** names what it knows for everyday sounds (ESC-50 gap 7 points).
- **Gemma** knows far more than words reach everywhere (ESC-50 gap 57 points).
- **Both** fail to name bird species.

**Effective vocabulary:** most families clear the 70% pairwise bar entirely (FSD50K human vocal 13/15, machines
22/26 for Gemma). The sibling sets are the sharper view.

## Checkpoint 5: words vs sounds (Q3)

`reports/phase5_words.json`, using pooled ESC-50, FSD50K, VocalSound and xeno-canto clips.

- **Synonym clouds:**
  - Synonyms of one concept retrieve largely *different* clips: top-25 Jaccard 0.06–0.25 for Gemma; only "siren"
    and "applause" agree much.
  - Gemma's synonyms barely cluster in text space (cohesion margin about 0.03–0.16, against 0.11–0.56 for CLAP).
- **Structure match** (Spearman between class-centroid and class-text similarity; all permutation p < .05 except
  VocalSound for Gemma):
  - Gemma ρ: ESC-50 0.24, FSD50K 0.24, xeno-canto 0.32.
  - CLAP ρ: 0.55, 0.58, 0.54, and VocalSound 0.61.
  - Strongest for human vocal sounds; absent for nature.
- **Gap per family:** a clip's own class text beats other class texts by only 0.03–0.04 cosine for Gemma, against
  0.31–0.48 for CLAP.
- **Centring sweep** (α tuned on the tune fold: 0.75 for both models):
  - Gemma overall 0.23 → 0.24, but animals fall 0.22 → 0.16 at full strength.
  - CLAP 0.71 → 0.73.
  - This is category retrieval only; Vibe Search showed the vibe can get worse even when categories improve.

## Checkpoint 6: emotion (Q4)

`reports/phase6_emotion.json`: CREMA-D, 7,442 clips, 91 actors, 12 sentences, 6 emotions.

| Probe accuracy (skill) | how (emotion) | what (sentence) | who (speaker) |
|---|---|---|---|
| Gemma | 0.56 (0.47) | **1.00** (1.00) | 0.06 (0.05) |
| CLAP | 0.58 (0.49) | 0.88 (0.87) | 0.48 (0.47) |
| Prosody features | 0.44 (0.33) | 0.38 (0.33) | 0.16 (0.15) |
| Prosody + MFCC | 0.47 (0.36) | 0.72 (0.70) | 0.49 (0.48) |

- **Zero-shot emotion:** Gemma 0.15 raw / 0.19 centred; CLAP 0.16 / 0.36 (chance 0.17). Raw, both models call
  nearly everything "neutral".
- **Anger:** probe recall 0.74 / 0.75; the words "an angry voice" find 0.02 / 0.00.
- **Verdict:**
  - **Knows but can't name.** The emotion signal is there, a bit above loudness/pitch features, but words don't
    reach it.
  - Gemma encodes what is said almost perfectly and who says it barely at all.

## Checkpoint 7: music without lyrics → genre and region words (Q5, part 1)

`reports/phase7_genre.json`: 3,050 instrumental FMA tracks, artist-grouped folds. The best word form (tune fold) is
the sentence form, raw for Gemma and centred for CLAP.

| | Gemma | CLAP | chance |
|---|---|---|---|
| Genre probe, 11 genres | 0.41 | 0.49 | 0.09 |
| Genre zero-shot (instrumental) | 0.27 | 0.37 | 0.09 |
| Genre zero-shot on *vocal* tracks (lyrics effect) | 0.30 | 0.30 | 0.09 |
| True genre in the word's top 3 | 0.51 | 0.61 | 0.27 |
| Region probe, 11 regions | 0.34 | 0.35 | 0.09 |
| Region zero-shot / top 3 | 0.21 / 0.49 | 0.17 / 0.33 | 0.09 / 0.27 |
| **Musopen canonical classical called "classical"** | **0.99** | **1.00** | — |

- **Precision of the 25 nearest tracks vs base rate (Gemma):** rock 0.84 vs 0.18, folk 0.72 vs 0.16, classical
  0.52 vs 0.12, hip-hop 0.36 vs 0.06, country 0.24 vs 0.03. Easy listening and soul both 0.
- **Lyrics barely help:** zero-shot on vocal tracks is about the same as on instrumental ones, with residual vocals
  in the instrumental set as a caveat.
- **Regions:** words beat chance, but only work as a coarse filter (the true region is in the top 3 about half the
  time).
- **Phase 8 genres**, fixed by the pre-set rule (precision@25 ≥ 2× base rate, Wilson bound above it, Gemma):
  classical, country, electronic, folk, hip-hop, pop, rock. **Contrast:** easy listening.

## Checkpoint 8: what makes a genre (Q5, part 2)

`reports/phase8_whatmakes.json`. Per genre: 16 member tracks plus others, non-ND licences only, 11 perturbations,
both models. Changes are read as a share of each genre's member-vs-rest gap.

- **"Classical", Gemma:**
  - **Keeping only the drums erases 105% of the classical gap.** Classical tracks still ranked "classical" first
    drop from 81% to 31%.
  - **Removing the drums raises the classical score of *any* track.**
  - Shuffling 0.5 s chunks erases 58%, but reversing barely matters (4%). So the model reacts to continuity
    (shuffling adds a cut every 0.5 s) more than to melody direction.
  - Tempo, pitch and dynamics barely move it.
  - **CLAP agrees:** drums-only erases 62%; low-pass and shuffling 37% each.
- **Features:**
  - The classical score falls with percussive share (ρ −0.63), bandwidth (−0.58) and brightness (−0.53).
  - 14 features explain R² 0.52 out of sample.
- **Instrument held fixed:**
  - Among MTG piano tracks, the classical score separates classical piano from other piano with AUC 0.75
    (CI 0.68–0.82). So "classical" is more than "piano".
  - Guitar, strings and synth have too few tracks to test.
- **Descriptors:**
  - Text side: orchestra, piano, melodic, film score.
  - Audio side: piano, flute, saxophone, melodic.
  - Agreement ρ 0.72.
- **False positives** (top 50 non-classical tracks by classical score): folk 14, jazz 8, pop 8.
- **Answer:** to this model, "classical" mostly means **tonal, drum-free, darker and narrower-band sound**: timbre and texture
  more than tempo, key or melodic direction. It holds beyond piano and transfers to canonical recordings.
- **Contrast genre:** easy listening has no member-vs-rest gap at all; the word doesn't single out its own tracks.

## Checkpoint 9: findings page and the bridge

- **Page:** `docs/index.html` + `docs/app.js` + `docs/data.js`, built by `vocab page` from `reports/*.json`.
  - The grep check for numbers typed into the HTML's visible text finds **none** (product names like
    "EmbeddingGemma 2" are excluded).
  - Headless-Edge screenshots were checked: charts render, labels fit, and drill-down works.
  - Attributions, limits and "internal prototype" are in the footer and method section.
  - Audio playback is not included (labels only), so no licence question arises on the page.
- **Bridge** (`reports/vibe_search_bridge.json`):
  - The pre-set rule marks 10 of 12 sister searches as "weak areas", including 6 that scored 3–4 of 4 by ear.
    **Per-class resolution on labelled data doesn't predict podcast listening success**: those searches succeed on
    speaking style and show-level sound.
  - The four failures (heated argument, whispering, rain/nature, music) all sit in weak or unnameable areas.
  - "Heated argument" unpacks to *knows but can't name* for anger (probe 74%, words 2%).
