# Audio Vocab: full context for a follow-on project

**Audience:** Claude (or any engineer) starting a project that builds on these findings. Read this end to end.
**What it is:** a self-contained briefing covering what was measured, how, every headline number, what to trust,
what not to trust, and how to reuse the artefacts. Every number below is generated from `reports/*.json` by
`scripts/make_context.py`, so regenerate this file after any rerun. **Scope rule:** every finding holds *in these
datasets, with these models*, not for all audio.

Repo: https://github.com/atharva-729/audio-vocab · sister project: `podcast-vibe-search` (Vibe Search) ·
companion docs: `README.md` (original plan), `PROJECT_STATUS.md` (handoff), `CHECKPOINTS.md` (per-phase),
`MODEL_NOTES.md` (model internals, sourced), `reports/decisions_log.md` (21 decisions), `data/LICENCES.md`,
`ENVIRONMENT.md`, results page `docs/index.html`.

---

## 1. The project in one page

**Question:** what can an audio–text embedding actually hear, and which of it can words reach?

**Models (frozen, nothing fine-tuned):**
- **EmbeddingGemma 2** (Google DeepMind, 768-d). This is the model the sister project's podcast "vibe search" was
  built on.
- **LAION-CLAP `htsat-unfused`** (512-d), as a second model. A finding that holds for both is about audio
  embeddings; one that holds for one only is about that model.

**Data:** labelled public datasets only, no hand-labelling:
- sound events: ESC-50, FSD50K;
- human vocal sounds: VocalSound;
- bird species: xeno-canto;
- acted emotion: CREMA-D;
- music: FMA, Musopen, MTG-Jamendo.

**The five questions and the short answers:**
1. **Resolution:** fine for everyday sounds, coarse for bird species. Giggle vs chuckle is told apart only coarsely,
   and the space is not tuned to human expression over wildlife.
2. **Knows vs can name:** CLAP names most of what it separates. **Gemma separates sound events well but its words
   barely reach them.** Both fail to name bird species.
3. **Words vs sounds:**
   - Text and audio similarity structures correlate moderately (stronger in CLAP).
   - Synonyms retrieve different clips.
   - Gemma's audio–text margin for sound events is tiny.
   - Centring helps category retrieval slightly, but hurts animals in Gemma.
4. **Emotion:** encoded in the audio (probe), not reachable by words (zero-shot ≈ chance, everything comes out
   "neutral"). Gemma encodes *what is said* almost perfectly and *who says it* barely at all.
5. **Music:**
   - Genre words partly work on instrumental music: rock, folk and classical well; easy listening, soul and blues
     not.
   - Region words work only as a coarse filter.
   - **"Classical" = tonal, drum-free, darker/narrow-band texture.** It is more than "piano", and it transfers to
     canonical recordings.
   - Lyrics barely help.

**Bridge to the sister project:** per-class resolution on labelled data does *not* predict which podcast vibe
searches sounded right. Vibe search succeeds on speaking style and show-level sound, which labelled sound datasets
don't measure.

---

## 2. The models: facts that matter for any reuse

(Full sourced notes: `MODEL_NOTES.md`.)

### EmbeddingGemma 2
- **Input:** 16 kHz mono → 128-bin log-mel (0–8 kHz, 20 ms frames, 10 ms hop) → 12-layer conformer (local, causal
  attention, about 1 s per layer) → **25 audio tokens per second** (measured: 1 s → 25,
  10 s → 250, 20 s → 500).
- **Hard limit: audio over 30 s is silently truncated** (30, 40 and 60 s all → 750 tokens;
  the embeddings are bit-identical). The model card's "~327 s" does not apply via transformers/sentence-transformers.
  `audio_seq_length: 280` in `processor_config.json` is *not* a limit (it's a vLLM helper).
- **Audio is fed into the same 24-layer text transformer as soft tokens.** It's one shared backbone, not two towers.
  The vector is the mean over all positions (including the special tokens), projected 512→768, then L2-normalised.
- **Text prefixes:** queries `task: search result | query: …` (prompt name `SearchQuery`); documents
  `title: none | text: …`. **Audio takes no prefix.** The prefix is mean-pooled into the vector, so it shifts every
  query vector. Chosen here on a tuning fold: `SearchQuery` (zero-shot
  21% vs 20%
  with no prefix).
- **Precision:** float32 on CPU. bf16 is about 12× slower on this CPU; **never fp16** (NaNs). Load with
  `config_kwargs={"vision_config": None}` (add `"audio_config": None` for text only).
- **Geometry:** a **narrow cone**. Mean cosine clip↔clip is about 0.76, text↔text
  0.67, text↔clip 0.60. Absolute cosines are not
  comparable across models; use rankings, contrasts or standardised features. **Linear probes need feature
  standardisation** (per-dimension std ≈ 0.015), or a fixed regularisation silently underfits.
- **Cost on this laptop** (Intel Core 5 120U, 15 W): 4.19 s per 20 s clip (clean);
  about 2–4 s per 10 s clip in batch runs. CLAP is about 7–10× cheaper (0.58 s per 20 s clip).

### CLAP (laion/clap-htsat-unfused)
- 48 kHz, 64-bin log-mel; **maximum 10 s**. The processor's default for longer audio is a **random crop**; our
  wrapper averages fixed 10 s windows instead, so it is deterministic.
- Two towers (HTS-AT audio, RoBERTa text) trained contrastively. No prompts or prefixes. 512-d.
- Its space is spread out (clip↔clip cosine about 0.56 on
  podcasts, against about 0.71 for Gemma).
- It may have seen ESC-50-like data in training, so ESC-50 zero-shot numbers may be inflated.

### Behaviour (Phase 1, 30 podcast clips)
| Test | Gemma | CLAP |
|---|---|---|
| 10 s window vs the same clip at 30 s (cosine) | 0.91 | 0.96 |
| 20 s vs 30 s | 0.97 | 0.99 |
| next 20 s of the same show (content drift) | 0.81 | 0.91 |
| −12 dB vs 0 dB | 1.00 | 0.92 |
| silence ↔ clips (mean cos) | 0.67 | 0.07 |
| clips whose nearest neighbour is silence | 0 | 0 |
| SearchQuery prefix vs bare text (same words, cos) | 0.89 | — |

**Take-aways:**
- **Content moves the vector more than length does**, so 10 s windows are fine.
- **Gemma is close to loudness-invariant.**
- **Silence is not a hub.**
- **Prefixes shift text vectors a lot.** Choose them on held-out data.

---

## 3. Data, windows, splits, filters

| Dataset | Items | What | Window embedded | Group (for splits) | Licence |
|---|---|---|---|---|---|
| ESC-50 | 2000 | 50 everyday classes × 40 | whole 5 s | source Freesound file (official 5 folds) | CC BY-NC 3.0 |
| FSD50K (eval) | 2399 | 62 classes, ≤ 40 each; giggle, chuckle, scream, shout, whisper… | first min(10 s, length) | Freesound uploader | per clip CC0/BY/BY-NC |
| VocalSound | 600 | laughter, sigh, cough, throat-clear, sneeze, sniff; 100 each, one per speaker | first ≤ 10 s | speaker | CC BY-SA 4.0 |
| xeno-canto | 680 | 17 species × 40 (cuckoo, 6 owls, 2 doves, hoopoe, 5 songbirds, 2 corvids) | loudest 5 s | recordist (≤ 2 rec each) | per recording, ND excluded |
| CREMA-D | 7442 | 91 actors × 12 sentences × 6 emotions | whole utterance | actor | ODbL / DbCL |
| FMA (fma_large) | 4192 | 30 s clips; 11 genres, 11 regional subgenres | 10–20 s | artist | per track CC |
| Musopen | 116 | 116 movements, 37 canonical works | 2 × 10 s (30%, 60%) | work | Public Domain Mark |
| MTG-Jamendo | 1135 | shards 53 + 86; genre + instrument tags | 10 s at 40% | artist | research-only |

- **How the data was acquired** (the work network blocks several hosts):
  - VocalSound and MTG came via a personal-laptop bundle.
  - FSD50K was embedded in **Colab** (`colab/fsd50k_embed.ipynb`) after Zenodo blocked the network; the result was
    verified identical to the local pipeline (parity 1.0000).
  - FMA tracks were pulled individually out of the 100 GB `fma_large.zip` with HTTP range requests.
- **FSD50K labels propagate up the AudioSet tree.** A clip is assigned to a class only if it has no rival label in
  the same sibling group; parent labels (Laughter, Water…) are not used as classes.
- **Folds:** ESC-50 uses its official folds; everything else uses 5 folds by a stable hash of the group id. **Fold 1
  is the tuning fold** (prefix, centring strength, word form); all reported numbers use folds 2–5. Probes and kNN
  never mix a group across train and test.
- **Instrumental filter (music):** Whisper small on a 20 s window. The rule was calibrated on
  100 + 100 held-out FMA tracks:
  - chosen: `words_all` ≥ 1;
  - vocal recall 61%, instrumental recall
    92%;
  - **residual vocals remain.** FMA's own "Instrumental" tag is sparse (0–20 tracks per region), and MTG's missing
    voice tag is not evidence of an instrumental track.
- **Excluded:** GTZAN (unclear licence), AudioSet and MusicCaps (YouTube), RAVDESS (backup only), Song Describer
  (optional).

---

## 4. Protocol and metric definitions

- **Linear probe ("what the audio contains"):**
  - logistic regression on **standardised** features;
  - **C chosen per outer fold from {0.001, 0.01, 0.1, 1} by leave-one-fold-out inside the training folds** (no
    leakage);
  - grouped cross-validation over all 5 folds, reported on folds 2–5.
- **kNN:** k = 10, cosine; neighbours from the clip's own group are excluded.
- **Zero-shot ("what a word finds"):**
  - each clip gets the label of the nearest class text vector;
  - a class text vector is the mean of the class name and 2 short descriptions (in `data/taxonomy.yaml`),
    re-normalised.
- **Centred zero-shot:** subtract the audio mean (tuning fold) from clips and the mean of 77 generic phrases from
  texts, then re-normalise. α = strength.
- **Probe − zero-shot gap** = "knows but can't name".
- **Skill** = (accuracy − chance) / (1 − chance). Use it to compare problems with different numbers of classes.
- **Effective vocabulary** = the largest set of classes in which every pair is separated with ≥ 70% binary-probe
  accuracy (a maximum clique). **It saturates in most families here**; use the sibling sets instead.
- **Uncertainty:** 95% cluster-bootstrap intervals over groups (1,000 resamples). Precision@25 uses Wilson
  intervals.
- **Genre score** (Phase 8) = cos(clip, genre word) − mean cos(clip, other genre words). Perturbation effects are
  reported as a **share of the genre's member-vs-rest gap**.

---

## 5. Findings, question by question

### Q1 + Q2: resolution, and knows vs can name

| Dataset | classes | clips | chance | Gemma probe | Gemma kNN | Gemma ZS | Gemma ZS centred | CLAP probe | CLAP kNN | CLAP ZS | CLAP ZS centred |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esc50 | 50 | 2000 | 2% | 83% (80%–85%) | 77% | 25% | 28% | 97% (96%–98%) | 97% | 90% | 92% |
| fsd50k | 62 | 2399 | 2% | 48% (45%–51%) | 43% | 19% | 20% | 75% (73%–77%) | 74% | 67% | 68% |
| vocalsound | 6 | 600 | 17% | 84% (81%–88%) | 80% | 34% | 43% | 82% (79%–85%) | 80% | 76% | 78% |
| xenocanto | 17 | 680 | 6% | 67% (62%–71%) | 60% | 17% | 12% | 66% (62%–70%) | 60% | 25% | 29% |


**Within-family problems** (only the classes of one family, in one dataset):

| model | dataset | family | classes | chance | probe | zero-shot | effective vocab |
|---|---|---|---|---|---|---|---|
| gemma | esc50 | animal | 12 | 8% | 88% (84%–91%) | 35% | 12/12 |
| gemma | esc50 | human_activity | 4 | 25% | 97% (93%–99%) | 66% | 4/4 |
| gemma | esc50 | human_vocal | 6 | 17% | 94% (90%–97%) | 49% | 6/6 |
| gemma | esc50 | machine_domestic | 22 | 5% | 85% (82%–88%) | 29% | 22/22 |
| gemma | esc50 | nature | 6 | 17% | 94% (90%–97%) | 24% | 6/6 |
| gemma | fsd50k | animal | 9 | 11% | 63% (56%–70%) | 40% | 8/9 |
| gemma | fsd50k | human_activity | 3 | 33% | 94% (88%–98%) | 61% | 3/3 |
| gemma | fsd50k | human_vocal | 15 | 7% | 57% (52%–62%) | 30% | 13/15 |
| gemma | fsd50k | machine_domestic | 26 | 4% | 53% (49%–57%) | 21% | 22/26 |
| gemma | fsd50k | nature | 8 | 12% | 64% (58%–71%) | 20% | 8/8 |
| gemma | vocalsound | human_vocal | 6 | 17% | 84% (81%–88%) | 34% | 6/6 |
| gemma | xenocanto | animal | 17 | 6% | 67% (62%–71%) | 17% | 17/17 |
| clap | esc50 | animal | 12 | 8% | 99% (97%–100%) | 92% | 12/12 |
| clap | esc50 | human_activity | 4 | 25% | 100% (100%–100%) | 98% | 4/4 |
| clap | esc50 | human_vocal | 6 | 17% | 98% (95%–100%) | 96% | 6/6 |
| clap | esc50 | machine_domestic | 22 | 5% | 97% (96%–98%) | 91% | 22/22 |
| clap | esc50 | nature | 6 | 17% | 98% (96%–100%) | 97% | 6/6 |
| clap | fsd50k | animal | 9 | 11% | 86% (82%–90%) | 79% | 9/9 |
| clap | fsd50k | human_activity | 3 | 33% | 98% (94%–100%) | 97% | 3/3 |
| clap | fsd50k | human_vocal | 15 | 7% | 71% (66%–75%) | 61% | 14/15 |
| clap | fsd50k | machine_domestic | 26 | 4% | 81% (78%–84%) | 74% | 25/26 |
| clap | fsd50k | nature | 8 | 12% | 85% (80%–90%) | 68% | 8/8 |
| clap | vocalsound | human_vocal | 6 | 17% | 82% (79%–85%) | 76% | 6/6 |
| clap | xenocanto | animal | 17 | 6% | 66% (62%–70%) | 25% | 17/17 |


**Sibling sets** (fine distinctions; sorted by Gemma skill):

| set | dataset | classes | chance | Gemma probe | Gemma skill | Gemma ZS | CLAP probe | CLAP skill | CLAP ZS | test clips |
|---|---|---|---|---|---|---|---|---|---|---|
| esc_door | esc50 | door wood creaks, door wood knock | 50% | 100% (100%–100%) | 1.00 | 77% | 100% (100%–100%) | 1.00 | 97% | 64 |
| esc_tools | esc50 | chainsaw, hand saw | 50% | 100% (100%–100%) | 1.00 | 52% | 100% (100%–100%) | 1.00 | 100% | 64 |
| esc_weather | esc50 | rain, thunderstorm, wind | 33% | 99% (97%–100%) | 0.98 | 39% | 99% (97%–100%) | 0.98 | 98% | 96 |
| esc_clocks | esc50 | church bells, clock alarm, clock tick | 33% | 98% (95%–100%) | 0.97 | 84% | 100% (100%–100%) | 1.00 | 95% | 96 |
| esc_pets | esc50 | cat, dog | 50% | 98% (95%–100%) | 0.97 | 69% | 100% (100%–100%) | 1.00 | 100% | 64 |
| cat_sounds | fsd50k | Hiss, Meow, Purr | 33% | 96% (92%–100%) | 0.94 | 79% | 100% (100%–100%) | 1.00 | 98% | 100 |
| corvids | xenocanto | carrion crow, eurasian magpie | 50% | 97% (91%–100%) | 0.93 | 93% | 93% (86%–98%) | 0.86 | 90% | 59 |
| esc_water | esc50 | pouring water, rain, sea waves, toilet flush, water drops | 20% | 94% (91%–98%) | 0.93 | 48% | 100% (100%–100%) | 1.00 | 91% | 160 |
| esc_birds | esc50 | chirping birds, crow, hen, rooster | 25% | 95% (90%–98%) | 0.93 | 51% | 99% (98%–100%) | 0.99 | 89% | 128 |
| esc_body | esc50 | breathing, coughing, crying baby, laughing, sneezing, snoring | 17% | 94% (90%–97%) | 0.92 | 49% | 98% (95%–100%) | 0.98 | 96% | 192 |
| esc_appliances | esc50 | vacuum cleaner, washing machine | 50% | 95% (90%–100%) | 0.91 | 50% | 98% (94%–100%) | 0.97 | 91% | 64 |
| esc_computer | esc50 | keyboard typing, mouse click | 50% | 94% (87%–98%) | 0.88 | 50% | 98% (94%–100%) | 0.97 | 91% | 64 |
| esc_farm | esc50 | cow, hen, pig, rooster, sheep | 20% | 89% (83%–95%) | 0.87 | 44% | 98% (95%–99%) | 0.97 | 90% | 160 |
| vocal_nonspeech | vocalsound | cough, laughter, sigh, sneeze, sniff, throatclearing | 17% | 84% (81%–88%) | 0.81 | 34% | 82% (79%–85%) | 0.79 | 76% | 482 |
| dog_sounds | fsd50k | Bark, Growling | 50% | 87% (77%–96%) | 0.74 | 65% | 96% (91%–100%) | 0.93 | 94% | 54 |
| esc_vehicles | esc50 | airplane, car horn, engine, helicopter, siren, train | 17% | 78% (71%–85%) | 0.73 | 42% | 95% (92%–98%) | 0.94 | 88% | 192 |
| voice_kinds | fsd50k | Crying and sobbing, Singing, Whispering | 33% | 81% (74%–89%) | 0.72 | 78% | 91% (84%–97%) | 0.86 | 91% | 97 |
| fsd_body | fsd50k | Breathing, Burping and eructation, Cough, Sigh, Sneeze | 20% | 77% (69%–85%) | 0.71 | 37% | 83% (76%–89%) | 0.78 | 76% | 161 |
| cuckoo_vs_owls | xenocanto | barn owl, common cuckoo, eurasian eagle-owl, eurasian scops owl, little owl, long-eared owl, tawny owl | 14% | 73% (67%–80%) | 0.69 | 22% | 74% (69%–80%) | 0.70 | 27% | 215 |
| loud_voice | fsd50k | Screaming, Shout | 50% | 84% (74%–93%) | 0.69 | 84% | 89% (80%–96%) | 0.77 | 87% | 70 |
| kitchen | fsd50k | Boiling, Cutlery and silverware, Dishes and pots and pans, Frying (food), Microwave oven | 20% | 74% (66%–81%) | 0.68 | 34% | 85% (78%–91%) | 0.81 | 87% | 154 |
| owls | xenocanto | barn owl, eurasian eagle-owl, eurasian scops owl, little owl, long-eared owl, tawny owl | 17% | 72% (66%–78%) | 0.67 | 22% | 71% (65%–78%) | 0.65 | 21% | 183 |
| hooting_calls | xenocanto | common cuckoo, common wood pigeon, eurasian collared dove, eurasian hoopoe, long-eared owl, tawny owl | 17% | 70% (63%–78%) | 0.64 | 30% | 69% (62%–75%) | 0.62 | 35% | 179 |
| signals | fsd50k | Bicycle bell, Doorbell, Ringtone, Siren, Vehicle horn and car horn and honking | 20% | 69% (61%–76%) | 0.61 | 47% | 85% (79%–91%) | 0.82 | 78% | 157 |
| tools | fsd50k | Drill, Hammer, Sawing | 33% | 74% (62%–84%) | 0.61 | 48% | 93% (86%–97%) | 0.89 | 96% | 95 |
| fsd_weather | fsd50k | Rain, Thunder, Wind | 33% | 71% (60%–82%) | 0.57 | 29% | 96% (92%–100%) | 0.94 | 92% | 80 |
| fsd_water | fsd50k | Drip, Stream, Trickle and dribble, Waves and surf | 25% | 66% (58%–75%) | 0.55 | 34% | 87% (81%–92%) | 0.83 | 70% | 125 |
| songbirds | xenocanto | common blackbird, common chaffinch, common nightingale, eurasian wren, great tit | 20% | 63% (56%–71%) | 0.54 | 20% | 69% (62%–74%) | 0.61 | 38% | 160 |
| speakers | fsd50k | Child speech and kid speaking, Female speech and woman speaking, Male speech and man speaking | 33% | 59% (50%–69%) | 0.39 | 51% | 71% (60%–80%) | 0.56 | 75% | 106 |
| laughter_types | fsd50k | Chuckle and chortle, Giggle | 50% | 69% (56%–82%) | 0.39 | 68% | 69% (57%–80%) | 0.39 | 56% | 59 |
| fsd_birds | fsd50k | Chirp and tweet, Gull and seagull | 50% | 69% (54%–82%) | 0.37 | 59% | 91% (84%–98%) | 0.81 | 76% | 54 |
| road_vehicles | fsd50k | Bus, Car, Motorcycle, Truck | 25% | 44% (36%–54%) | 0.26 | 32% | 59% (49%–68%) | 0.45 | 52% | 124 |


**Interpretation:**
- **Everyday sounds are finely resolved by both models**: ESC-50 within-family probes are about 0.84–1.0.
- **FSD50K is much harder** (62 classes, noisier and multi-label sources); Gemma struggles notably there. The
  weakest sets are road vehicles and speaker types.
- **Bird species are the coarse spot in both models**: owls about 0.72; cuckoo vs owls
  73% (67%–80%) (Gemma), against a chance of 14%.
- **Laughter types:** giggle vs chuckle 69% (56%–82%) (Gemma),
  69% (57%–80%) (CLAP), on 59 test clips.
  In skill terms that's lower than cuckoo vs owls. **The space is not concentrated on human expression over
  wildlife.** It has a modest laughter-type distinction and a decent species distinction.
- **Naming differs:** "giggle" finds giggles in
  90% (Gemma) /
  97% (CLAP) of cases; "chuckle" finds chuckles in
  43% /
  11%. Bird-species names barely
  beat chance.
- **Gemma's zero-shot is very uneven.** Many ESC-50 classes it separates at 90–100% get 0% zero-shot (see below).
  **Don't use Gemma text queries for sound-event retrieval without checking**; CLAP is far better there.

- **gemma: biggest knows-but-can't-name classes** (probe → zero-shot recall): thunderstorm (esc50) 100%→3%; toilet_flush (esc50) 94%→0%; Crash_cymbal (fsd50k) 91%→0%; brushing_teeth (esc50) 91%→0%; rooster (esc50) 91%→0%; sea_waves (esc50) 91%→0%; vacuum_cleaner (esc50) 91%→0%; chainsaw (esc50) 88%→0%; cow (esc50) 88%→0%; glass_breaking (esc50) 88%→3%
- **gemma: hardest classes for the probe** (confused with): Car (fsd50k) 6% ↔ Rain; Chuckle_and_chortle (fsd50k) 11% ↔ Giggle; Gull_and_seagull (fsd50k) 16% ↔ Thunder; Sawing (fsd50k) 18% ↔ Drill; Wind (fsd50k) 19% ↔ Rain; Truck (fsd50k) 19% ↔ Thunder; Male_speech_and_man_speaking (fsd50k) 21% ↔ Burping_and_eructation; Train (fsd50k) 22% ↔ Thunder
- **clap: biggest knows-but-can't-name classes** (probe → zero-shot recall): barn owl (xenocanto) 86%→10%; eurasian eagle-owl (xenocanto) 70%→0%; eurasian wren (xenocanto) 83%→14%; eurasian scops owl (xenocanto) 84%→19%; eurasian magpie (xenocanto) 67%→4%; Trickle_and_dribble (fsd50k) 69%→7%; little owl (xenocanto) 61%→0%; water_drops (esc50) 97%→41%; eurasian hoopoe (xenocanto) 55%→0%; eurasian collared dove (xenocanto) 57%→3%
- **clap: hardest classes for the probe** (confused with): Chuckle_and_chortle (fsd50k) 29% ↔ Giggle; Car (fsd50k) 34% ↔ Motorcycle; Truck (fsd50k) 35% ↔ Vehicle_horn_and_car_horn_and_honking; long-eared owl (xenocanto) 36% ↔ tawny owl; great tit (xenocanto) 44% ↔ common chaffinch; Male_speech_and_man_speaking (fsd50k) 47% ↔ Singing; Chirp_and_tweet (fsd50k) 52% ↔ Gull_and_seagull; Hiss (fsd50k) 53% ↔ Boiling

---

### Q3: words vs sounds

**gemma: synonym clouds** (raw; pooled sound clips):

| concept | text cohesion (within − across) | top-25 overlap (Jaccard) | precision@25 (mean over synonyms) | base rate | best / worst synonym |
|---|---|---|---|---|---|
| laugh | 0.08 | 0.11 | 46% | 4% | laugh 96% / chortle 0% |
| cry | 0.10 | 0.13 | 7% | 1% | crying 24% / wailing 0% |
| scream | 0.08 | 0.12 | 26% | 1% | yell 52% / bellow 0% |
| whisper | 0.05 | 0.06 | 3% | 1% | soft speech 8% / murmur 0% |
| cough | 0.03 | 0.07 | 4% | 5% | coughing 12% / cough 0% |
| sneeze | 0.08 | 0.11 | 43% | 5% | achoo 76% / snort 8% |
| rain | 0.13 | 0.19 | 18% | 1% | rain 24% / pouring rain 8% |
| wind | 0.07 | 0.15 | 13% | 1% | windstorm 28% / gale 0% |
| water | 0.07 | 0.11 | 52% | 5% | dripping 84% / stream 16% |
| dog | 0.08 | 0.11 | 32% | 2% | dog 72% / yapping 4% |
| cat | 0.09 | 0.12 | 23% | 3% | meow 68% / tomcat 0% |
| bird | 0.07 | 0.09 | 50% | 5% | birdsong 96% / trill 0% |
| owl | 0.14 | 0.13 | 16% | 4% | owl call 48% / owl 0% |
| engine | 0.11 | 0.20 | 43% | 4% | idling engine 76% / vroom 4% |
| applause | 0.14 | 0.23 | 65% | 1% | round of applause 96% / acclaim 8% |
| siren | 0.16 | 0.25 | 42% | 1% | emergency siren 64% / alarm 0% |

**gemma: structure match** (Spearman ρ, class-centroid similarity vs class-text similarity; permutation p):

| dataset | all classes | by family |
|---|---|---|
| esc50 | ρ 0.24 (p 0.0005) | animal 0.42 (p 0.009); human_activity 0.60 (p 0.1724); human_vocal 0.75 (p 0.001); machine_domestic 0.26 (p 0.0025); nature -0.19 (p 0.7211) |
| fsd50k | ρ 0.24 (p 0.0005) | animal 0.12 (p 0.2524); human_vocal 0.59 (p 0.0005); machine_domestic 0.31 (p 0.0005); nature -0.05 (p 0.5967) |
| vocalsound | ρ -0.04 (p 0.5177) | human_vocal -0.04 (p 0.5047) |
| xenocanto | ρ 0.32 (p 0.004) | animal 0.32 (p 0.0015) |

**gemma: gap per family** (cosine):

| family | clip↔own class text | clip↔other class texts | margin | clip↔clip within family |
|---|---|---|---|---|
| animal | 0.70 | 0.66 | 0.04 | 0.86 |
| human_activity | 0.68 | 0.65 | 0.03 | 0.85 |
| human_vocal | 0.70 | 0.66 | 0.04 | 0.82 |
| machine_domestic | 0.69 | 0.65 | 0.03 | 0.84 |
| nature | 0.69 | 0.66 | 0.03 | 0.88 |
| music | 0.64 | 0.65 | -0.01 | 0.90 |

**gemma: centring strength** (α chosen on the tuning fold = 0.75; zero-shot accuracy on report folds):

| family | α=0.0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---|---|---|---|---|
| animal | 22% | 20% | 19% | 17% | 16% |
| human_activity | 18% | 19% | 21% | 25% | 27% |
| human_vocal | 32% | 35% | 36% | 36% | 37% |
| machine_domestic | 20% | 22% | 22% | 23% | 25% |
| music | 0% | 0% | 0% | 9% | 15% |
| nature | 12% | 13% | 14% | 13% | 13% |
| all | 23% | 24% | 24% | 24% | 24% |

**clap: synonym clouds** (raw; pooled sound clips):

| concept | text cohesion (within − across) | top-25 overlap (Jaccard) | precision@25 (mean over synonyms) | base rate | best / worst synonym |
|---|---|---|---|---|---|
| laugh | 0.32 | 0.08 | 54% | 4% | laugh 92% / snicker 0% |
| cry | 0.15 | 0.02 | 12% | 1% | bawling 68% / sobbing 0% |
| scream | 0.34 | 0.19 | 64% | 1% | yell 96% / howl 0% |
| whisper | 0.16 | 0.07 | 20% | 1% | whisper 72% / mutter 0% |
| cough | 0.11 | 0.01 | 32% | 5% | cough 100% / splutter 0% |
| sneeze | 0.21 | 0.04 | 52% | 5% | sneeze 100% / achoo 0% |
| rain | 0.38 | 0.11 | 50% | 1% | rain 96% / sprinkle 0% |
| wind | 0.17 | 0.12 | 41% | 1% | wind 84% / gust 0% |
| water | 0.27 | 0.09 | 72% | 5% | waves 100% / trickle 12% |
| dog | 0.14 | 0.07 | 50% | 2% | dog 100% / canine 0% |
| cat | 0.20 | 0.08 | 67% | 3% | cat 100% / caterwaul 0% |
| bird | 0.22 | 0.03 | 44% | 5% | songbird 92% / trill 0% |
| owl | 0.28 | 0.19 | 66% | 4% | night owl 96% / hooting 0% |
| engine | 0.36 | 0.15 | 54% | 4% | rumbling motor 80% / vroom 0% |
| applause | 0.39 | 0.19 | 66% | 1% | applause 96% / ovation 0% |
| siren | 0.56 | 0.30 | 90% | 1% | siren 96% / klaxon 60% |

**clap: structure match** (Spearman ρ, class-centroid similarity vs class-text similarity; permutation p):

| dataset | all classes | by family |
|---|---|---|
| esc50 | ρ 0.55 (p 0.0005) | animal 0.72 (p 0.0005); human_activity 0.60 (p 0.1719); human_vocal 0.74 (p 0.002); machine_domestic 0.63 (p 0.0005); nature 0.10 (p 0.3798) |
| fsd50k | ρ 0.58 (p 0.0005) | animal 0.45 (p 0.0045); human_vocal 0.75 (p 0.0005); machine_domestic 0.63 (p 0.0005); nature 0.45 (p 0.0195) |
| vocalsound | ρ 0.61 (p 0.019) | human_vocal 0.61 (p 0.0225) |
| xenocanto | ρ 0.54 (p 0.0005) | animal 0.54 (p 0.0005) |

**clap: gap per family** (cosine):

| family | clip↔own class text | clip↔other class texts | margin | clip↔clip within family |
|---|---|---|---|---|
| animal | 0.45 | 0.14 | 0.31 | 0.39 |
| human_activity | 0.49 | 0.02 | 0.47 | 0.26 |
| human_vocal | 0.43 | 0.12 | 0.31 | 0.34 |
| machine_domestic | 0.46 | -0.00 | 0.47 | 0.13 |
| nature | 0.42 | 0.03 | 0.39 | 0.23 |
| music | 0.43 | -0.06 | 0.48 | 0.51 |

**clap: centring strength** (α chosen on the tuning fold = 0.75; zero-shot accuracy on report folds):

| family | α=0.0 | α=0.25 | α=0.5 | α=0.75 | α=1.0 |
|---|---|---|---|---|---|
| animal | 57% | 59% | 61% | 61% | 60% |
| human_activity | 89% | 88% | 89% | 91% | 92% |
| human_vocal | 71% | 72% | 72% | 73% | 74% |
| machine_domestic | 81% | 81% | 82% | 82% | 81% |
| music | 88% | 88% | 88% | 88% | 88% |
| nature | 64% | 65% | 66% | 67% | 66% |
| all | 71% | 72% | 73% | 73% | 73% |

**Interpretation:**
- **Synonyms mostly retrieve different clips.** Only "siren" and "applause" synonyms agree much. Changing the word
  moves the results, usually not toward a better match. The text side is finer-grained than what the audio side
  responds to.
- **The text and audio similarity structures correlate** (CLAP ρ about 0.55–0.6, Gemma about 0.24–0.32). They are
  strongest for human vocal sounds and absent for nature.
- **Gemma's audio–text margin for sound events is about 0.03 cosine** (CLAP about 0.3–0.5). That is why Gemma
  zero-shot is weak despite a good audio space.
- **Centring (removing modality means) helps category retrieval a few points, but can hurt specific families**
  (Gemma animals). The sister project showed that centring hurts *vibe* retrieval by ear, so treat centring as
  task-dependent.

---

### Q4: emotion (CREMA-D: 91 actors × 12 sentences × 6 emotions, 7442 clips)

| representation | emotion (actor-held-out) | skill | sentence (actor-held-out) | skill | speaker (sentence-held-out) | skill |
|---|---|---|---|---|---|---|
| gemma | 56% (54%–57%) | 0.47 | 100% (100%–100%) | 1.00 | 6% (5%–7%) | 0.05 |
| clap | 58% (55%–59%) | 0.49 | 88% (87%–89%) | 0.87 | 48% (42%–54%) | 0.47 |
| prosody | 44% (42%–45%) | 0.33 | 38% (37%–40%) | 0.33 | 16% (13%–19%) | 0.15 |
| prosody_mfcc | 47% (45%–49%) | 0.36 | 72% (70%–75%) | 0.70 | 49% (41%–58%) | 0.48 |

| model | zero-shot raw | zero-shot centred | ANG probe / ZS recall | DIS probe / ZS recall | FEA probe / ZS recall | HAP probe / ZS recall | NEU probe / ZS recall | SAD probe / ZS recall |
|---|---|---|---|---|---|---|---|---|
| gemma | 15% | 19% | 74% / 2% | 48% / 0% | 45% / 1% | 49% / 5% | 69% / 96% | 54% / 1% |
| clap | 16% | 36% | 75% / 0% | 47% / 6% | 47% / 3% | 55% / 0% | 64% / 100% | 59% / 0% |

Gemma probe confusion (rows = true, cols = predicted, order ANG, DIS, FEA, HAP, NEU, SAD): `[[941, 68, 43, 145, 63, 11], [104, 607, 122, 135, 85, 218], [81, 107, 567, 174, 91, 251], [184, 137, 182, 619, 101, 48], [39, 54, 67, 75, 747, 105], [21, 178, 178, 42, 165, 687]]`

**Interpretation:**
- **Emotion is in the audio:** probes reach about 56–58%, above prosody features (about 44%). Anger is the easiest
  emotion.
- **Words can't reach it:** raw zero-shot labels almost everything "neutral". Verdict: **knows but can't name.**
- **Gemma encodes *what* is said almost perfectly, and *who* says it barely.** CLAP keeps speaker identity. This
  matches "Gemma is a speech/semantics model with audio grafted on", and it explains why the sister's "heated
  argument" search found overlapping talk but not hostility.

---

### Q5, part 1: genre and region words on instrumental music (FMA, artist-grouped)

Instrumental tracks: 3050, vocal tracks: 927.

**gemma** (best word form, chosen on tuning fold: `sentence_raw`; forms on tuning fold: bare_raw 14%, bare_centred 16%, phrase_raw 23%, phrase_centred 25%, sentence_raw 26%, sentence_centred 24%)

| genre | instrumental tracks | artists | probe recall | zero-shot recall | precision@25 (Wilson) | base rate |
|---|---|---|---|---|---|---|
| blues | 71 | 38 | 0% | 12% | 4% (1%–20%) | 3% |
| classical | 258 | 128 | 46% | 71% | 52% (34%–70%) | 12% |
| country | 88 | 50 | 2% | 0% | 24% (12%–43%) | 3% |
| easy listening | 54 | 38 | 0% | 0% | 0% (0%–13%) | 3% |
| electronic | 420 | 280 | 61% | 66% | 36% (20%–55%) | 17% |
| folk | 394 | 182 | 61% | 13% | 72% (52%–86%) | 16% |
| hip-hop | 143 | 82 | 11% | 9% | 36% (20%–55%) | 6% |
| jazz | 233 | 141 | 24% | 1% | 20% (9%–39%) | 9% |
| pop | 203 | 137 | 17% | 16% | 32% (17%–52%) | 9% |
| rock | 431 | 262 | 58% | 15% | 84% (65%–94%) | 18% |
| soul | 70 | 35 | 0% | 2% | 0% (0%–13%) | 3% |

Overall: probe 41% (38%–43%) · zero-shot instrumental 27% (24%–29%) · zero-shot on vocal tracks 30% (26%–34%) · true genre in top 3 51% (chance 27%)

Regions (gemma): probe 34% (30%–38%), zero-shot 21% (18%–24%), top-3 49% (chance 9%, top-3 chance 27%). Per region: African n=133 ZS 46%; Balkan n=262 ZS 13%; Brazilian n=75 ZS 25%; East Asian n=46 ZS 43%; European n=242 ZS 19%; French n=116 ZS 20%; Indian n=39 ZS 27%; Latin n=48 ZS 2%; Latin American n=219 ZS 15%; Middle Eastern n=56 ZS 9%; Spanish n=48 ZS 39%

Canonical classical (gemma): Musopen 116 movements → labelled classical 99%; FMA classical recall 73%; mean classical score Musopen 0.05, FMA classical 0.04, other genres 0.01.

**clap** (best word form, chosen on tuning fold: `sentence_centred`; forms on tuning fold: bare_raw 13%, bare_centred 17%, phrase_raw 27%, phrase_centred 25%, sentence_raw 27%, sentence_centred 34%)

| genre | instrumental tracks | artists | probe recall | zero-shot recall | precision@25 (Wilson) | base rate |
|---|---|---|---|---|---|---|
| blues | 71 | 38 | 0% | 33% | 20% (9%–39%) | 3% |
| classical | 258 | 128 | 55% | 60% | 44% (27%–63%) | 12% |
| country | 88 | 50 | 0% | 4% | 4% (1%–20%) | 3% |
| easy listening | 54 | 38 | 0% | 17% | 12% (4%–30%) | 3% |
| electronic | 420 | 280 | 71% | 44% | 32% (17%–52%) | 17% |
| folk | 394 | 182 | 67% | 34% | 68% (48%–83%) | 16% |
| hip-hop | 143 | 82 | 39% | 41% | 44% (27%–63%) | 6% |
| jazz | 233 | 141 | 47% | 30% | 24% (12%–43%) | 9% |
| pop | 203 | 137 | 19% | 5% | 12% (4%–30%) | 9% |
| rock | 431 | 262 | 62% | 52% | 56% (37%–73%) | 18% |
| soul | 70 | 35 | 2% | 2% | 8% (2%–25%) | 3% |

Overall: probe 49% (46%–52%) · zero-shot instrumental 37% (34%–40%) · zero-shot on vocal tracks 30% (26%–35%) · true genre in top 3 61% (chance 27%)

Regions (clap): probe 35% (31%–40%), zero-shot 17% (13%–21%), top-3 33% (chance 9%, top-3 chance 27%). Per region: African n=133 ZS 0%; Balkan n=262 ZS 0%; Brazilian n=75 ZS 6%; East Asian n=46 ZS 0%; European n=242 ZS 10%; French n=116 ZS 0%; Indian n=39 ZS 0%; Latin n=48 ZS 0%; Latin American n=219 ZS 86%; Middle Eastern n=56 ZS 5%; Spanish n=48 ZS 0%

Canonical classical (clap): Musopen 116 movements → labelled classical 100%; FMA classical recall 58%; mean classical score Musopen 0.36, FMA classical 0.17, other genres -0.05.

**Phase 8 genres** (rule fixed in advance: precision@25 ≥ 2× base rate and Wilson lower bound > base rate, Gemma):
classical, country, electronic, folk, hip-hop, pop, rock; contrast genre: easy listening.

**Interpretation:**
- **Genre is partly in the audio** (probe about 0.41–0.49 over 11 genres, chance 0.09).
- **Genre words work for some genres** (rock, folk, classical; hip-hop and country relative to their rarity) and
  not for others (easy listening, soul, blues).
- **The sentence form ("a piece of X music") beats the bare word** for both models.
- **Lyrics barely change zero-shot genre accuracy.**
- **"Classical" generalises from FMA's indie classical to canonical recordings** (Musopen about 99–100%).
- **Region words beat chance but are only a coarse filter**: the true region is in the top 3 about half the time.

---

### Q5, part 2: what makes a genre (perturbations, features, instrument held fixed, descriptors)

Perturbations (deterministic; parameters in `src/audiovocab/perturb.py`): `harmonic_only` = remove percussion (HPSS harmonic part); `percussive_only` = remove harmony (HPSS percussive part); `tempo_x0.75` = time-stretch to 0.75x tempo, pitch kept; `tempo_x1.33` = time-stretch to 1.33x tempo, pitch kept; `pitch_-3` = pitch shift -3 semitones; `pitch_+3` = pitch shift +3 semitones; `lowpass_1500` = darker: 4th-order low-pass at 1.5 kHz; `highpass_500` = brighter/thinner: 4th-order high-pass at 500 Hz; `compress` = flatten dynamics: divide by a 50 ms RMS envelope; `shuffle_0.5s` = shuffle 0.5 s chunks (seed 0): keeps timbre, destroys melody and phrasing; `reverse` = play backwards.
Applied to non-ND-licensed instrumental FMA clips, 16 per genre, re-embedded with both models.

#### classical

*gemma*: member score 0.04 vs others 0.02 (gap 0.02); members ranked classical first 81%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | +0.0083 (+0.0043…+0.0135) | +35% | 88% | +0.0201 |
| percussive_only | -0.0245 (-0.0307…-0.0183) | -105% | 31% | -0.0180 |
| tempo_x0.75 | +0.0016 (-0.0009…+0.0050) | +7% | 81% | +0.0075 |
| tempo_x1.33 | +0.0017 (-0.0004…+0.0038) | +7% | 88% | +0.0032 |
| pitch_-3 | +0.0012 (-0.0028…+0.0058) | +5% | 81% | +0.0049 |
| pitch_+3 | +0.0016 (-0.0013…+0.0052) | +7% | 75% | +0.0058 |
| lowpass_1500 | +0.0036 (-0.0032…+0.0114) | +15% | 88% | +0.0077 |
| highpass_500 | -0.0023 (-0.0047…+0.0009) | -10% | 88% | -0.0005 |
| compress | -0.0011 (-0.0027…+0.0012) | -5% | 75% | -0.0004 |
| shuffle_0.5s | -0.0136 (-0.0170…-0.0103) | -58% | 62% | -0.0076 |
| reverse | +0.0010 (-0.0014…+0.0038) | +4% | 81% | -0.0004 |

Features (Spearman with classical score, across / within genre): percussive_ratio -0.63/-0.50; bandwidth -0.58/-0.44; centroid -0.53/-0.33; mfcc1 0.52/0.33; mfcc5 -0.35/-0.29; onset_rate -0.31/-0.22. Regression CV R² 0.52 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.75 (0.68–0.82), 54 vs 103; guitar: too few (2 vs 20); strings: too few (1 vs 5); synthesizer: too few (3 vs 24).
Descriptors: nearest the word: orchestra, piano, melodic, film score, violin, string quartet; nearest the sound: piano, flute, saxophone, melodic, violin, harmonious; agreement ρ 0.72.
False positives (top 50 non-members): {'folk': 14, 'jazz': 8, 'pop': 8, 'electronic': 6, 'rock': 5, 'country': 5, 'easy listening': 4}.

*clap*: member score 0.13 vs others -0.05 (gap 0.18); members ranked classical first 56%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | +0.0152 (+0.0003…+0.0317) | +9% | 50% | +0.0657 |
| percussive_only | -0.1085 (-0.1516…-0.0697) | -62% | 12% | -0.0605 |
| tempo_x0.75 | +0.0104 (-0.0042…+0.0245) | +6% | 38% | +0.0295 |
| tempo_x1.33 | +0.0088 (-0.0054…+0.0186) | +5% | 44% | +0.0267 |
| pitch_-3 | +0.0112 (-0.0036…+0.0247) | +6% | 50% | +0.0262 |
| pitch_+3 | +0.0029 (-0.0115…+0.0156) | +2% | 44% | +0.0228 |
| lowpass_1500 | -0.0648 (-0.0891…-0.0406) | -37% | 6% | +0.0032 |
| highpass_500 | -0.0117 (-0.0352…+0.0063) | -7% | 31% | +0.0036 |
| compress | -0.0251 (-0.0409…-0.0142) | -14% | 31% | -0.0076 |
| shuffle_0.5s | -0.0648 (-0.0857…-0.0407) | -37% | 0% | -0.0334 |
| reverse | -0.0420 (-0.0784…-0.0131) | -24% | 38% | -0.0022 |

Features (Spearman with classical score, across / within genre): percussive_ratio -0.48/-0.39; bandwidth -0.47/-0.29; mfcc5 -0.40/-0.45; mfcc1 0.40/0.22; centroid -0.38/-0.15; onset_rate -0.33/-0.29. Regression CV R² 0.43 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.83 (0.77–0.89), 54 vs 103; guitar: too few (2 vs 20); strings: too few (1 vs 5); synthesizer: too few (3 vs 24).
Descriptors: nearest the word: film score, string quartet, orchestra, wedding, concert hall, choir; nearest the sound: piano, ornate, sad, organ, string quartet, orchestra; agreement ρ 0.58.
False positives (top 50 non-members): {'folk': 35, 'jazz': 6, 'pop': 4, 'electronic': 3, 'easy listening': 1, 'rock': 1}.

#### country

*gemma*: member score -0.03 vs others -0.03 (gap 0.00); members ranked country first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0006 (-0.0040…+0.0037) | -33% | 0% | -0.0015 |
| percussive_only | +0.0044 (-0.0017…+0.0096) | +244% | 0% | +0.0004 |
| tempo_x0.75 | +0.0009 (-0.0036…+0.0057) | +50% | 0% | -0.0028 |
| tempo_x1.33 | +0.0004 (-0.0035…+0.0043) | +22% | 0% | -0.0025 |
| pitch_-3 | -0.0036 (-0.0071…+0.0005) | -200% | 0% | -0.0049 |
| pitch_+3 | +0.0011 (-0.0042…+0.0066) | +61% | 0% | -0.0013 |
| lowpass_1500 | -0.0119 (-0.0173…-0.0064) | -661% | 0% | -0.0108 |
| highpass_500 | +0.0004 (-0.0047…+0.0048) | +22% | 0% | +0.0008 |
| compress | +0.0017 (-0.0004…+0.0039) | +94% | 0% | +0.0009 |
| shuffle_0.5s | -0.0007 (-0.0042…+0.0028) | -39% | 0% | -0.0012 |
| reverse | +0.0004 (-0.0037…+0.0053) | +22% | 0% | -0.0004 |

Features (Spearman with country score, across / within genre): centroid 0.37/0.43; mfcc1 -0.33/-0.36; bandwidth 0.30/0.41; dynamic_range -0.29/-0.43; mfcc2 -0.28/0.13; mfcc4 -0.20/-0.00. Regression CV R² 0.27 (n 2257).
Instrument held fixed (MTG): piano: too few (0 vs 157); guitar: too few (0 vs 22); strings: too few (0 vs 6); synthesizer: too few (0 vs 27).
Descriptors: nearest the word: melodic, acoustic, acoustic guitar, bass, piano, sad; nearest the sound: melodic, bass, saxophone, drums, flute, piano; agreement ρ 0.62.
False positives (top 50 non-members): {'folk': 16, 'rock': 13, 'pop': 6, 'electronic': 5, 'blues': 3, 'hip-hop': 2, 'classical': 2, 'jazz': 2, 'soul': 1}.

*clap*: member score 0.04 vs others 0.03 (gap 0.01); members ranked country first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0217 (-0.0287…-0.0141) | -203% | 0% | -0.0265 |
| percussive_only | -0.0046 (-0.0255…+0.0123) | -43% | 0% | -0.0029 |
| tempo_x0.75 | -0.0015 (-0.0105…+0.0047) | -14% | 0% | +0.0005 |
| tempo_x1.33 | -0.0085 (-0.0173…-0.0010) | -79% | 0% | -0.0005 |
| pitch_-3 | -0.0081 (-0.0157…-0.0008) | -76% | 0% | +0.0053 |
| pitch_+3 | -0.0058 (-0.0130…+0.0008) | -54% | 0% | -0.0011 |
| lowpass_1500 | -0.0031 (-0.0119…+0.0047) | -29% | 0% | -0.0092 |
| highpass_500 | -0.0154 (-0.0211…-0.0088) | -144% | 0% | -0.0130 |
| compress | -0.0068 (-0.0152…+0.0008) | -64% | 0% | -0.0010 |
| shuffle_0.5s | +0.0022 (-0.0019…+0.0055) | +21% | 0% | +0.0074 |
| reverse | -0.0136 (-0.0267…+0.0000) | -127% | 0% | -0.0228 |

Features (Spearman with country score, across / within genre): mfcc2 0.29/0.37; mfcc4 0.28/0.21; mfcc5 0.24/0.34; mfcc3 0.23/0.29; onset_rate 0.20/-0.06; harmony_change 0.13/-0.17. Regression CV R² 0.24 (n 2257).
Instrument held fixed (MTG): piano: too few (0 vs 157); guitar: too few (0 vs 22); strings: too few (0 vs 6); synthesizer: too few (0 vs 27).
Descriptors: nearest the word: film score, acoustic guitar, joyful, wedding, acoustic, quiet; nearest the sound: ornate, acoustic, brass, loud, saxophone, acoustic guitar; agreement ρ 0.40.
False positives (top 50 non-members): {'hip-hop': 9, 'folk': 9, 'jazz': 7, 'blues': 7, 'pop': 5, 'electronic': 5, 'rock': 4, 'soul': 2, 'classical': 2}.

#### electronic

*gemma*: member score 0.03 vs others 0.03 (gap 0.01); members ranked electronic first 50%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0125 (-0.0209…-0.0045) | -160% | 12% | -0.0057 |
| percussive_only | +0.0121 (+0.0048…+0.0193) | +155% | 75% | +0.0111 |
| tempo_x0.75 | -0.0132 (-0.0214…-0.0050) | -169% | 31% | -0.0087 |
| tempo_x1.33 | -0.0090 (-0.0171…-0.0019) | -115% | 31% | -0.0039 |
| pitch_-3 | -0.0037 (-0.0113…+0.0035) | -47% | 44% | -0.0003 |
| pitch_+3 | -0.0108 (-0.0180…-0.0027) | -138% | 44% | -0.0038 |
| lowpass_1500 | +0.0058 (-0.0000…+0.0120) | +74% | 56% | +0.0092 |
| highpass_500 | +0.0030 (-0.0030…+0.0091) | +38% | 62% | +0.0068 |
| compress | +0.0019 (-0.0012…+0.0057) | +24% | 56% | +0.0040 |
| shuffle_0.5s | +0.0092 (+0.0043…+0.0148) | +118% | 56% | +0.0061 |
| reverse | +0.0017 (-0.0041…+0.0079) | +22% | 62% | +0.0064 |

Features (Spearman with electronic score, across / within genre): percussive_ratio 0.44/0.54; mfcc5 0.33/0.24; bandwidth 0.32/0.45; key_clarity -0.29/-0.24; centroid 0.29/0.41; harmony_change -0.27/-0.02. Regression CV R² 0.31 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.71 (0.63–0.80), 27 vs 130; guitar: too few (2 vs 20); strings: too few (1 vs 5); synthesizer: AUC 0.48 (0.27–0.73), 11 vs 16.
Descriptors: nearest the word: melodic, synthesizer, rhythmic, festival, video game, electric guitar; nearest the sound: drums, rhythmic, bass, melodic, flute, harmonious; agreement ρ 0.56.
False positives (top 50 non-members): {'hip-hop': 15, 'rock': 13, 'soul': 10, 'pop': 6, 'jazz': 2, 'folk': 2, 'blues': 1, 'easy listening': 1}.

*clap*: member score -0.05 vs others -0.09 (gap 0.04); members ranked electronic first 6%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0182 (-0.0470…+0.0096) | -42% | 6% | +0.0059 |
| percussive_only | +0.0340 (-0.0047…+0.0761) | +78% | 6% | +0.0834 |
| tempo_x0.75 | -0.0531 (-0.0747…-0.0302) | -122% | 0% | -0.0294 |
| tempo_x1.33 | -0.0354 (-0.0471…-0.0211) | -81% | 0% | -0.0155 |
| pitch_-3 | -0.0471 (-0.0622…-0.0288) | -108% | 0% | -0.0278 |
| pitch_+3 | -0.0433 (-0.0650…-0.0206) | -99% | 0% | -0.0171 |
| lowpass_1500 | +0.0068 (-0.0109…+0.0257) | +16% | 0% | +0.0064 |
| highpass_500 | +0.0317 (+0.0186…+0.0471) | +73% | 6% | +0.0519 |
| compress | -0.0007 (-0.0103…+0.0096) | -2% | 6% | +0.0013 |
| shuffle_0.5s | +0.0181 (+0.0044…+0.0325) | +42% | 6% | +0.0111 |
| reverse | +0.0321 (+0.0088…+0.0572) | +74% | 6% | +0.0567 |

Features (Spearman with electronic score, across / within genre): harmony_change -0.37/-0.20; bandwidth 0.21/0.23; centroid 0.20/0.22; mfcc1 -0.19/-0.24; key_clarity -0.16/-0.12; tempo_stability 0.13/0.09. Regression CV R² 0.22 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.76 (0.69–0.85), 27 vs 130; guitar: too few (2 vs 20); strings: too few (1 vs 5); synthesizer: AUC 0.69 (0.52–0.85), 11 vs 16.
Descriptors: nearest the word: joyful, melodic, film score, loud, dramatic, sad; nearest the sound: ornate, loud, melodic, piano, fast, complex; agreement ρ 0.53.
False positives (top 50 non-members): {'rock': 10, 'jazz': 9, 'pop': 8, 'hip-hop': 6, 'folk': 6, 'classical': 4, 'soul': 4, 'easy listening': 3}.

#### folk

*gemma*: member score -0.01 vs others -0.01 (gap 0.00); members ranked folk first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | +0.0030 (-0.0004…+0.0066) | +120% | 0% | +0.0069 |
| percussive_only | -0.0003 (-0.0071…+0.0061) | -12% | 0% | -0.0032 |
| tempo_x0.75 | -0.0009 (-0.0052…+0.0036) | -36% | 0% | -0.0022 |
| tempo_x1.33 | -0.0001 (-0.0054…+0.0054) | -4% | 6% | -0.0002 |
| pitch_-3 | -0.0094 (-0.0151…-0.0037) | -376% | 0% | -0.0080 |
| pitch_+3 | +0.0031 (-0.0023…+0.0090) | +124% | 0% | +0.0031 |
| lowpass_1500 | -0.0061 (-0.0121…-0.0006) | -244% | 0% | -0.0044 |
| highpass_500 | -0.0046 (-0.0093…-0.0007) | -184% | 0% | -0.0016 |
| compress | -0.0026 (-0.0053…-0.0005) | -104% | 0% | -0.0011 |
| shuffle_0.5s | -0.0075 (-0.0121…-0.0030) | -300% | 0% | -0.0044 |
| reverse | -0.0054 (-0.0115…+0.0003) | -216% | 0% | -0.0039 |

Features (Spearman with folk score, across / within genre): harmony_change 0.44/0.46; percussive_ratio -0.29/-0.01; mfcc5 -0.29/-0.04; key_clarity 0.27/0.14; mfcc4 -0.21/-0.28; bandwidth -0.19/0.23. Regression CV R² 0.33 (n 2257).
Instrument held fixed (MTG): piano: too few (2 vs 155); guitar: too few (0 vs 22); strings: too few (0 vs 6); synthesizer: too few (0 vs 27).
Descriptors: nearest the word: acoustic, acoustic guitar, melodic, violin, piano, rhythmic; nearest the sound: flute, saxophone, melodic, piano, orchestra, bass; agreement ρ 0.67.
False positives (top 50 non-members): {'jazz': 15, 'pop': 10, 'rock': 9, 'country': 5, 'electronic': 4, 'classical': 3, 'hip-hop': 1, 'soul': 1, 'blues': 1, 'easy listening': 1}.

*clap*: member score 0.05 vs others 0.03 (gap 0.02); members ranked folk first 19%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | +0.0621 (+0.0419…+0.0824) | +263% | 50% | +0.0671 |
| percussive_only | -0.0365 (-0.0611…-0.0100) | -155% | 6% | -0.0288 |
| tempo_x0.75 | +0.0491 (+0.0312…+0.0666) | +208% | 38% | +0.0494 |
| tempo_x1.33 | +0.0426 (+0.0281…+0.0585) | +181% | 44% | +0.0495 |
| pitch_-3 | +0.0358 (+0.0193…+0.0510) | +152% | 44% | +0.0383 |
| pitch_+3 | +0.0523 (+0.0307…+0.0774) | +222% | 50% | +0.0562 |
| lowpass_1500 | +0.0077 (-0.0106…+0.0271) | +33% | 25% | +0.0157 |
| highpass_500 | +0.0349 (+0.0125…+0.0588) | +148% | 31% | +0.0327 |
| compress | -0.0094 (-0.0220…+0.0013) | -40% | 12% | +0.0001 |
| shuffle_0.5s | -0.0153 (-0.0325…-0.0014) | -65% | 6% | -0.0170 |
| reverse | -0.0236 (-0.0519…+0.0006) | -100% | 6% | -0.0179 |

Features (Spearman with folk score, across / within genre): mfcc5 -0.37/-0.17; harmony_change 0.37/0.32; mfcc4 -0.35/-0.44; mfcc2 -0.33/-0.32; bandwidth -0.32/0.09; percussive_ratio -0.28/0.07. Regression CV R² 0.38 (n 2257).
Instrument held fixed (MTG): piano: too few (2 vs 155); guitar: too few (0 vs 22); strings: too few (0 vs 6); synthesizer: too few (0 vs 27).
Descriptors: nearest the word: acoustic guitar, saxophone, wedding, film score, string quartet, acoustic; nearest the sound: acoustic, ornate, saxophone, acoustic guitar, brass, orchestra; agreement ρ 0.64.
False positives (top 50 non-members): {'jazz': 19, 'classical': 17, 'country': 4, 'blues': 3, 'electronic': 3, 'rock': 2, 'hip-hop': 1, 'pop': 1}.

#### hip-hop

*gemma*: member score 0.00 vs others -0.01 (gap 0.02); members ranked hip-hop first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0290 (-0.0375…-0.0193) | -166% | 0% | -0.0165 |
| percussive_only | +0.0312 (+0.0209…+0.0424) | +178% | 6% | +0.0356 |
| tempo_x0.75 | -0.0111 (-0.0161…-0.0068) | -63% | 0% | -0.0053 |
| tempo_x1.33 | -0.0069 (-0.0132…-0.0006) | -39% | 0% | -0.0021 |
| pitch_-3 | -0.0047 (-0.0117…+0.0020) | -27% | 0% | +0.0004 |
| pitch_+3 | -0.0092 (-0.0149…-0.0032) | -53% | 0% | -0.0035 |
| lowpass_1500 | +0.0029 (-0.0024…+0.0077) | +17% | 0% | +0.0044 |
| highpass_500 | +0.0058 (-0.0005…+0.0124) | +33% | 0% | +0.0084 |
| compress | -0.0002 (-0.0046…+0.0042) | -1% | 0% | +0.0006 |
| shuffle_0.5s | +0.0053 (+0.0006…+0.0099) | +30% | 0% | +0.0077 |
| reverse | -0.0039 (-0.0128…+0.0048) | -22% | 0% | +0.0048 |

Features (Spearman with hip-hop score, across / within genre): percussive_ratio 0.64/0.31; bandwidth 0.45/0.27; centroid 0.44/0.29; mfcc1 -0.41/-0.29; key_clarity -0.33/-0.46; mfcc5 0.31/0.16. Regression CV R² 0.44 (n 2257).
Instrument held fixed (MTG): piano: too few (7 vs 150); guitar: too few (2 vs 20); strings: too few (0 vs 6); synthesizer: too few (3 vs 24).
Descriptors: nearest the word: rhythmic, melodic, video game, film score, drums, brass; nearest the sound: drums, rhythmic, bass, melodic, orchestra, harmonious; agreement ρ 0.79.
False positives (top 50 non-members): {'electronic': 22, 'rock': 12, 'soul': 6, 'country': 3, 'blues': 2, 'pop': 2, 'jazz': 1, 'folk': 1, 'easy listening': 1}.

*clap*: member score 0.05 vs others 0.00 (gap 0.05); members ranked hip-hop first 6%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0810 (-0.0976…-0.0621) | -162% | 0% | -0.0515 |
| percussive_only | +0.0403 (+0.0173…+0.0651) | +81% | 50% | +0.0559 |
| tempo_x0.75 | -0.0181 (-0.0313…-0.0065) | -36% | 0% | -0.0211 |
| tempo_x1.33 | -0.0136 (-0.0261…-0.0005) | -27% | 6% | -0.0097 |
| pitch_-3 | -0.0123 (-0.0211…-0.0020) | -25% | 0% | -0.0124 |
| pitch_+3 | -0.0093 (-0.0213…+0.0037) | -19% | 0% | -0.0148 |
| lowpass_1500 | -0.0307 (-0.0426…-0.0172) | -61% | 0% | -0.0239 |
| highpass_500 | -0.0126 (-0.0384…+0.0102) | -25% | 12% | +0.0026 |
| compress | -0.0094 (-0.0152…-0.0039) | -19% | 0% | +0.0018 |
| shuffle_0.5s | +0.0064 (-0.0004…+0.0142) | +13% | 6% | +0.0064 |
| reverse | -0.0409 (-0.0529…-0.0278) | -82% | 0% | -0.0147 |

Features (Spearman with hip-hop score, across / within genre): percussive_ratio 0.60/0.33; bandwidth 0.55/0.30; centroid 0.48/0.30; mfcc1 -0.46/-0.39; mfcc5 0.42/0.30; onset_rate 0.28/-0.04. Regression CV R² 0.54 (n 2257).
Instrument held fixed (MTG): piano: too few (7 vs 150); guitar: too few (2 vs 20); strings: too few (0 vs 6); synthesizer: too few (3 vs 24).
Descriptors: nearest the word: film score, wedding, drums, joyful, dramatic, acoustic guitar; nearest the sound: ornate, fast, loud, piano, video game, slow; agreement ρ 0.31.
False positives (top 50 non-members): {'electronic': 28, 'soul': 8, 'rock': 7, 'jazz': 3, 'classical': 2, 'easy listening': 1, 'folk': 1}.

#### pop

*gemma*: member score 0.01 vs others -0.01 (gap 0.01); members ranked pop first 25%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0147 (-0.0223…-0.0070) | -109% | 6% | -0.0081 |
| percussive_only | +0.0016 (-0.0038…+0.0075) | +12% | 25% | +0.0058 |
| tempo_x0.75 | -0.0058 (-0.0108…-0.0006) | -43% | 12% | -0.0001 |
| tempo_x1.33 | -0.0007 (-0.0044…+0.0030) | -5% | 19% | +0.0009 |
| pitch_-3 | -0.0039 (-0.0092…+0.0017) | -29% | 19% | +0.0006 |
| pitch_+3 | -0.0030 (-0.0080…+0.0022) | -22% | 19% | +0.0032 |
| lowpass_1500 | -0.0122 (-0.0188…-0.0058) | -90% | 12% | -0.0044 |
| highpass_500 | +0.0095 (+0.0056…+0.0140) | +70% | 19% | +0.0126 |
| compress | +0.0013 (-0.0003…+0.0030) | +10% | 31% | +0.0001 |
| shuffle_0.5s | +0.0012 (-0.0042…+0.0058) | +9% | 25% | +0.0056 |
| reverse | -0.0052 (-0.0112…+0.0014) | -39% | 19% | -0.0014 |

Features (Spearman with pop score, across / within genre): centroid 0.56/0.49; mfcc1 -0.50/-0.47; bandwidth 0.50/0.50; percussive_ratio 0.46/0.33; onset_rate 0.16/0.17; dynamic_range -0.15/-0.14. Regression CV R² 0.29 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.73 (0.63–0.83), 20 vs 137; guitar: too few (6 vs 16); strings: too few (0 vs 6); synthesizer: too few (5 vs 22).
Descriptors: nearest the word: melodic, piano, acoustic, rhythmic, joyful, bass; nearest the sound: melodic, bass, drums, rhythmic, saxophone, harmonious; agreement ρ 0.57.
False positives (top 50 non-members): {'rock': 17, 'hip-hop': 7, 'folk': 7, 'country': 4, 'electronic': 4, 'soul': 3, 'classical': 2, 'jazz': 2, 'easy listening': 2, 'blues': 2}.

*clap*: member score 0.01 vs others 0.00 (gap 0.01); members ranked pop first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0204 (-0.0336…-0.0066) | -181% | 0% | -0.0235 |
| percussive_only | +0.0004 (-0.0116…+0.0119) | +4% | 0% | +0.0031 |
| tempo_x0.75 | -0.0405 (-0.0489…-0.0339) | -358% | 0% | -0.0293 |
| tempo_x1.33 | -0.0215 (-0.0294…-0.0135) | -190% | 0% | -0.0191 |
| pitch_-3 | -0.0232 (-0.0322…-0.0125) | -205% | 0% | -0.0195 |
| pitch_+3 | -0.0319 (-0.0395…-0.0240) | -282% | 0% | -0.0249 |
| lowpass_1500 | +0.0024 (-0.0123…+0.0178) | +21% | 0% | +0.0025 |
| highpass_500 | +0.0106 (+0.0019…+0.0193) | +94% | 6% | +0.0093 |
| compress | -0.0022 (-0.0063…+0.0025) | -19% | 0% | -0.0008 |
| shuffle_0.5s | +0.0115 (+0.0024…+0.0217) | +102% | 0% | +0.0141 |
| reverse | +0.0046 (-0.0096…+0.0178) | +41% | 0% | +0.0037 |

Features (Spearman with pop score, across / within genre): bandwidth 0.32/0.32; mfcc2 0.30/0.41; mfcc5 0.29/0.33; mfcc4 0.29/0.33; harmony_change -0.26/-0.08; tempo_stability 0.21/0.14. Regression CV R² 0.26 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.60 (0.51–0.68), 20 vs 137; guitar: too few (6 vs 16); strings: too few (0 vs 6); synthesizer: too few (5 vs 22).
Descriptors: nearest the word: film score, joyful, melodic, loud, wedding, quiet; nearest the sound: ornate, loud, fast, video game, joyful, acoustic; agreement ρ 0.30.
False positives (top 50 non-members): {'electronic': 30, 'hip-hop': 9, 'soul': 4, 'folk': 3, 'easy listening': 2, 'rock': 1, 'classical': 1}.

#### rock

*gemma*: member score -0.00 vs others -0.01 (gap 0.01); members ranked rock first 6%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0057 (-0.0105…-0.0008) | -54% | 6% | -0.0085 |
| percussive_only | +0.0121 (+0.0025…+0.0224) | +114% | 6% | +0.0199 |
| tempo_x0.75 | +0.0036 (-0.0002…+0.0074) | +34% | 12% | +0.0035 |
| tempo_x1.33 | +0.0058 (-0.0001…+0.0124) | +55% | 19% | +0.0027 |
| pitch_-3 | +0.0083 (+0.0030…+0.0148) | +78% | 19% | +0.0081 |
| pitch_+3 | -0.0010 (-0.0054…+0.0032) | -9% | 12% | -0.0003 |
| lowpass_1500 | -0.0062 (-0.0163…+0.0019) | -58% | 0% | -0.0048 |
| highpass_500 | +0.0025 (-0.0076…+0.0106) | +24% | 0% | +0.0068 |
| compress | +0.0020 (-0.0001…+0.0044) | +19% | 12% | +0.0029 |
| shuffle_0.5s | +0.0030 (-0.0004…+0.0066) | +28% | 6% | +0.0017 |
| reverse | +0.0065 (+0.0005…+0.0116) | +61% | 6% | +0.0095 |

Features (Spearman with rock score, across / within genre): centroid 0.47/0.45; percussive_ratio 0.46/0.33; bandwidth 0.41/0.32; mfcc1 -0.38/-0.39; harmony_change -0.35/-0.27; dynamic_range -0.28/-0.33. Regression CV R² 0.45 (n 2257).
Instrument held fixed (MTG): piano: too few (8 vs 149); guitar: too few (4 vs 18); strings: too few (1 vs 5); synthesizer: too few (4 vs 23).
Descriptors: nearest the word: electric guitar, bass, melodic, rhythmic, acoustic, drums; nearest the sound: drums, bass, melodic, rhythmic, harmonious, orchestra; agreement ρ 0.68.
False positives (top 50 non-members): {'electronic': 16, 'hip-hop': 9, 'pop': 7, 'jazz': 6, 'classical': 5, 'country': 3, 'soul': 2, 'folk': 1, 'blues': 1}.

*clap*: member score 0.11 vs others 0.02 (gap 0.09); members ranked rock first 44%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0713 (-0.0904…-0.0534) | -80% | 25% | -0.0685 |
| percussive_only | -0.0658 (-0.1035…-0.0271) | -73% | 12% | +0.0047 |
| tempo_x0.75 | -0.0151 (-0.0298…+0.0007) | -17% | 44% | +0.0088 |
| tempo_x1.33 | -0.0092 (-0.0205…+0.0017) | -10% | 31% | -0.0058 |
| pitch_-3 | -0.0193 (-0.0347…-0.0033) | -22% | 44% | +0.0047 |
| pitch_+3 | -0.0259 (-0.0453…-0.0077) | -29% | 31% | -0.0052 |
| lowpass_1500 | -0.0802 (-0.1170…-0.0430) | -90% | 12% | -0.0396 |
| highpass_500 | -0.0714 (-0.0908…-0.0518) | -80% | 25% | -0.0438 |
| compress | +0.0087 (-0.0038…+0.0209) | +10% | 38% | +0.0094 |
| shuffle_0.5s | +0.0089 (-0.0026…+0.0218) | +10% | 44% | +0.0013 |
| reverse | -0.0259 (-0.0496…-0.0063) | -29% | 38% | -0.0082 |

Features (Spearman with rock score, across / within genre): percussive_ratio 0.45/0.38; bandwidth 0.44/0.42; centroid 0.42/0.46; mfcc3 0.39/0.27; mfcc1 -0.36/-0.42; mfcc5 0.29/0.14. Regression CV R² 0.42 (n 2257).
Instrument held fixed (MTG): piano: too few (8 vs 149); guitar: too few (4 vs 18); strings: too few (1 vs 5); synthesizer: too few (4 vs 23).
Descriptors: nearest the word: film score, acoustic guitar, dramatic, wedding, loud, drums; nearest the sound: ornate, loud, elegant, brass, film score, dramatic; agreement ρ 0.42.
False positives (top 50 non-members): {'country': 12, 'folk': 7, 'pop': 7, 'jazz': 6, 'soul': 5, 'blues': 5, 'electronic': 4, 'hip-hop': 3, 'easy listening': 1}.

#### easy listening

*gemma*: member score -0.00 vs others 0.00 (gap -0.01); members ranked easy listening first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | +0.0037 (+0.0005…+0.0071) | n/a | 0% | +0.0002 |
| percussive_only | -0.0202 (-0.0256…-0.0132) | n/a | 0% | -0.0215 |
| tempo_x0.75 | +0.0037 (-0.0013…+0.0092) | n/a | 0% | +0.0016 |
| tempo_x1.33 | -0.0022 (-0.0078…+0.0032) | n/a | 0% | -0.0010 |
| pitch_-3 | +0.0010 (-0.0030…+0.0050) | n/a | 0% | -0.0006 |
| pitch_+3 | +0.0009 (-0.0040…+0.0051) | n/a | 0% | +0.0003 |
| lowpass_1500 | +0.0070 (+0.0034…+0.0113) | n/a | 0% | +0.0015 |
| highpass_500 | -0.0049 (-0.0087…-0.0010) | n/a | 0% | -0.0090 |
| compress | +0.0004 (-0.0012…+0.0024) | n/a | 0% | -0.0023 |
| shuffle_0.5s | +0.0006 (-0.0015…+0.0032) | n/a | 0% | -0.0012 |
| reverse | -0.0015 (-0.0055…+0.0033) | n/a | 0% | -0.0067 |

Features (Spearman with easy listening score, across / within genre): percussive_ratio -0.31/-0.48; mfcc2 0.28/0.19; centroid -0.21/-0.23; onset_rate -0.18/-0.43; mfcc4 0.18/-0.06; key_clarity 0.18/0.33. Regression CV R² 0.24 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.54 (0.43–0.64), 15 vs 142; guitar: too few (0 vs 22); strings: too few (2 vs 4); synthesizer: too few (2 vs 25).
Descriptors: nearest the word: simple, harmonious, melodic, acoustic, rhythmic, piano; nearest the sound: drums, melodic, bass, rhythmic, harmonious, saxophone; agreement ρ 0.50.
False positives (top 50 non-members): {'folk': 17, 'pop': 14, 'country': 6, 'rock': 4, 'soul': 3, 'classical': 2, 'blues': 2, 'hip-hop': 1, 'jazz': 1}.

*clap*: member score -0.12 vs others -0.11 (gap -0.01); members ranked easy listening first 0%; n = 16 members, 134 others.

| perturbation | Δ score (members, 95% CI) | share of gap | still ranked first | Δ non-members |
|---|---|---|---|---|
| harmonic_only | -0.0016 (-0.0189…+0.0159) | n/a | 0% | +0.0019 |
| percussive_only | -0.0391 (-0.0765…+0.0008) | n/a | 0% | -0.0265 |
| tempo_x0.75 | -0.0651 (-0.0849…-0.0435) | n/a | 0% | -0.0324 |
| tempo_x1.33 | -0.0403 (-0.0585…-0.0234) | n/a | 0% | -0.0255 |
| pitch_-3 | -0.0447 (-0.0590…-0.0313) | n/a | 0% | -0.0306 |
| pitch_+3 | -0.0597 (-0.0747…-0.0436) | n/a | 0% | -0.0310 |
| lowpass_1500 | -0.0160 (-0.0341…+0.0003) | n/a | 0% | -0.0159 |
| highpass_500 | -0.0164 (-0.0348…+0.0053) | n/a | 0% | -0.0178 |
| compress | -0.0073 (-0.0188…+0.0024) | n/a | 0% | -0.0100 |
| shuffle_0.5s | +0.0055 (-0.0053…+0.0180) | n/a | 0% | +0.0069 |
| reverse | +0.0414 (+0.0101…+0.0795) | n/a | 0% | +0.0336 |

Features (Spearman with easy listening score, across / within genre): percussive_ratio -0.31/-0.34; mfcc2 0.28/0.22; harmony_change -0.26/-0.25; mfcc4 0.26/0.09; centroid -0.19/-0.01; mfcc1 0.13/0.04. Regression CV R² 0.25 (n 2257).
Instrument held fixed (MTG): piano: AUC 0.65 (0.53–0.78), 15 vs 142; guitar: too few (0 vs 22); strings: too few (2 vs 4); synthesizer: too few (2 vs 25).
Descriptors: nearest the word: joyful, film score, quiet, melodic, loud, sad; nearest the sound: ornate, loud, piano, joyful, melodic, film score; agreement ρ 0.23.
False positives (top 50 non-members): {'electronic': 26, 'classical': 8, 'folk': 5, 'jazz': 2, 'country': 2, 'pop': 2, 'rock': 2, 'soul': 1, 'hip-hop': 1, 'blues': 1}.

**Interpretation for "classical"** (the genre the README asked about):
- **Keeping only the drums erases all of the classical signal.** Removing drums makes *any* track more classical.
- **Shuffling 0.5 s chunks hurts.** In Gemma, reversing barely matters (about 4% of the gap), so it reacts to continuity
  (the cuts) rather than melodic direction. In CLAP, reversing also hurts (about 24%).
- **Tempo, pitch and dynamics barely matter.**
- **Features:** the classical score falls with percussive share, bandwidth and brightness.
- **It's more than "piano":** classical piano vs other piano gives AUC 0.75 (Gemma) and 0.83 (CLAP).
- **Summary:** to these models, "classical" = **tonal, drum-free, darker/narrower-band texture**. It is not tempo
  or key, and only weakly composition.
- **Caveat:** 16 member tracks per genre, so read effects with their CIs. For the contrast genre (easy listening)
  the word doesn't single out its own tracks at all.

---

### Bridge back to Vibe Search

| Vibe Search query | listener (of 4) | classes mapped (fixed in advance) | probe | zero-shot | verdict |
|---|---|---|---|---|---|
| two friends laughing about football, late-night energy | 4 | laughing (esc50), laughter (vocalsound) | 96% | 100% | space and words fine here: likely a podcast-corpus data gap |
| calm, slow, one voice, like a bedtime story | 4 | NEU (cremad), SAD (cremad) | 61% | 48% | weak area of the space |
| heated argument, people talking over each other | 1 | ANG (cremad), Shout (fsd50k), Screaming (fsd50k) | 64% | 43% | weak area of the space |
| soft whispering, very quiet | 2 | Whispering (fsd50k) | 39% | 3% | weak area of the space |
| excited fast talking about sports | 3 | HAP (cremad) | 49% | 5% | weak area of the space |
| serious news voice reading the headlines | 3 | Male_speech_and_man_speaking (fsd50k), Female_speech_and_woman_speaking (fsd50k) | 27% | 16% | weak area of the space |
| dramatic storytelling about a murder | 3 | FEA (cremad) | 45% | 1% | weak area of the space |
| rain and nature sounds, no talking | 1 | rain (esc50), wind (esc50), Rain (fsd50k), Stream (fsd50k) | 69% | 16% | weak area of the space |
| upbeat music and singing | 2 | Singing (fsd50k) | 57% | 51% | weak area of the space |
| thoughtful slow interview about life | 3 | NEU (cremad) | 69% | 96% | weak area of the space |
| kids story with funny voices | 4 | Child_speech_and_kid_speaking (fsd50k) | 31% | 12% | weak area of the space |
| relaxed chat between comedians, lots of laughs | 3 | laughing (esc50), laughter (vocalsound), Giggle (fsd50k) | 89% | 97% | space and words fine here: likely a podcast-corpus data gap |

Rule: probe < 70% → weak area; probe ≥ 70% but zero-shot < 50% → knows but can't name; both high → likely a podcast-data gap.
**Result:**
- The rule marks most searches "weak", including ones that scored 3–4 of 4 by ear. **Per-class resolution on
  labelled sounds doesn't predict podcast vibe success.**
- "Heated argument" unpacks to *knows but can't name* for anger.
- Whispering, rain/nature and music are genuinely weak or rare areas.

---

## 6. Method lessons (things that went wrong and how they were caught)

1. **Gemma's narrow cone breaks fixed-C probes.** The first ESC-50 probe (0.71) scored below kNN (0.77). The fix
   was to standardise features (decision 17).
2. **One fixed C can't fit 2,000-clip sound sets and 150-track genres.** The genre probe scored *below* zero-shot.
   The fix was nested C selection inside the training folds (decision 21). **Sanity rule:** probe < kNN or probe <
   zero-shot means a probe problem, not a finding.
3. **The pre-set Whisper rule caught only 37% of vocal tracks.** The calibration set now chooses the rule
   (decision 19).
4. **A chained job started early.** Its process check fell between a stop and restart of Whisper. The results were
   discarded and rerun (decision 20).
5. **Perturbation deltas are tiny in Gemma's cone** (about 0.01–0.02). Always scale them by the genre's
   member-vs-rest gap.
6. **Data/infra gotchas:**
   - pandas 3 string dtype breaks numpy (`.values` of a string column).
   - `df.clip` is a DataFrame method; use `df["clip"]`.
   - Python `zipfile` can't open multi-disk zip64 archives; join with `zip -s 0`.
   - ffmpeg's own TLS rejects the corporate proxy certificate; download with Python + truststore.
   - The proxy drops HEAD requests to some hosts; use a 1-byte ranged GET.
   - Zenodo rate-limits the corporate IP.

## 7. Reusable artefacts (how to pick up where this left off)

- **Clips:** `data/clips.parquet`, one row per embedded window. Columns: `clip_id, dataset, item_id, window,
  source, start, dur, label, family, groups, group_id, split, licence`, plus dataset-specific columns (`genre,
  region, instr_tag, nd, pool, sentence, mtg_genres, instruments, composer`).
- **Embeddings (gitignored, local):**
  - `index/gemma.npy` (N×768) and `index/clap.npy` (N×512), row-aligned with `index/<model>_clip_ids.json`;
  - per-clip cache in `runs/cache/emb_<model>/`, keyed by
    `{dataset, clip, source, start, dur[, perturb]}` (see `embed.clip_key`).
  - Load with `audiovocab.data.subset(model, dataset)` → (rows, X). Zero rows = undecodable clips (30).
- **Text side:** `index/text.parquet` (789 texts: class names and descriptions, genre/region word forms, synonyms,
  emotion prompts, descriptors, 77 generic phrases), plus `index/text_gemma.npy` (SearchQuery prefix),
  `index/text_gemma_none.npy` and `index/text_clap.npy`.
- **Vocal flags:** `data/vocal_flags.parquet` (FMA + MTG piano). The Whisper transcripts in `data/whisper.parquet`
  are gitignored (lyric snippets).
- **Code** (`src/audiovocab/`):
  - `embed.py` (models, cache, runner);
  - `metrics.py` (probe with nested C, kNN, zero-shot, bootstrap, effective vocabulary);
  - `splits.py`;
  - `centering.py`;
  - `perturb.py`;
  - one module per phase;
  - `page.py` + `findings*.py` for the page.
  - CLI: `vocab setup | notes | plan | fetch | clips | embed | analyse | page`.
- **Colab route** for blocked datasets: `colab/fsd50k_embed.ipynb` (self-contained; checks weights and parity) and
  `fetch.import_colab_fsd`.
- **Models** are read from `../podcast-vibe-search/models/` by absolute path (`config.yaml`). `HF_HUB_OFFLINE=1`.

## 8. How much to trust each finding

| Finding | Confidence | Why |
|---|---|---|
| Gemma words barely reach sound events; CLAP's do | **High** | large n, both directions consistent, per-class detail agrees |
| Emotion encoded but not nameable | **High** | 7,442 clips, actor-held-out, both models agree |
| Gemma: sentence ≫ speaker | **High** | huge effect, clean design |
| Bird species coarse in both models | **High** | 680 clips, recordist-held-out |
| Giggle vs chuckle "coarse" | **Medium** | only 59 test clips; CI 56–82% |
| Synonym / structure results | **Medium** | depends on hand-written synonym lists and class descriptions |
| Genre words work for rock/folk/classical | **Medium** | FMA indie music; residual vocals; artist-grouped |
| "Classical = tonal, drum-free texture" | **Medium** | perturbations causal but n = 16 per genre; features and instrument control agree |
| Region words weak | **Medium-low** | sparse regional tags, residual vocals |
| Effective vocabulary | **Low information** | saturates at the 70% threshold |
| Bridge verdicts | **Low-medium** | pre-registered mapping, but per-class recalls aren't vibe measures |

## 9. Practical guidance for a follow-on project

- **Model choice:**
  - text→audio retrieval of **sound events, animals or music genre**: **prefer CLAP**, or check Gemma per class
    first;
  - **speech content, speaking style or show-level similarity**: Gemma (sister project: 33/48 listening hits;
    sentence probe 1.00);
  - **speaker identity**: CLAP or MFCCs, not Gemma.
- **Labels:** if a label exists for training, **a linear probe on frozen embeddings beats text queries by a wide
  margin** for emotion, bird species and fine sound classes. Use text only where zero-shot is shown to work.
- **Emotion or hostility from audio:** train a small probe (CREMA-D-style data). Don't rely on prompts.
- **Prompts:**
  - Gemma: use the `SearchQuery` prefix for queries, no prefix for audio;
  - music genres: "a piece of X music" beats the bare word;
  - text vectors: average the class name and 2 descriptions.
- **Centring:** helps category lookups a little, hurts some families and (per the sister project) the vibe. Make
  it a per-use switch with α tuned on held-out data.
- **Windows:** ≤ 30 s for Gemma (anything longer is silently dropped); 10 s keeps about 0.91 of the 30 s vector.
- **Licences:** ESC-50 (NC) and MTG-Jamendo (research-only) are internal-only. FSD50K and FMA are per clip/track.
  Never host ND or NC audio publicly.
- **Compute:** this laptop does about 2–4 s per 10 s clip with Gemma. Use Colab (T4) for anything over about 5k
  clips; it matches the local vectors exactly in float32.

## 10. Open questions this project did not settle

- Does centring only the main gap direction (rather than the full mean) keep the category gains without losing the
  vibe? (Sister project's untested next step.)
- Would more laughter subtypes, beyond giggle and chuckle, show a finer human-vocal resolution? There are no
  licence-clear labels.
- A larger perturbation set, other genres' "what makes it", and source separation to remove vocals properly (out
  of scope in v1).
- Non-English words; more listeners for any vibe test.
