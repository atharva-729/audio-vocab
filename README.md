# Audio Vocab

**What can an audio embedding actually hear, and which of it can words reach?**

EmbeddingGemma 2 (Google, open weights) puts sounds and words into one 768-d space. The sister project
`podcast-vibe-search` used that space to search podcasts by how they sound. It worked for some vibes and failed for
others, and it surfaced two facts about the space:
- a **modality gap** (words form their own island; centring closes it but loses the vibe);
- **mechanics, not emotion** (it finds people talking over each other, but not hostility).

This project studies the space directly, on labelled datasets instead of podcasts. It starts from Anand's question:

> With words, a dozen synonyms of "love" form a tight cluster. What's the audio equivalent? Are there 50 kinds of
> laughter the model can tell apart? Can it finely separate types of laughter but barely tell a cuckoo from an owl,
> so it's concentrated on human expression rather than wildlife, or the other way round?

## Questions

- **Q1. Resolution:** where is the audio space fine-grained, and where is it coarse? Compare human vocal sounds,
  animals, nature, machines, music and speech emotion. (Giggle vs chuckle? Cuckoo vs owl?)
- **Q2. Knows vs can name:** when the model separates two sounds, can a *word* also find them? Or is the information
  in the audio embedding but not aligned to text?
- **Q3. Words vs sounds:** is the text vocabulary (synonyms, related words) organised like the audio vocabulary?
  Where does the modality gap sit, and does it differ by kind of sound?
- **Q4. Emotion:** can the space separate *how* something is said (angry, happy, sad), or only *that* it's said?
- **Q5. Music without lyrics:** can instrumental music be tied to **genre words** ("classical", "jazz", "folk",
  "electronic") and to **region words** ("Brazilian", "Indian")? If yes, **what makes classical sound "classical" to
  the model**, and the same for whichever other genres work: tempo, instruments, brightness, rhythm, structure, or
  something else?

All labels come from **existing labelled datasets**. No hand-labelling.

Context: an innovation-team study for Anand, published as a GitHub Pages findings page. It runs on a laptop CPU with
no meaningful LLM spend.

---

## How to use this README (for Claude Code)

- Work **one phase at a time**. At each **CHECKPOINT**, stop, show what it asks for, and wait for a go-ahead.
- Don't build later phases early. Don't add dependencies beyond the ones already used in `podcast-vibe-search`
  without asking.
- Keep it simple: Python, Parquet and NumPy files, numpy/scikit-learn/librosa/plotly, and a small CLI. No vector DB, no
  agent framework, no web framework.
- If a sanity check fails, stop and report. Don't hide it.
- The user dictates messages with Windows `Win+H`; interpret them charitably. Explain findings in plain language, but
  the audience (Anand) is technical, so keep the real terms (cosine, UMAP, linear probe).
- Every finding is worded as **"in these datasets, with this model…"**. Never as a claim about all audio.

## Ground rules

**Do**
- **Reuse what the sister project already has** (Phase 0): the model weights, the loading code, the environment fixes
  and the design rules. Don't rediscover them.
- **Verify, don't guess,** anything about the model's internals or a dataset's licence. Cite the model card, config
  file or processor code for every claim in `MODEL_NOTES.md`.
- Cache every embedding, keyed by dataset + clip id + model + preprocessing/perturbation hash. Reruns must be close to
  free.
- **Use the provided folds and splits, or group by source recording, speaker or artist, for every probe and kNN
  number.** Clips from the same recording, speaker or artist sound alike, so random splits leak and inflate the scores.
- Report counts with sample sizes, plus bootstrap CIs for the headline numbers.
- Run **CLAP as a cheap second model** on the same clips. A finding that holds for both models is about audio
  embeddings; one that holds for only one is about that model.
- Write tests for the deterministic parts: splits, metrics, centring maths and perturbations.

**Don't**
- **Don't modify `podcast-vibe-search`.** Read from it and copy from it; never write to it.
- **Don't hand-label.** Dataset labels and tags only. The only manual step is an optional short listen-through of
  examples.
- **Don't download from YouTube,** or use datasets that are YouTube ID lists (AudioSet, MusicCaps).
- Don't use any dataset whose licence is unclear (e.g. GTZAN). Skip it and note why in `data/LICENCES.md`.
- **Don't host dataset audio on the public page** unless its licence clearly allows it. Local playback is fine.
- Don't fine-tune anything. Probes are simple linear classifiers on frozen embeddings.
- **Don't try to bypass the corporate proxy.** If Hugging Face, Zenodo or other hosts are blocked, use the
  personal-laptop bundle route the sister project used.
- Don't tune thresholds, prompts, genre word lists or centring strength on the same data you report on. Hold out a
  split.
- No LLM is needed. If one is used (for example to draft synonym lists), it goes through the copied LLM Foundry client
  and spend guard, and its output is saved to a file for review before use.

---

## Phase 0: Set up from the sister project

**Goal:** a working repo that loads both models and embeds audio and text exactly as `podcast-vibe-search` does,
without redoing any of that work.

1. **Find the sister folder.** It's expected at `../podcast-vibe-search` (confirm the path; don't guess). Read its
   `README.md`, `PROJECT_STATUS.md`, `CLAUDE.md`, `config.yaml` and `pyproject.toml`.
2. **Models:** point `config.yaml` at the sister's `models/` folder (`google--embeddinggemma-2`,
   `laion--clap-htsat-unfused`) by absolute path. **Don't copy the weights.** Run with `HF_HUB_OFFLINE=1`. If the user
   later wants the repos independent, copying is a one-line change.
3. **Copy, don't import,** the small pieces this project needs. Adapt the imports to the new package name
   (`audiovocab`):
   - the model-loading and embedding part of `embed.py`:
     - float32 on CPU, never fp16;
     - `config_kwargs={"vision_config": None}` for audio + text, and also `"audio_config": None` for text only;
   - the cache helpers;
   - `centering.py`, plus its 77 `GENERIC_PHRASES`;
   - the truststore setup (`truststore.inject_into_ssl()`), needed for the corporate TLS proxy on Python 3.13+;
   - the LLM Foundry client with its spend guard ($0.80/day), only if Phase 5's synonym drafting needs it;
   - `scripts/fetch_bundle.py` / `import_bundle.py` as a template for the dataset bundle;
   - `CLAUDE.md` (page design rules), unchanged.
4. **Environment notes:** copy the environment and gotchas section of the sister's `PROJECT_STATUS.md` into
   `ENVIRONMENT.md`. It covers:
   - Windows 11, 12-core CPU, no GPU;
   - Python 3.14;
   - the proxy-blocked hosts;
   - bfloat16 being 12× slower;
   - running long jobs detached with sleep blocked;
   - Edge headless screenshots.
5. **Venv:** create a fresh `.venv` with the same pinned versions as the sister (torch CPU, transformers,
   sentence-transformers, librosa, pillow, torchvision, truststore, scikit-learn, plotly, pyarrow). `pip install -e .`
   provides the `vocab` CLI.
6. **Record provenance** in `IMPORTED.md`: every copied file, its source path, and the sister repo's commit hash.
7. **Parity check:** embed 10 of the sister's podcast clips and 5 text phrases with the copied code. The vectors must
   match the sister's stored `index/audio_emb.npy` rows within float tolerance. If they don't, stop.
8. `git init`, a `.gitignore` (`data/raw/`, `.venv/`, caches, `.env`), and a first commit.

**Don't:** download any datasets yet.

### CHECKPOINT 0
Show:
- The sister path and the files copied (from `IMPORTED.md`).
- Model paths in config.
- The parity-check result.
- Embedding time for one 20 s clip, for Gemma and for CLAP.

---

## Phase 1: Read the model (how audio and text are actually handled)

**Goal:** understand what EmbeddingGemma 2 does to a clip and to a sentence before measuring anything. The sister
project used it as a black box.

1. Read the model card, `config.json`, and the processor and modeling code in the local install. Write
   `MODEL_NOTES.md` covering:
   - **Audio path:** expected sample rate, how audio becomes features (e.g. mel frames), the **maximum audio length**
     and what happens beyond it (truncation? chunking?), roughly how many audio tokens per second, and how the final
     vector is pooled.
   - **Text path:** tokenisation, any **task prefixes or prompt templates** for queries vs documents, the maximum
     length, and pooling.
   - **The shared space:** how audio and text are brought together (shared backbone, projection heads, training
     objective), as far as the docs say. Mark anything undocumented as "not stated".
   - The same, more briefly, for CLAP.
   - Every claim needs a source (file and line, or a model card section).
2. **Cross-check the sister project (read-only).** Were its 20 s clips within the max length, or silently truncated?
   Were queries embedded with the right prefix? Write the answer to `reports/vibe_search_crosscheck.md`. **Don't
   touch the sister repo;** the user will decide whether to fix it there.
3. **Small behaviour tests** on about 30 of the sister's podcast clips (read by path):
   - **Length:** the same clip at 2, 5, 10, 20 and 30 s. How fast does the vector stabilise?
   - **Loudness:** −12, −6, 0 and +6 dB. Is the embedding loudness-invariant?
   - **Silence and noise:** pure silence, white noise and pink noise. Where do they land? Is silence a hub that
     everything is close to?
   - **Text wording:** with and without the task prefix; single words vs short phrases vs long descriptions. How much
     does wording move the vector?
4. Measure **embedding time per second of audio** for 1–10 s clips. This sizes Phase 2.

### CHECKPOINT 1
Show:
- `MODEL_NOTES.md`.
- The Vibe Search cross-check.
- The behaviour-test tables.
- Embedding cost per second of audio, for both models.

---

## Phase 2: Datasets, licences and the download plan

**Goal:** pick labelled, licence-clear datasets that cover the sound families, include fine sibling classes (the
"giggle vs chuckle" and "cuckoo vs owl" kind), and give **instrumental music with genre, region and instrument
labels**.

**Candidates.** Verify each one's licence, size, label vocabulary and host reachability. Don't assume.

| Family | Candidate | Why | Licence (expected, verify) |
|---|---|---|---|
| General sound events | **ESC-50** | 50 classes × 40 clips, 5 s, 5 provided folds; animals, nature, human, domestic, urban | CC BY-NC 3.0 |
| Fine sound classes | **FSD50K** | AudioSet-ontology labels with hierarchy; check whether it includes laughter subtypes (e.g. giggle, chuckle/chortle) and other sibling sets | Per-clip CC (mixed); labels CC BY |
| Human non-speech vocal sounds | **VocalSound** | Laughter, sigh, cough, throat clearing, sneeze, sniff, from many speakers | Check |
| Animals, fine-grained | **xeno-canto** (API) | Species-level bird calls, so cuckoo vs owl and near relatives can be tested directly | Per-recording CC; **exclude ND, keep NC internal** |
| Speech emotion | **CREMA-D** | The same sentences spoken by many actors in 6 emotions, so the words are held fixed | ODbL (check) |
| Speech emotion (backup) | RAVDESS | 8 emotions, 2 intensities | CC BY-NC-SA (check) |
| Music: genre + region | **FMA** (`fma_medium` or `fma_large` metadata + a subset of audio) | Genre tree with top genres (check that Classical, Jazz, Folk, Electronic, Blues, Country… exist) and "International" subgenres that may give regions; artist ids for grouped splits | Per-track CC; metadata CC BY |
| Music: instruments + genre + mood | **MTG-Jamendo** | Genre, instrument and mood tags on the same tracks, so genre can be tested *with instruments held fixed* (Phase 8) | Per-track CC (check) |
| Music: real-world classical | **Musopen** or another public-domain classical source | FMA's "classical" is mostly indie/amateur; a canonical classical set checks that the result isn't an FMA quirk | Public domain / CC (check per recording) |
| Music: free-text descriptions (optional) | **Song Describer Dataset** | Human-written captions of MTG-Jamendo tracks, so full sentences (not just genre words) can be tested | CC BY-SA (check) |

**Instrumental filter for music** (needed for Q5):
- Use metadata first: genre or tag information saying "instrumental", or instrument tags without voice.
- Then a **vocal check** using the faster-whisper small model from the sister's `models/` folder. Flag a clip as
  having vocals if Whisper transcribes more than a few confident words. Report how many clips each step removes.
- Keep a separate set of **vocal tracks in the same genres.** Phase 7 compares the two, to show how much lyrics help
  the genre match.
- Calibrate the Whisper threshold on a held-out set of clips whose tags say vocal vs instrumental, not on the
  analysis set.

**Sizing:**
- Sound sets: about **4,000–5,000 short clips** (1–5 s), about 30–50 per class, more for sibling sets.
- Music: about **2,000–3,000 tracks**, with ≥ 50 per genre and ≥ 30 per region. Use 1–3 fixed windows per track,
  each within the model's max length from Phase 1.
- Use Phase 1's timing to make each embedding run fit overnight. If it doesn't, cut classes or genres, not clips per
  class.

**Taxonomy** (`data/taxonomy.yaml`):
- Map every class to a **family** (human vocal, speech emotion, animal, nature, machine/domestic, music).
- Map classes to a **sibling group** where one exists (laughter types, owl species, rain vs wind vs water…).
- For music, map each genre tag to a **genre word** and each regional subgenre to a **region word**.
- Drop labels that aren't real genre words (e.g. "Instrumental", "Experimental", "Spoken") from the genre-word tests,
  with a note.

**Download plan:**
- Check which hosts are reachable from the work laptop.
- For anything blocked, write a bundle script for the personal laptop.
- Write `data/LICENCES.md`: every dataset's licence, attribution text and what that means for the page.

**Don't:** download anything until the plan is approved.

### CHECKPOINT 2
Show:
- The chosen datasets with licences, reachable or blocked, and sizes.
- Class, genre and region lists with counts (genres and regions after the instrumental filter, estimated).
- Total clip count and estimated embedding time.
- The download plan.

---

## Phase 3: Embed and first look

1. Download (or import the bundle), verify, cut clips per the plan, and run the instrumental filter.
2. Write `data/clips.parquet`: clip id, dataset, class/genre/region, family, sibling group, source recording, speaker
   or artist id, fold or split, instrumental flag, instrument tags, licence, window start and end.
3. Embed every clip with **Gemma** and **CLAP**, L2-normalised, into `index/gemma.npy` and `index/clap.npy`.
4. Embed the **text side**, into `index/text_gemma.npy` and `index/text_clap.npy`:
   - every class name, plus 2–3 short natural descriptions per class ("a person giggling", "an owl hooting at night");
   - every genre and region word in three forms: bare ("jazz"), short phrase ("jazz music"), and a neutral sentence
     ("a piece of jazz music");
   - use the prefix rules from `MODEL_NOTES.md`.
5. Make a UMAP of all clips coloured by family, and a joint audio and text UMAP to show the modality gap on clean data.

**Sanity checks** (stop if they fail):
- Clips from the same ESC-50 class should be closer to each other than to random clips.
- A linear probe on ESC-50 using the official folds should score well above chance.
- The text "dog barking" should retrieve dog clips more often than chance.
- The music probe for genre, with artist-grouped splits, should score above chance.

### CHECKPOINT 3
Show:
- Clip and embedding counts per dataset and model.
- Instrumental-filter numbers.
- Embedding time.
- The two UMAPs.
- Sanity-check results.

---

## Phase 4: Resolution (Q1) and knows vs can name (Q2)

**Goal:** measure where the audio space is fine and where it's coarse. This is Anand's question.

Per class, family and sibling group, for both models, using **grouped splits** throughout:

| Measure | What it says |
|---|---|
| **Tightness** = mean within-class cosine − mean cosine to the nearest other class | How compact a sound's cluster is |
| **kNN accuracy** (k = 10, leave-one-source-out) | Can similar sounds find each other? |
| **Sibling accuracy:** probe or kNN *within* a sibling group only | Fine resolution: giggle vs chuckle vs belly laugh; cuckoo vs owl species |
| **Linear probe accuracy** (logistic regression, fixed regularisation) | What the audio embedding *contains* |
| **Zero-shot text accuracy:** label each clip by its nearest class text, raw and centred | What a *word* can find |
| **Probe − zero-shot gap** | "Knows but can't name it" |

- **Compare families:** is resolution higher for human vocal sounds than for animals? Within animals, is it higher
  for common sounds (dog, cat) than for bird species?
- **Effective vocabulary size:** per family, how many classes are told apart above a stated threshold (e.g. 70%
  pairwise)? Report it as "the space reliably distinguishes about N kinds of X", with the threshold shown.
- **Don't** change sibling groups after seeing results. Log any change with its reason in `reports/decisions_log.md`.

### CHECKPOINT 4
Show:
- The family resolution table (Gemma vs CLAP) with CIs.
- Sibling-group results, highlighting laughter types and bird species.
- The probe vs zero-shot gap chart.
- A one-paragraph plain-English answer to Anand's question.

---

## Phase 5: Words vs sounds, and the gap (Q3)

1. **Synonym clouds** (Anand's love/hate example, done for sounds):
   - For about 15 concepts (laugh, cry, scream, whisper, rain, wind, dog, bird, engine, applause, music…), list about
     10–12 synonyms or variants each.
   - Use WordNet (`nltk`; ask before adding) or a fixed list in config. LLM drafts are allowed only after review.
   - **Measure:** do the synonyms cluster tightly in text space? Do they retrieve the *same* audio clips (overlap of
     their top-25)? Report where text is finer than audio, and where audio is finer than text.
2. **Structure match:** correlate a class×class similarity matrix from audio centroids with one from class texts
   (Spearman, with a permutation test), overall and per family.
3. **Modality gap per family:** text↔audio cosine for matching vs non-matching pairs.
4. **Centring strength sweep:**
   - Subtract α × the means, for α ∈ {0, 0.25, 0.5, 0.75, 1}. Report zero-shot accuracy per family.
   - Tune α on one split and report on the other.
   - Note the limit: this measures sound-event retrieval. The sister project showed that *vibe* retrieval can get
     worse even when category retrieval improves.

### CHECKPOINT 5
Show:
- The synonym-cloud table.
- Structure-match correlations.
- Gap per family.
- The α sweep chart.
- One paragraph on what it means.

---

## Phase 6: How it's said vs what's said (Q4)

Use CREMA-D (or RAVDESS), where the same sentences are spoken in different emotions.
1. **Probes,** with speaker-grouped splits, for **emotion**, **speaker identity** and **sentence identity**. Which does
   the space prioritise: who, what, or how?
2. **Zero-shot:** can "an angry voice", "a happy voice" or "a sad voice" find the right emotion? Raw and centred.
3. **Baseline:** simple acoustic features (energy, pitch spread, pace, as in the sister project) in the same probe.
   If loudness and pitch separate anger as well as the embedding does, say so.
4. **Tie back to the "heated argument" failure:** is there no hostility signal (probe fails), or can words not reach
   it (probe works, zero-shot fails)?

### CHECKPOINT 6
Show:
- The probe table (emotion / speaker / sentence × Gemma / CLAP / acoustic baseline).
- Zero-shot emotion accuracy.
- The confusion matrix.
- A one-line verdict.

---

## Phase 7: Music without lyrics → genre and region words (Q5, part 1)

**Goal:** can a genre or region *word* find instrumental music of that genre or region?

All on the **instrumental** set, with **artist-grouped** splits:
1. **Probe** accuracy over genres, and over regions. This is what the audio contains.
2. **Zero-shot** accuracy over genres and regions, using each word form (bare, phrase, sentence), raw and centred.
   Per genre: precision of the top-25 clips nearest each genre word, against the base rate.
3. **Probe − zero-shot gap per genre:** which genres does the model "know but can't name"?
4. **Confusions:** which genres blur together (folk ↔ country? electronic ↔ experimental?), and do the blurs match
   music-theory intuition or look arbitrary?
5. **Lyrics effect:** the same zero-shot test on the **vocal** tracks of the same genres. How much do lyrics or
   vocals add?
6. **Canonical check:** run the classical zero-shot on the public-domain classical set too. Does "classical" find
   canonical classical recordings as well as FMA's indie "classical"?
7. **Region narrowing** (Anand's background-music idea): how often is the true region in the top 3? That's what would
   let an agent filter a catalogue before an exact match. This is region recognition, not song identification.

**Decide which genres go to Phase 8:**
- A genre qualifies if its zero-shot precision is clearly above base rate (CI excludes it) and above a stated bar, on
  the held-out split.
- Expect classical to qualify. Also pick **one genre that fails,** as a contrast.
- Fix the list here, and don't add to it later.

### CHECKPOINT 7
Show:
- Genre and region tables (probe vs zero-shot, Gemma vs CLAP, word forms, raw vs centred).
- The confusion matrix.
- The lyrics effect.
- The canonical classical check.
- Region top-3.
- The Phase 8 genre list with reasons.

**If no genre qualifies,** stop and discuss before Phase 8.

---

## Phase 8: What makes classical "classical"? (Q5, part 2)

**Goal:** for each genre picked in Phase 7, find out what the model is responding to when it ties music to that
word. Four methods, all automatic, so the findings can be cross-checked against each other.

**Genre score** = cosine of a clip's audio embedding with the genre word's text embedding (the best word form from
Phase 7), minus its mean cosine with the other genre words. This is the model's "how classical is this?".

1. **Measured features vs genre score.**
   - Compute plain music features per clip with librosa:
     - tempo and tempo stability;
     - onset rate (note density);
     - percussive vs harmonic energy (HPSS);
     - spectral centroid (brightness) and bandwidth;
     - dynamic range;
     - key clarity (chroma);
     - harmony change rate;
     - an MFCC timbre summary.
   - Correlate each feature with the genre score **within the genre and across genres.**
   - Fit a small regression (genre score ~ standardised features). Which few features explain it?
2. **Perturbations (the causal test).** Take about 150 instrumental clips (genre members and non-members), change one
   thing at a time, and re-embed:
   - **remove percussion:** keep only the harmonic part;
   - **remove harmony:** keep only the percussive part;
   - **tempo:** time-stretch ×0.75 and ×1.33, without changing pitch;
   - **pitch shift:** ±3 semitones;
   - **timbre:** low-pass and high-pass filters (darker or brighter);
   - **dynamics:** heavy compression (flattened dynamics);
   - **structure:** shuffle 0.5 s chunks (keeps timbre, destroys melody and phrasing), and reverse the audio.

   How much does each change move the genre score? If shuffling barely moves "classical" but removing the harmonic
   part kills it, the model is hearing **timbre and instruments, not composition**. Make every perturbation
   deterministic, and record the parameters.
3. **Instruments held fixed** (MTG-Jamendo tags):
   - Compare tracks with the *same* main instrument across genres, for example piano in classical vs piano in jazz vs
     piano in pop/ambient.
   - Does the classical score still separate them? If yes, "classical" is more than "piano and strings". If no, the
     word mostly means the instruments.
4. **Descriptor words:**
   - Embed about 40 descriptive words, in fixed lists in config: instruments ("strings", "harpsichord", "orchestra"),
     qualities ("ornate", "slow", "dramatic", "acoustic", "ancient", "elegant"), contexts ("film score", "church",
     "concert hall").
   - Rank them by closeness to (a) the genre word in text space and (b) the genre's audio centroid.
   - Do the text side and the audio side agree on what "classical" means?
5. **False positives:** list the non-classical tracks with the highest classical scores (e.g. ambient piano, film
   score, lo-fi), with their tags and features. These boundary cases show what the model thinks "classical" is.

Do 1–5 for each Phase 8 genre, plus the failing contrast genre. Methods 1–2 should show *why* it fails: no consistent
feature signature, or a signature the text side doesn't share.

### CHECKPOINT 8
Show, per genre:
- The feature-correlation chart.
- The perturbation chart (change in genre score per perturbation, with CIs).
- The instrument-held-fixed result.
- The descriptor-word ranking (text side vs audio side).
- The top false positives.
- A one-paragraph answer: "to this model, 'classical' mostly means…".

---

## Phase 9: Findings page, and the bridge back to Vibe Search

**Goal:** one page Anand can read in two minutes, with drill-down for the curious.

1. **Bridge to Vibe Search** (`reports/vibe_search_bridge.json`):
   - Map the sister project's listening-test searches onto these findings, reading its `reports/listening_test.json`
     by path.
   - Do the failures (heated argument, rain/nature, crowd/music) line up with weak areas of the space, or with
     missing data in the podcast corpus?
2. **Page** (`docs/index.html`, plus a detail page if needed), following `CLAUDE.md`:
   - Headline answers first, about 4 bullets, then one self-explaining chart per question.
   - **What the space can hear:** resolution by family (Gemma vs CLAP); laughter types and bird species.
   - **Knows vs can name:** the probe vs zero-shot gap.
   - **Words vs sounds:** synonym-cloud explorer (pick a concept, see its synonyms in text space and the clips they
     retrieve), plus the gap per family.
   - **Emotion:** who / what / how.
   - **Music without lyrics:** genre and region results, then the "what makes classical classical" section
     (perturbation chart, instrument-held-fixed result, descriptor words, false positives).
   - **Back to Vibe Search:** the bridge, with a link to the sister project's page.
   - **Limits:**
     - two models only;
     - dataset label noise;
     - FMA is indie/amateur music;
     - the instrumental filter is imperfect;
     - genre words are culturally loaded;
     - correlation isn't cause (the perturbations are the causal part);
     - English words only.
   - Audio playback only where the licence allows it. Otherwise show labels, not sound.
   - Every number comes from `reports/*.json`. None are typed by hand.
   - Footer: dataset attributions from `LICENCES.md`, model credits, and "internal prototype".
3. Write `PROJECT_STATUS.md` in the same style as the sister project's: what was built, how it diverged from this
   README, findings, environment notes and open items.

### CHECKPOINT 9
Show:
- Page screenshots.
- The bridge table.
- A grep showing no hard-coded numbers.
- Attributions and limits visible.

---

## Repo layout

```
audio-vocab/
  README.md  MODEL_NOTES.md  ENVIRONMENT.md  IMPORTED.md  CLAUDE.md  PROJECT_STATUS.md  config.yaml
  data/  taxonomy.yaml  LICENCES.md  clips.parquet  raw/ (gitignored)
  index/ gemma.npy  clap.npy  text_gemma.npy  text_clap.npy  perturb/
  scripts/ fetch_bundle.py  import_bundle.py
  src/audiovocab/
    embed.py  cache.py  centering.py  llm.py     # copied + adapted from podcast-vibe-search
    modelnotes.py # Phase 1 behaviour tests
    datasets.py   # fetch/import, cut clips, instrumental filter, clips.parquet
    splits.py     # official folds / grouped splits
    resolution.py # tightness, kNN, sibling accuracy, probes, zero-shot
    words.py      # synonym clouds, structure match, gap per family, α sweep
    emotion.py    # emotion vs speaker vs sentence
    genre.py      # Phase 7 genre/region words
    whatmakes.py  # Phase 8 features, perturbations, instrument control, descriptors
    bridge.py     # Vibe Search bridge
    page.py       # reports/*.json → docs data
    cli.py
  reports/  docs/  tests/  runs/
```

CLI: `vocab <step>`, for example `vocab setup`, `vocab notes`, `vocab plan`, `vocab fetch`, `vocab embed`,
`vocab resolution`, `vocab words`, `vocab emotion`, `vocab genre`, `vocab whatmakes --genre classical`,
`vocab page`.

## Out of scope (v1)
- Exact song identification or fingerprinting (the Shazam problem).
- Phonetic/transliteration search for Hindi song titles (a separate project).
- Fine-tuning, or training anything beyond linear probes.
- Source separation to strip vocals from songs (ask first if it becomes important).
- YouTube, AudioSet or MusicCaps downloads.
- Any change to `podcast-vibe-search`.