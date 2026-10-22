# Audio Vocab: where the project actually went

Context handoff for anyone (or any AI chat) picking this up. `README.md` is the **original build plan**; this file
describes **what was built, how it diverged, what was found, and what's open**. Last updated 2026-10-22.

**For a follow-on project, read CONTEXT.md:** the full briefing, with every finding table, per-class detail,
method lessons, reusable artefacts, trust levels and practical guidance (regenerate it with
`python scripts/make_context.py` after any rerun).

## 1. In one paragraph

Audio Vocab studies the shared audio–text embedding space of **EmbeddingGemma 2** (with **LAION-CLAP** as a second
model) on labelled data, instead of podcasts. It asks:
- where the audio space is fine or coarse;
- whether words can reach what the audio contains;
- whether text and sound vocabularies share a shape;
- whether emotion is encoded;
- what a genre word like "classical" means to the model.

All phases 0–9 are built. The results page is `docs/index.html`; per-phase summaries are in `CHECKPOINTS.md`, and
every number lives in `reports/*.json`.

## 2. Headline findings (in these datasets, with these models)

1. **Gemma's audio space is good, but its words barely reach it for sound events.**
   - ESC-50: probe 0.83 vs zero-shot 0.25 (CLAP: 0.97 vs 0.90).
   - Gemma's clip-to-own-class-text margin is about 0.03 cosine, against about 0.3–0.5 for CLAP.
   - This matches the sister finding that Gemma is strongest on speech, not sound events.
2. **Giggle vs chuckle is told apart only coarsely** (0.69 for both models, chance 0.50).
   - Cuckoo vs owls is separated at least as well relative to chance.
   - So the space is not tuned to human expression over wildlife. But words reach laughter (90–97%) far better
     than bird species (near chance).
3. **Emotion is encoded but unnameable.** Anger probe recall is 0.74–0.75; "an angry voice" finds 0–2%. Gemma
   encodes the sentence perfectly (1.00) and the speaker barely (0.06); CLAP keeps the speaker (0.48).
4. **"Classical" means tonal, drum-free texture.**
   - It is the most transferable genre word: 99–100% of Musopen's canonical recordings are labelled classical.
   - Keeping only the drums erases the classical signal.
   - Tempo, pitch and dynamics barely matter.
   - It's more than "piano" (classical piano vs other piano: AUC 0.75).
5. **Bridge to Vibe Search:** per-class resolution on labelled data does *not* predict listening success. Six of
   the "weak area" searches still scored 3–4 of 4 by ear, because Vibe Search succeeds on speaking style and
   show-level sound.

## 3. How it diverged from the README plan

| README plan | What happened | Why |
|---|---|---|
| Stop at every checkpoint | Phases 0–2 ran straight through, then everything else; checkpoints written to `CHECKPOINTS.md` | User's instruction |
| Timelines in the README | Removed | User's instruction ("we move fast") |
| Download FSD50K from Zenodo | Zenodo blocked the work network mid-download ("unusual traffic"). FSD50K was embedded in **Colab** (`colab/fsd50k_embed.ipynb`) and imported (`fetch.import_colab_fsd`); parity 1.0000 and weight SHA-256 checked | Proxy/rate limit |
| VocalSound, MTG-Jamendo direct | Personal-laptop bundle (`scripts/fetch_bundle.py`), checksums verified | Dropbox and Freesound CDN blocked |
| FMA `fma_medium` download | Only the selected 30 s clips, read from the 100 GB `fma_large.zip` with HTTP range requests (`remotezip.py`) | Size; Phase 2 needed regional tracks beyond `fma_medium` |
| Fixed-C probes | Standardised features; C chosen per fold by cross-validation inside training folds (decisions 17, 21) | Fixed C underfit Gemma's narrow-cone vectors and overfit small music sets; caught by probe < kNN and probe < zero-shot |
| Whisper vocal check, pre-set confidence rule | The calibration set chooses the count and threshold (decision 19) | The pre-set rule caught only 37% of held-out vocal tracks |
| Whisper on all MTG tracks | Only the 186 piano tracks Phase 8 uses (decision 18) | MTG's missing voice tags aren't evidence of instrumental |
| 1–3 windows per music track | 1 window (10–20 s) for FMA/MTG, 2 for Musopen | CPU time |
| Family list | Added `human_activity` (clapping, footsteps…) | Fits neither "human vocal" nor "machine" |
| Laughter: "50 kinds" | Only giggle and chuckle exist as labels in licence-clear data | Data availability |

All decisions with reasons: `reports/decisions_log.md` (21 entries).

## 4. Environment and gotchas

See `ENVIRONMENT.md`. In short:
- Windows 11 laptop with an **Intel Core 5 120U (15 W)**: Gemma takes about 2–4 s per 10 s clip.
- Python 3.14; models are read from the sister repo by absolute path.
- **Proxy blocks:** Dropbox, Freesound CDN, Jamendo, the HF CDN, and (since 2026-10-22) **large Zenodo downloads**.
- Long jobs run detached. Laptop sleep pauses them, but they resume. Whisper and embedding are resumable.
- pandas 3 string dtype breaks numpy arrays (`data.clips()` converts to object).
- Python's `zipfile` can't read multi-disk zip64 archives: join the parts with `zip -s 0` first.

## 5. Pipeline

```
vocab plan       metadata → data/selection/*.parquet, reports/phase2_plan.json
vocab fetch X    esc50 | cremad | xenocanto | fma | musopen  (fsd50k: Colab notebook + import)
vocab clips      → data/clips.parquet (windows per dataset; see datasets.py)
vocab embed      → index/{gemma,clap}.npy (cached per clip in runs/cache/)
python -c "from audiovocab import texts; texts.run()"   text side → index/text*.npy
python -c "from audiovocab import instrumental; instrumental.transcribe_all()"   Whisper
vocab analyse    resolution emotion words phase3 instrumental genre whatmakes bridge → reports/*.json
vocab page       → docs/data.js (the page reads only this)
```

## 6. Open items

- **GitHub Pages:** enable Pages for `/docs` on the repo, or serve locally. No audio is hosted.
- **Perturbation sets are small** (16 tracks per genre). A larger Phase 8 run would tighten the intervals; it costs
  about 1,650 Gemma embeddings per 150 clips.
- **Instrumental filter:** residual vocals remain (vocal recall 0.61 on noisy calibration labels). A dedicated
  vocal detector, or source separation (out of scope in v1), would sharpen the lyrics-effect and region results.
- **The effective-vocabulary measure hits its ceiling** at 70%. A stricter threshold would be more informative, but
  it must be fixed before looking.
- **Flag to Anand:** ESC-50 is CC BY-NC and MTG-Jamendo is research-only. That's fine for this internal study, but
  check before any client-facing use.
- **Sister repo:** nothing was changed there. The cross-check (`reports/vibe_search_crosscheck.md`) found no
  problems.
