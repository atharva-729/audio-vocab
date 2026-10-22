# Decisions log

Every choice that deviates from the README, or that fixes something before results are seen, with its reason.
Entries are append-only.

## 2026-10-22 (Phases 0–2, run without checkpoint stops at the user's request)

1. **Timelines removed from the README** at the user's request ("timelines do not matter").
2. **Phases 0–2 run straight through**, without stopping at Checkpoints 0–2 (user's explicit permission). The
   checkpoint material is still produced; the stop before Phase 3 (first large downloads) is kept.
3. **CLAP parity rows.** The sister's Phase 1 CLAP cache was built on its original segment set (10 s hop,
   minutes 0–15), which it later replaced. Parity uses the 5 cache rows whose Gemma twins exactly match a current
   `audio_emb.npy` row. Result: exact match (max |diff| 0.0).
4. **Centring means passed explicitly** (`center(x, mean, strength)`) instead of one global `centering.npz`, so each
   analysis computes its audio mean on its tuning split (README: don't tune on reported data).
5. **New family `human_activity`** for non-vocal human sounds (clapping, footsteps, brushing teeth, chewing...),
   which fit neither "human vocal" nor "machine/domestic". All other classes use the README's families.
6. **FSD50K: eval split only.** It has ≥ 25 single-class clips for 62 of the 63 chosen classes (Crow drops out), and
   it's 6.3 GB versus 24 GB for dev over a slow (~0.5 MB/s) Zenodo link. FSD50K labels propagate up the AudioSet
   hierarchy, so a clip is assigned to a class only when it carries no rival label from the same sibling group,
   and parent labels (Laughter, Water, Vehicle...) aren't used as classes.
7. **Laughter subtypes: only two exist** in licence-clear labelled data: FSD50K `Giggle` and `Chuckle_and_chortle`.
   There's no belly laugh, snicker or baby-laughter label (AudioSet's "Baby laughter" and "Belly laugh" aren't in
   FSD50K's 200-class vocabulary). The "50 kinds of laughter" question can be tested only as giggle vs chuckle,
   plus laughter vs other vocal sounds (VocalSound, ESC-50).
8. **Bird species from xeno-canto: 17 species**, chosen before any embedding:
   - the cuckoo;
   - 6 owls;
   - other "hooting" callers that could be confused with cuckoo or owl (collared dove, wood pigeon, hoopoe);
   - 5 songbirds and 2 corvids as contrasts.

   For each species: 40 recordings, quality A first then B, 5–60 s, at most 2 per recordist, ND licences excluded.
9. **Region words: International subgenres with ≥ 20 artists.** Reggae (Dub, Dancehall) is excluded as a genre, not
   a region. The most specific subgenre wins (Spanish over Europe; Afrobeat counts as African).
10. **FMA genre membership:** a track belongs to a genre word only if its `genres_all` contains exactly one of the
    11 genre top-levels (after dropping Instrumental, Experimental, International, Spoken, Old-Time).
11. **FMA's "Instrumental" tag is not an instrumental filter for regions.** Only 0–20 tracks per region carry it,
    so every region track goes through the Whisper vocal check. Echo Nest `instrumentalness` was considered and
    rejected as unreliable: it calls 86% of "French" tracks instrumental, which is implausible.
12. **FMA audio via ranged reads of `fma_large.zip`.** HTTP range requests extract only the selected 30 s clips
    (tested: listing 9 s, about 2.6 s per 1.2 MB clip), instead of downloading the 100 GB archive.
13. **MTG-Jamendo: shards 53 and 86 only** (about 3.3 GB, 1,135 tracks), chosen to maximise artists in the
    piano × {classical, jazz, ambient, pop} cells used by Phase 8. Each cell has ≥ 25 tracks from ≥ 20 artists.
    Guitar and strings cells are too thin (≤ 11 tracks) and are not tested.
14. **NoDerivatives licences:** ND tracks/clips are used for embedding only (no redistribution, no altered copies
    shared). They're excluded from Phase 8 perturbations, which create altered audio. xeno-canto ND recordings are
    excluded entirely, per the README.
15. **RAVDESS and Song Describer not used:** RAVDESS was a backup for CREMA-D (which is reachable); Song Describer is
    optional in the README.
16. **Musopen via archive.org** (Public Domain Mark), because musopen.org sits behind a Cloudflare bot check.
    At most 4 movements per work, so the 32-movement Goldberg Variations doesn't dominate. That gives 116 movements
    from 37 works.

## 2026-10-22 (Phase 4, after the first Gemma run)

17. **Probes standardise features (fit on training folds) before logistic regression.** The first run used raw
    L2-normalised vectors with fixed C = 1. For Gemma the probe came out *below* kNN on ESC-50 (0.71 vs 0.77), a
    sign of underfitting rather than missing information: Gemma's vectors sit in a narrow cone (mean per-dimension
    std 0.015), so C = 1 is effectively a very strong penalty. CLAP's spread-out vectors were barely affected.
    Standardising is a scale correction applied identically to both models and every probe; C stays fixed at 1 and
    nothing was tuned on reported folds. Check on two ESC-50 folds: Gemma 0.70 → 0.83.

18. **Whisper on MTG-Jamendo only for the 186 piano-tagged tracks** (the only MTG tracks Phase 8 uses), instead of
    all 1,135. MTG's missing voice tag can't stand in for "instrumental": only 31 of 1,135 tracks carry a voice/vocals
    tag, and 563 carry no instrument tag at all, so tags are present-when-noticed, not complete. Whisper order:
    calibration set first (it fixes the threshold), then FMA, then the MTG piano tracks.

19. **Vocal detection counts all transcribed words, as chosen by the calibration set.** The pre-set rule (count only
    words in confident segments) caught only 37% of the held-out vocal-tagged tracks at its best threshold. The
    calibration step now chooses the count (confident-only or all words) as well as T, by balanced accuracy on the
    200 held-out calibration tracks, which are never analysed. It chose all words with T = 1: vocal recall 0.61,
    instrumental recall 0.92. Several "vocal"-tagged calibration tracks have no words at all in the 20 s window
    (likely instrumental passages), so true vocal recall is probably higher. Residual vocals are a stated limit.
20. **Phase 7/8 rerun.** The first Phase 7/8 run started before Whisper had finished: the chain's process check fell
    in the seconds between stopping and restarting Whisper (decision 18). Its results were discarded and rerun on
    the complete Whisper output.
21. **Probe regularisation chosen by nested cross-validation.** A single fixed C (1.0) gave contradictory results on
    the small music sets: Gemma's genre probe (26%) scored below zero-shot (27%), and classical recall fell from
    0.71 zero-shot to 0.29 by probe. On the tuning fold, C = 0.001 gave 0.39 vs 0.28 at C = 1, so the fixed C was
    overfitting 768 standardised dimensions on ~150 tracks per genre. Each probe now picks C from
    {0.001, 0.01, 0.1, 1} per outer fold, by leave-one-fold-out over up to 3 of its training folds. Reported folds
    never influence the choice; the chosen C per fold is reproducible. Pairwise probes (thousands of binary
    problems for the effective vocabulary) use a fixed C = 0.01. Phases 3, 4, 6 and 7 were rerun with this.
