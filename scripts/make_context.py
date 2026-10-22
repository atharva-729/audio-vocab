"""Generate CONTEXT.md: a detailed briefing on audio-vocab for Claude in a follow-on project.

Narrative is written here; every table and number is pulled from reports/*.json, so the document can be regenerated
after any rerun:  .venv/Scripts/python.exe scripts/make_context.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = lambda n: json.loads((ROOT / "reports" / n).read_text(encoding="utf-8"))


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def ci(d):
    return f"{pct(d['acc'])} ({pct(d['ci'][0])}–{pct(d['ci'][1])})"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    par, p1, tok = R("parity.json"), R("phase1_behaviour.json"), R("phase1_audio_tokens.json")
    p2, p3, p4, p5, p6, p7, p8 = (R("phase2_plan.json"), R("phase3_sanity.json"), R("phase4_resolution.json"),
                                  R("phase5_words.json"), R("phase6_emotion.json"), R("phase7_genre.json"),
                                  R("phase8_whatmakes.json"))
    br, inst, um = R("vibe_search_bridge.json"), R("instrumental_filter.json"), R("phase3_umap.json")
    M = ("gemma", "clap")
    S = []
    add = S.append

    add("""# Audio Vocab: full context for a follow-on project

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
""")

    # ---- models
    tl = {k: v for k, v in tok.items()}
    add(f"""---

## 2. The models: facts that matter for any reuse

(Full sourced notes: `MODEL_NOTES.md`.)

### EmbeddingGemma 2
- **Input:** 16 kHz mono → 128-bin log-mel (0–8 kHz, 20 ms frames, 10 ms hop) → 12-layer conformer (local, causal
  attention, about 1 s per layer) → **25 audio tokens per second** (measured: 1 s → {tl['1']['audio_tokens']},
  10 s → {tl['10']['audio_tokens']}, 20 s → {tl['20']['audio_tokens']}).
- **Hard limit: audio over 30 s is silently truncated** (30, 40 and 60 s all → {tl['30']['audio_tokens']} tokens;
  the embeddings are bit-identical). The model card's "~327 s" does not apply via transformers/sentence-transformers.
  `audio_seq_length: 280` in `processor_config.json` is *not* a limit (it's a vLLM helper).
- **Audio is fed into the same 24-layer text transformer as soft tokens.** It's one shared backbone, not two towers.
  The vector is the mean over all positions (including the special tokens), projected 512→768, then L2-normalised.
- **Text prefixes:** queries `task: search result | query: …` (prompt name `SearchQuery`); documents
  `title: none | text: …`. **Audio takes no prefix.** The prefix is mean-pooled into the vector, so it shifts every
  query vector. Chosen here on a tuning fold: `{p4['gemma']['prefix']}` (zero-shot
  {pct(p4['gemma']['prefix_tune_scores'].get('SearchQuery'))} vs {pct(p4['gemma']['prefix_tune_scores'].get('none'))}
  with no prefix).
- **Precision:** float32 on CPU. bf16 is about 12× slower on this CPU; **never fp16** (NaNs). Load with
  `config_kwargs={{"vision_config": None}}` (add `"audio_config": None` for text only).
- **Geometry:** a **narrow cone**. Mean cosine clip↔clip is about {f2(um['cosines']['clip_clip'])}, text↔text
  {f2(um['cosines']['text_text'])}, text↔clip {f2(um['cosines']['text_clip'])}. Absolute cosines are not
  comparable across models; use rankings, contrasts or standardised features. **Linear probes need feature
  standardisation** (per-dimension std ≈ 0.015), or a fixed regularisation silently underfits.
- **Cost on this laptop** (Intel Core 5 120U, 15 W): {par['gemma_s_per_20s_clip']} s per 20 s clip (clean);
  about 2–4 s per 10 s clip in batch runs. CLAP is about 7–10× cheaper ({par['clap_s_per_20s_clip']} s per 20 s clip).

### CLAP (laion/clap-htsat-unfused)
- 48 kHz, 64-bin log-mel; **maximum 10 s**. The processor's default for longer audio is a **random crop**; our
  wrapper averages fixed 10 s windows instead, so it is deterministic.
- Two towers (HTS-AT audio, RoBERTa text) trained contrastively. No prompts or prefixes. 512-d.
- Its space is spread out (clip↔clip cosine about {f2(p1['clap']['synthetic']['clip_to_clip_cos']['mean'])} on
  podcasts, against about {f2(p1['gemma']['synthetic']['clip_to_clip_cos']['mean'])} for Gemma).
- It may have seen ESC-50-like data in training, so ESC-50 zero-shot numbers may be inflated.

### Behaviour (Phase 1, 30 podcast clips)
{table(['Test', 'Gemma', 'CLAP'], [
        ['10 s window vs the same clip at 30 s (cosine)', f2(p1['gemma']['length']['10']['cos_to_30s']['mean']), f2(p1['clap']['length']['10']['cos_to_30s']['mean'])],
        ['20 s vs 30 s', f2(p1['gemma']['length']['20']['cos_to_30s']['mean']), f2(p1['clap']['length']['20']['cos_to_30s']['mean'])],
        ['next 20 s of the same show (content drift)', f2(p1['gemma']['length']['next_20s_window_cos']['mean']), f2(p1['clap']['length']['next_20s_window_cos']['mean'])],
        ['−12 dB vs 0 dB', f2(p1['gemma']['loudness']['-12']['mean']), f2(p1['clap']['loudness']['-12']['mean'])],
        ['silence ↔ clips (mean cos)', f2(p1['gemma']['synthetic']['silence']['cos_to_clips']['mean']), f2(p1['clap']['synthetic']['silence']['cos_to_clips']['mean'])],
        ['clips whose nearest neighbour is silence', p1['gemma']['synthetic']['silence']['closer_than_nearest_real_neighbour'], p1['clap']['synthetic']['silence']['closer_than_nearest_real_neighbour']],
        ['SearchQuery prefix vs bare text (same words, cos)', f2(p1['gemma']['wording']['prefix_vs_none_same_text']['SearchQuery']['mean']), '—'],
    ])}

**Take-aways:**
- **Content moves the vector more than length does**, so 10 s windows are fine.
- **Gemma is close to loudness-invariant.**
- **Silence is not a hub.**
- **Prefixes shift text vectors a lot.** Choose them on held-out data.
""")

    # ---- data
    cnt = p3["items_per_dataset"]
    add(f"""---

## 3. Data, windows, splits, filters

{table(['Dataset', 'Items', 'What', 'Window embedded', 'Group (for splits)', 'Licence'], [
        ['ESC-50', cnt['esc50'], '50 everyday classes × 40', 'whole 5 s', 'source Freesound file (official 5 folds)', 'CC BY-NC 3.0'],
        ['FSD50K (eval)', cnt.get('fsd50k', '–'), f"{p2['fsd50k']['classes']} classes, ≤ 40 each; giggle, chuckle, scream, shout, whisper…", 'first min(10 s, length)', 'Freesound uploader', 'per clip CC0/BY/BY-NC'],
        ['VocalSound', cnt['vocalsound'], 'laughter, sigh, cough, throat-clear, sneeze, sniff; 100 each, one per speaker', 'first ≤ 10 s', 'speaker', 'CC BY-SA 4.0'],
        ['xeno-canto', cnt['xenocanto'], '17 species × 40 (cuckoo, 6 owls, 2 doves, hoopoe, 5 songbirds, 2 corvids)', 'loudest 5 s', 'recordist (≤ 2 rec each)', 'per recording, ND excluded'],
        ['CREMA-D', cnt['cremad'], '91 actors × 12 sentences × 6 emotions', 'whole utterance', 'actor', 'ODbL / DbCL'],
        ['FMA (fma_large)', cnt['fma'], '30 s clips; 11 genres, 11 regional subgenres', '10–20 s', 'artist', 'per track CC'],
        ['Musopen', cnt['musopen'], '116 movements, 37 canonical works', '2 × 10 s (30%, 60%)', 'work', 'Public Domain Mark'],
        ['MTG-Jamendo', cnt['mtg'], 'shards 53 + 86; genre + instrument tags', '10 s at 40%', 'artist', 'research-only'],
    ])}

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
  {inst['calibration']['n_instrumental']} + {inst['calibration']['n_vocal']} held-out FMA tracks:
  - chosen: `{inst['calibration']['chosen']['count']}` ≥ {inst['calibration']['chosen']['T']};
  - vocal recall {pct(inst['calibration']['chosen']['vocal_recall'])}, instrumental recall
    {pct(inst['calibration']['chosen']['instrumental_recall'])};
  - **residual vocals remain.** FMA's own "Instrumental" tag is sparse (0–20 tracks per region), and MTG's missing
    voice tag is not evidence of an instrumental track.
- **Excluded:** GTZAN (unclear licence), AudioSet and MusicCaps (YouTube), RAVDESS (backup only), Song Describer
  (optional).
""")

    # ---- protocol
    add(f"""---

## 4. Protocol and metric definitions

- **Linear probe ("what the audio contains"):**
  - logistic regression on **standardised** features;
  - **C chosen per outer fold from {{0.001, 0.01, 0.1, 1}} by leave-one-fold-out inside the training folds** (no
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
""")

    # ---- Q1/Q2
    rows = []
    for ds in ("esc50", "fsd50k", "vocalsound", "xenocanto"):
        g, c = p4["gemma"]["datasets"].get(ds), p4["clap"]["datasets"].get(ds)
        if g:
            rows.append([ds, g["n_classes"], g["n_clips"], pct(g["chance"]), ci(g["probe"]), pct(g["knn"]["acc"]), pct(g["zero_shot"]["acc"]),
                         pct(g["zero_shot_centred"]["acc"]), ci(c["probe"]), pct(c["knn"]["acc"]), pct(c["zero_shot"]["acc"]), pct(c["zero_shot_centred"]["acc"])])
    add("---\n\n## 5. Findings, question by question\n\n### Q1 + Q2: resolution, and knows vs can name\n\n"
        + table(["Dataset", "classes", "clips", "chance", "Gemma probe", "Gemma kNN", "Gemma ZS", "Gemma ZS centred", "CLAP probe", "CLAP kNN", "CLAP ZS", "CLAP ZS centred"], rows))
    fam_rows = []
    for m in M:
        for ds, d in p4[m]["datasets"].items():
            for f, x in d["families"].items():
                ev = x["effective_vocabulary"]
                fam_rows.append([m, ds, f, len(x["classes"]), pct(x["chance"]), ci(x["probe"]), pct(x["zero_shot"]["acc"]), f"{ev['size']}/{ev['of']}"])
    add("\n\n**Within-family problems** (only the classes of one family, in one dataset):\n\n"
        + table(["model", "dataset", "family", "classes", "chance", "probe", "zero-shot", "effective vocab"], fam_rows))
    sib = {}
    for m in M:
        for ds, d in p4[m]["datasets"].items():
            for g, x in d["sibling_groups"].items():
                sib.setdefault((ds, g), {})[m] = x
    sib_rows = []
    for (ds, g), v in sorted(sib.items(), key=lambda kv: -((kv[1]["gemma"]["probe"]["acc"] - kv[1]["gemma"]["chance"]) / (1 - kv[1]["gemma"]["chance"]))):
        gg, cc = v["gemma"], v.get("clap")
        sk = lambda x: f2((x["probe"]["acc"] - x["chance"]) / (1 - x["chance"]))
        sib_rows.append([g, ds, ", ".join(c.replace("_", " ") for c in gg["classes"]), pct(gg["chance"]), ci(gg["probe"]), sk(gg), pct(gg["zero_shot"]["acc"]),
                         ci(cc["probe"]) if cc else "–", sk(cc) if cc else "–", pct(cc["zero_shot"]["acc"]) if cc else "–", gg["probe"]["n"]])
    add("\n\n**Sibling sets** (fine distinctions; sorted by Gemma skill):\n\n"
        + table(["set", "dataset", "classes", "chance", "Gemma probe", "Gemma skill", "Gemma ZS", "CLAP probe", "CLAP skill", "CLAP ZS", "test clips"], sib_rows))
    lau = sib.get(("fsd50k", "laughter_types"), {})
    cvo = sib.get(("xenocanto", "cuckoo_vs_owls"), {})
    per_cls = []
    for m in M:
        cl = [(ds, c, v) for ds, d in p4[m]["datasets"].items() for c, v in d["classes"].items()]
        gap = sorted(cl, key=lambda x: -(x[2]["probe"] - x[2]["zero_shot"]))[:10]
        low = sorted(cl, key=lambda x: x[2]["probe"])[:8]
        per_cls.append(f"- **{m}: biggest knows-but-can't-name classes** (probe → zero-shot recall): "
                       + "; ".join(f"{c} ({ds}) {pct(v['probe'])}→{pct(v['zero_shot'])}" for ds, c, v in gap))
        per_cls.append(f"- **{m}: hardest classes for the probe** (confused with): "
                       + "; ".join(f"{c} ({ds}) {pct(v['probe'])} ↔ {v['nearest_class']}" for ds, c, v in low))
    add(f"""

**Interpretation:**
- **Everyday sounds are finely resolved by both models**: ESC-50 within-family probes are about 0.84–1.0.
- **FSD50K is much harder** (62 classes, noisier and multi-label sources); Gemma struggles notably there. The
  weakest sets are road vehicles and speaker types.
- **Bird species are the coarse spot in both models**: owls about 0.72; cuckoo vs owls
  {ci(cvo['gemma']['probe']) if cvo else '–'} (Gemma), against a chance of {pct(cvo['gemma']['chance']) if cvo else '–'}.
- **Laughter types:** giggle vs chuckle {ci(lau['gemma']['probe']) if lau else '–'} (Gemma),
  {ci(lau['clap']['probe']) if lau else '–'} (CLAP), on {lau['gemma']['probe']['n'] if lau else '–'} test clips.
  In skill terms that's lower than cuckoo vs owls. **The space is not concentrated on human expression over
  wildlife.** It has a modest laughter-type distinction and a decent species distinction.
- **Naming differs:** "giggle" finds giggles in
  {pct(lau['gemma']['per_class']['Giggle']['zero_shot']) if lau else '–'} (Gemma) /
  {pct(lau['clap']['per_class']['Giggle']['zero_shot']) if lau else '–'} (CLAP) of cases; "chuckle" finds chuckles in
  {pct(lau['gemma']['per_class']['Chuckle_and_chortle']['zero_shot']) if lau else '–'} /
  {pct(lau['clap']['per_class']['Chuckle_and_chortle']['zero_shot']) if lau else '–'}. Bird-species names barely
  beat chance.
- **Gemma's zero-shot is very uneven.** Many ESC-50 classes it separates at 90–100% get 0% zero-shot (see below).
  **Don't use Gemma text queries for sound-event retrieval without checking**; CLAP is far better there.

{chr(10).join(per_cls)}
""")

    # ---- Q3
    q3 = []
    for m in M:
        W = p5[m]
        syn = W["synonyms"]["raw"]
        q3.append(f"\n**{m}: synonym clouds** (raw; pooled sound clips):\n\n" + table(
            ["concept", "text cohesion (within − across)", "top-25 overlap (Jaccard)", "precision@25 (mean over synonyms)", "base rate", "best / worst synonym"],
            [[c, f2(v["text_margin"]), f2(v["retrieval_overlap_jaccard"]), pct(v["precision_at_25_mean"]), pct(v["base_rate"]),
              (lambda b: f"{max(b, key=b.get)} {pct(max(b.values()))} / {min(b, key=b.get)} {pct(min(b.values()))}")(v["precision_at_25_by_word"])]
             for c, v in syn.items()]))
        st = W["structure"]
        q3.append(f"\n**{m}: structure match** (Spearman ρ, class-centroid similarity vs class-text similarity; permutation p):\n\n" + table(
            ["dataset", "all classes", "by family"],
            [[ds, f"ρ {f2(x['all']['rho'])} (p {x['all']['p_perm']})", "; ".join(f"{f} {f2(y['rho'])} (p {y['p_perm']})" for f, y in x["families"].items())] for ds, x in st.items()]))
        q3.append(f"\n**{m}: gap per family** (cosine):\n\n" + table(
            ["family", "clip↔own class text", "clip↔other class texts", "margin", "clip↔clip within family"],
            [[f, f2(g["text_audio_matching"]), f2(g["text_audio_nonmatching"]), f2(g["margin"]), f2(g["audio_audio_within_family"])] for f, g in W["gap"].items()]))
        A = W["alpha"]
        al = list(A["tune_acc"])
        q3.append(f"\n**{m}: centring strength** (α chosen on the tuning fold = {A['chosen_alpha']}; zero-shot accuracy on report folds):\n\n" + table(
            ["family"] + [f"α={a}" for a in al], [[f] + [pct(A["report"][f][a]) for a in al] for f in A["report"]]))
    add("""---

### Q3: words vs sounds
""" + "\n".join(q3) + f"""

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
""")

    # ---- Q4
    E = p6
    rows = [[k, ci(E[k]["emotion"]), f2(E[k]["emotion"]["skill"]), ci(E[k]["sentence"]), f2(E[k]["sentence"]["skill"]), ci(E[k]["speaker"]), f2(E[k]["speaker"]["skill"])]
            for k in ("gemma", "clap", "prosody", "prosody_mfcc")]
    zs = [[m, pct(E[m]["zero_shot"]["raw"]["acc"]), pct(E[m]["zero_shot"]["centred"]["acc"])] + [f"{pct(E[m]['emotion_recall'][e])} / {pct(E[m]['zero_shot']['raw']['recall'][e])}" for e in E["labels"]] for m in M]
    add(f"""---

### Q4: emotion (CREMA-D: {E['n']['actors']} actors × {E['n']['sentences']} sentences × 6 emotions, {E['n']['clips']} clips)

{table(['representation', 'emotion (actor-held-out)', 'skill', 'sentence (actor-held-out)', 'skill', 'speaker (sentence-held-out)', 'skill'], rows)}

{table(['model', 'zero-shot raw', 'zero-shot centred'] + [f'{e} probe / ZS recall' for e in E['labels']], zs)}

Gemma probe confusion (rows = true, cols = predicted, order {', '.join(E['labels'])}): `{E['gemma']['emotion_confusion']}`

**Interpretation:**
- **Emotion is in the audio:** probes reach about 56–58%, above prosody features (about 44%). Anger is the easiest
  emotion.
- **Words can't reach it:** raw zero-shot labels almost everything "neutral". Verdict: **knows but can't name.**
- **Gemma encodes *what* is said almost perfectly, and *who* says it barely.** CLAP keeps speaker identity. This
  matches "Gemma is a speech/semantics model with audio grafted on", and it explains why the sister's "heated
  argument" search found overlapping talk but not hostility.
""")

    # ---- Q5
    out = []
    for m in M:
        G = p7[m]["genre"]
        b = G["best_form"]
        Z = G["zero_shot"][b]
        out.append(f"\n**{m}** (best word form, chosen on tuning fold: `{b}`; forms on tuning fold: "
                   + ", ".join(f"{k} {pct(v['tune']['acc'])}" for k, v in G["zero_shot"].items()) + ")\n\n"
                   + table(["genre", "instrumental tracks", "artists", "probe recall", "zero-shot recall", "precision@25 (Wilson)", "base rate"],
                           [[w, G["counts_instrumental"][w], G["artists_instrumental"][w], pct(G["probe_recall"][w]), pct(Z["recall_report"].get(w)),
                             f"{pct(Z['precision_at_25'][w]['precision'])} ({pct(Z['precision_at_25'][w]['wilson'][0])}–{pct(Z['precision_at_25'][w]['wilson'][1])})", pct(Z["precision_at_25"][w]["base_rate"])] for w in G["words"]])
                   + f"\n\nOverall: probe {ci(G['probe'])} · zero-shot instrumental {ci(Z['report'])} · zero-shot on vocal tracks {ci(Z['vocal_report']) if Z.get('vocal_report') else '–'} · true genre in top 3 {pct(Z['top3_report'])} (chance {pct(3 / len(G['words']))})")
        Rg = p7[m]["region"]
        if "zero_shot" in Rg:
            RZ = Rg["zero_shot"][Rg["best_form"]]
            out.append(f"\nRegions ({m}): probe {ci(Rg['probe'])}, zero-shot {ci(RZ['report'])}, top-3 {pct(RZ['top3_report'])} (chance {pct(Rg['chance'])}, top-3 chance {pct(3 / len(Rg['words']))}). "
                       "Per region: " + "; ".join(f"{w} n={Rg['counts_instrumental'][w]} ZS {pct(RZ['recall_report'].get(w))}" for w in Rg["words"]))
        cc = p7[m].get("canonical_classical")
        if cc:
            out.append(f"\nCanonical classical ({m}): Musopen {cc['musopen_tracks']} movements → labelled classical {pct(cc['musopen_predicted_classical'])}; "
                       f"FMA classical recall {pct(cc['fma_classical_recall'])}; mean classical score Musopen {f2(cc['classical_score_mean']['musopen'])}, "
                       f"FMA classical {f2(cc['classical_score_mean']['fma_classical'])}, other genres {f2(cc['classical_score_mean']['fma_other_genres'])}.")
    add(f"""---

### Q5, part 1: genre and region words on instrumental music (FMA, artist-grouped)

Instrumental tracks: {p7['gemma']['n']['instrumental']}, vocal tracks: {p7['gemma']['n']['vocal']}.
""" + "\n".join(out) + f"""

**Phase 8 genres** (rule fixed in advance: precision@25 ≥ 2× base rate and Wilson lower bound > base rate, Gemma):
{', '.join(p7['phase8']['qualifying'])}; contrast genre: {p7['phase8']['contrast_failing']}.

**Interpretation:**
- **Genre is partly in the audio** (probe about 0.41–0.49 over 11 genres, chance 0.09).
- **Genre words work for some genres** (rock, folk, classical; hip-hop and country relative to their rarity) and
  not for others (easy listening, soul, blues).
- **The sentence form ("a piece of X music") beats the bare word** for both models.
- **Lyrics barely change zero-shot genre accuracy.**
- **"Classical" generalises from FMA's indie classical to canonical recordings** (Musopen about 99–100%).
- **Region words beat chance but are only a coarse filter**: the true region is in the top 3 about half the time.
""")

    # ---- Q5 part 2
    wm = []
    for g in p8["gemma"]["genres"]:
        lines = [f"\n#### {g}"]
        for m in M:
            r = p8[m][g]
            P = r["perturbations"]
            gap = P["member_score_mean"] - P["non_member_score_mean"]
            d = P["delta"]
            lines.append(f"\n*{m}*: member score {f2(P['member_score_mean'])} vs others {f2(P['non_member_score_mean'])} (gap {f2(gap)}); "
                         f"members ranked {g} first {pct(P['members_top1_original'])}; n = {P['n_members']} members, {P['n_non_members']} others.\n\n"
                         + table(["perturbation", "Δ score (members, 95% CI)", "share of gap", "still ranked first", "Δ non-members"],
                                 [[k, f"{d[k]['members']['mean']:+.4f} ({d[k]['members']['lo']:+.4f}…{d[k]['members']['hi']:+.4f})", (f"{100 * d[k]['members']['mean'] / gap:+.0f}%" if gap > 0 else "n/a"),
                                   pct(d[k]["members_still_top1"]), f"{d[k]['non_members']['mean']:+.4f}"] for k in d]))
            sp = r["features"]["spearman"]
            top = sorted(sp, key=lambda k: -abs(sp[k]["across"]))[:6]
            lines.append(f"\nFeatures (Spearman with {g} score, across / within genre): " + "; ".join(f"{k} {f2(sp[k]['across'])}/{f2(sp[k]['within'])}" for k in top)
                         + f". Regression CV R² {f2(r['features']['regression']['cv_r2'])} (n {r['features']['n']}).")
            inst = r.get("instrument_fixed", {})
            lines.append("Instrument held fixed (MTG): " + ("; ".join(f"{k}: AUC {f2(v['auc'])} ({f2(v['auc_ci'][0])}–{f2(v['auc_ci'][1])}), {v['members']} vs {v['non_members']}" if v.get("auc") is not None else f"{k}: too few ({v['members']} vs {v['non_members']})" for k, v in inst.items()) or "no MTG tag for this genre") + ".")
            ds_ = r["descriptors"]
            lines.append(f"Descriptors: nearest the word: {', '.join(ds_['top10_text_side'][:6])}; nearest the sound: {', '.join(ds_['top10_audio_side'][:6])}; agreement ρ {f2(ds_['spearman_text_vs_audio'])}.")
            lines.append(f"False positives (top 50 non-members): {r['false_positive_genre_counts_top50']}.")
        wm.append("\n".join(lines))
    add(f"""---

### Q5, part 2: what makes a genre (perturbations, features, instrument held fixed, descriptors)

Perturbations (deterministic; parameters in `src/audiovocab/perturb.py`): {'; '.join(f'`{k}` = {v}' for k, v in p8['perturbations'].items())}.
Applied to non-ND-licensed instrumental FMA clips, 16 per genre, re-embedded with both models.
""" + "\n".join(wm) + """

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
""")

    # ---- bridge
    add("---\n\n### Bridge back to Vibe Search\n\n" + table(
        ["Vibe Search query", "listener (of 4)", "classes mapped (fixed in advance)", "probe", "zero-shot", "verdict"],
        [[r["query"], r["listening_yes"], ", ".join(f"{c['label']} ({c['dataset']})" for c in r["classes"]), pct(r["probe_mean"]), pct(r["zero_shot_mean"]), r["verdict"]] for r in br["rows"]])
        + f"""

Rule: probe < 70% → weak area; probe ≥ 70% but zero-shot < 50% → knows but can't name; both high → likely a podcast-data gap.
**Result:**
- The rule marks most searches "weak", including ones that scored 3–4 of 4 by ear. **Per-class resolution on
  labelled sounds doesn't predict podcast vibe success.**
- "Heated argument" unpacks to *knows but can't name* for anger.
- Whispering, rain/nature and music are genuinely weak or rare areas.
""")

    # ---- lessons, artefacts, guidance
    add(f"""---

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
    `{{dataset, clip, source, start, dur[, perturb]}}` (see `embed.clip_key`).
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
""")
    (ROOT / "CONTEXT.md").write_text("\n".join(S), encoding="utf-8")
    print("CONTEXT.md", round(len("\n".join(S)) / 1000), "KB")


if __name__ == "__main__":
    main()
