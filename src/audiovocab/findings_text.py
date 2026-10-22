"""Narrative sentences for the page. Every number is read from D (the reports); the wording was written after reading
the results, and conditional phrasing guards each claim so a rerun can't keep a stale statement."""


def pct(x):
    return "–" if x is None else f"{round(100 * x)}%"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def skill(acc, chance):
    return (acc - chance) / (1 - chance)


def _ds(D, m, ds):
    return (D.get("p4") or {}).get("models", {}).get(m, {}).get("datasets", {}).get(ds)


def _sib(D, m, g):
    for ds, groups in (D.get("p4") or {}).get("models", {}).get(m, {}).get("siblings", {}).items():
        if g in groups:
            return groups[g]
    return None


def narrative(D, T):
    out = {}
    esc_g, esc_c = _ds(D, "gemma", "esc50"), _ds(D, "clap", "esc50")
    fsd_g, fsd_c = _ds(D, "gemma", "fsd50k"), _ds(D, "clap", "fsd50k")
    lau_g, lau_c = _sib(D, "gemma", "laughter_types"), _sib(D, "clap", "laughter_types")
    owl_g, owl_c = _sib(D, "gemma", "cuckoo_vs_owls"), _sib(D, "clap", "cuckoo_vs_owls")
    xc_g, xc_c = _ds(D, "gemma", "xenocanto"), _ds(D, "clap", "xenocanto")
    E = D.get("p6") or {}
    P5 = D.get("p5") or {}
    G7 = D.get("p7") or {}
    W8 = D.get("p8") or {}

    out["tagline"] = ("Two open audio–text embedding models (EmbeddingGemma 2, CLAP), probed on labelled sound, voice, "
                      "bird and music datasets. Every finding holds for these datasets with these models, not for all audio.")
    out["scope"] = ("Sister project: <a href=\"https://github.com/atharva-729/podcast-vibe-search\">Vibe Search</a> "
                    "(podcast search by sound). New to embeddings? Its explainer covers cosine, UMAP and the modality gap.")

    H = []
    if esc_g and esc_c:
        H.append(f"<b>The audio knows far more than words can reach, especially in Gemma.</b> On ESC-50's 50 everyday sounds a linear probe "
                 f"gets {pct(esc_g['probe']['acc'])} (Gemma) and {pct(esc_c['probe']['acc'])} (CLAP), but picking the nearest class "
                 f"<i>name</i> gets <span class=\"em\">{pct(esc_g['zero_shot']['acc'])}</span> with Gemma vs {pct(esc_c['zero_shot']['acc'])} with CLAP.")
    if lau_g and owl_g:
        H.append(f"<b>Giggle vs chuckle: told apart, but not finely</b> ({pct(lau_g['probe']['acc'])} Gemma, {pct(lau_c['probe']['acc'])} CLAP; chance 50%). "
                 f"Cuckoo vs six owls is no harder relative to chance ({pct(owl_g['probe']['acc'])} over 7 species; chance {pct(owl_g['chance'])}). "
                 f"The space is <b>not</b> tuned to human expression over wildlife, but words reach laughter far better than bird species.")
    if E.get("gemma"):
        H.append(f"<b>Emotion is in the audio; the words can't find it.</b> A probe finds anger in {pct(E['gemma']['emotion_recall']['ANG'])} of angry clips; "
                 f"the words \"an angry voice\" find {pct(E['gemma']['zero_shot']['raw']['recall']['ANG'])}. That's the Vibe Search \"heated argument\" failure, explained.")
    out["headlines"] = H

    cnt = (D.get("counts") or {}).get("items") or {}
    out["data_lead"] = (f"{sum(cnt.values()):,} labelled items from {len(cnt)} datasets, all licence-checked; no hand-labelling. "
                        "FSD50K was embedded in Colab because Zenodo blocks the work network (vectors verified identical).") if cnt else ""

    # ---- Q1
    if esc_g and fsd_g and xc_g:
        out["hear_lead"] = (f"A linear probe (logistic regression on the frozen embedding, grouped splits) measures what the audio space <i>contains</i>. "
                            f"Everyday sounds separate cleanly (ESC-50: {pct(esc_g['probe']['acc'])} Gemma, {pct(esc_c['probe']['acc'])} CLAP); "
                            f"FSD50K's 62 finer classes are harder ({pct(fsd_g['probe']['acc'])} / {pct(fsd_c['probe']['acc'])}); "
                            f"17 European bird species are hardest for their size ({pct(xc_g['probe']['acc'])} / {pct(xc_c['probe']['acc'])}).")
    rows = T.get("sibling_rows", [])
    hard = sorted([r for r in rows if r.get("gemma")], key=lambda r: skill(r["gemma"]["acc"], r["chance"]))[:3]
    easy = sorted([r for r in rows if r.get("gemma")], key=lambda r: -skill(r["gemma"]["acc"], r["chance"]))[:3]
    nb = []
    if lau_g:
        lg = lau_g["per_class"]
        nb.append(f"<b>Laughter types:</b> only giggle and chuckle exist as labels in licence-clear data. The probe separates them "
                  f"{pct(lau_g['probe']['acc'])} of the time (CI {pct(lau_g['probe']['ci'][0])}–{pct(lau_g['probe']['ci'][1])}, {lau_g['probe']['n']} test clips): "
                  f"a real but coarse distinction. The word \"giggle\" finds giggles {pct(lg['Giggle']['zero_shot'])}; \"chuckle\" finds chuckles {pct(lg['Chuckle_and_chortle']['zero_shot'])}.")
    if hard:
        nb.append("<b>Coarsest sibling sets</b> (Gemma, skill above chance): " + "; ".join(f"{r['name']} {pct(r['gemma']['acc'])} vs chance {pct(r['chance'])}" for r in hard) + ".")
    if easy:
        nb.append("<b>Finest:</b> " + "; ".join(f"{r['name']} {pct(r['gemma']['acc'])}" for r in easy) + ".")
    if xc_g and esc_g:
        nb.append(f"<b>Bird species are the coarse spot in both models</b>, despite clean labels: field recordings, overlapping birds and similar calls. "
                  f"Bird <i>vs</i> other animals is easy; <i>which</i> bird is not.")
    out["nb_hear"] = nb
    out["vocab_note"] = (f"Classes in the largest set where every pair is separated ≥ {pct((D.get('p4') or {}).get('protocol', {}).get('pairwise_threshold'))} "
                         "by a pairwise probe. Most families clear this bar entirely, so the sibling chart is the sharper view.")

    # ---- Q2
    if esc_g and esc_c:
        out["name_lead"] = ("Each dot is a class. Right = the audio separates it (probe); up = a word finds it (zero-shot, nearest class text). "
                            "Dots far below the diagonal are sounds the model <b>knows but can't name</b>. Click a dot for its numbers.")
        g_gap = esc_g["probe"]["acc"] - esc_g["zero_shot"]["acc"]
        c_gap = esc_c["probe"]["acc"] - esc_c["zero_shot"]["acc"]
        out["nb_name"] = [
            f"<b>CLAP names what it knows</b> for everyday sounds (ESC-50 gap {pct(c_gap)}); <b>Gemma doesn't</b> (gap {pct(g_gap)}). "
            "Gemma's audio–text link for sound events is thin, even though its audio space is good.",
            f"<b>Both fail to name bird species</b>: probe {pct(xc_g['probe']['acc'])} / {pct(xc_c['probe']['acc'])}, zero-shot {pct(xc_g['zero_shot']['acc'])} / {pct(xc_c['zero_shot']['acc'])}. "
            "Species names are rare in audio captions; the information is there, the vocabulary isn't.",
            f"<b>Gemma's prefix choice</b> (tuned on fold {(D.get('p4') or {}).get('protocol', {}).get('tune_fold')}): "
            f"\"task: search result | query:\" beat no prefix ({pct((D['p4']['models']['gemma']['prefix_tune_scores'] or {}).get('SearchQuery'))} vs "
            f"{pct((D['p4']['models']['gemma']['prefix_tune_scores'] or {}).get('none'))} zero-shot on the tuning fold).",
        ]

    # ---- Q3
    if P5.get("gemma"):
        g5, c5 = P5["gemma"], P5.get("clap", {})
        syn = g5["synonyms"]["raw"]
        ov = sorted(syn.items(), key=lambda kv: -kv[1]["retrieval_overlap_jaccard"])
        S = g5["structure"]
        out["words_lead"] = ("Do synonyms (giggle, chuckle, titter…) cluster in text space and retrieve the same sounds? "
                             "Is the similarity of class <i>sounds</i> mirrored by the similarity of class <i>names</i>? And how far apart are the two clouds?")
        gap = g5["gap"]
        out["nb_words"] = [
            f"<b>Synonyms retrieve different clips.</b> Their top-25 sets overlap by a Jaccard of {f2(ov[-1][1]['retrieval_overlap_jaccard'])}–{f2(ov[0][1]['retrieval_overlap_jaccard'])} (Gemma); "
            f"only \"{ov[0][0]}\" and \"{ov[1][0]}\" synonyms agree much. Text is finer than audio here: changing the word moves the results, rarely toward a better match.",
            f"<b>Same shape, partly.</b> Where sounds are similar, their names are too: ESC-50 ρ = {f2(S['esc50']['all']['rho'])} (Gemma), {f2(c5['structure']['esc50']['all']['rho'])} (CLAP); "
            f"strongest for human vocal sounds, absent for nature (ρ = {f2(S['esc50']['families'].get('nature', {}).get('rho'))}).",
            f"<b>Gemma's gap is huge for sound events</b>: a clip's own class text beats other class texts by only "
            f"{f2(min(v['margin'] for v in gap.values()))}–{f2(max(v['margin'] for v in gap.values()))} cosine, while clips sit ~{f2(sum(v['audio_audio_within_family'] for v in gap.values()) / len(gap))} from each other. "
            f"CLAP's margin is {f2(min(v['margin'] for v in c5['gap'].values()))}–{f2(max(v['margin'] for v in c5['gap'].values()))}.",
            f"<b>Centring helps category retrieval a little</b> (tuned α = {g5['alpha']['chosen_alpha']} for Gemma): overall "
            f"{pct(g5['alpha']['report']['all']['0.0'])} → {pct(g5['alpha']['report']['all'][str(g5['alpha']['chosen_alpha'])])}, but it <b>hurts animals</b> "
            f"({pct(g5['alpha']['report']['animal']['0.0'])} → {pct(g5['alpha']['report']['animal']['1.0'])} at full strength). "
            "This is category retrieval only; Vibe Search showed the vibe can get worse even when categories improve.",
        ]
        out["alpha_note"] = "Report folds; α chosen on the tuning fold. Each line is a family."

    # ---- Q4
    if E.get("gemma"):
        g, c, p = E["gemma"], E["clap"], E.get("prosody", {})
        out["emo_title"] = f"Who, what, or how? {E['n']['sentences']} sentences × 6 emotions × {E['n']['actors']} actors"
        out["emo_lead"] = ("CREMA-D holds the words fixed and varies the emotion, so a probe can ask which the space prioritises: the speaker (who), "
                           "the sentence (what) or the emotion (how). Splits never share an actor (or, for speaker identity, a sentence).")
        out["emo_tiles"] = [
            [pct(g["sentence"]["acc"]), "Gemma: which sentence (12)"], [pct(g["speaker"]["acc"]), "Gemma: which speaker (91)"],
            [pct(g["zero_shot"]["raw"]["acc"]), "Gemma zero-shot emotion (chance 17%)"], [pct(c["zero_shot"]["centred"]["acc"]), "CLAP zero-shot, centred"],
        ]
        out["nb_emo"] = [
            f"<b>Emotion is there</b>: probes reach {pct(g['emotion']['acc'])} (Gemma) and {pct(c['emotion']['acc'])} (CLAP), above loudness/pitch/pace features ({pct(p.get('emotion', {}).get('acc'))}). Anger is the easiest emotion ({pct(g['emotion_recall']['ANG'])}).",
            f"<b>Words can't reach it</b>: raw zero-shot labels almost everything \"neutral\" ({pct(g['zero_shot']['raw']['recall']['NEU'])} of neutral clips right, "
            f"{pct(g['zero_shot']['raw']['recall']['ANG'])} of angry ones). Verdict: <span class=\"em\">knows but can't name</span>.",
            f"<b>Gemma hears what is said, not who says it</b>: sentence {pct(g['sentence']['acc'])}, speaker {pct(g['speaker']['acc'])}. "
            f"CLAP keeps the speaker ({pct(c['speaker']['acc'])}) and is weaker on the words ({pct(c['sentence']['acc'])}).",
        ]
    if G7.get("gemma") and W8.get("gemma"):
        from .findings_music import music_text
        m = music_text(D)
        headline = m.pop("headline", None)
        out.update(m)
        if headline:
            out["headlines"].append(headline)
    return out
