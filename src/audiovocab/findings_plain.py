"""Plain-language page text (structure of EXPLAINED.md). Every number is read from D (the reports)."""

DSN = {"esc50": "Everyday sounds (ESC-50)", "fsd50k": "Fine sound classes (FSD50K)", "vocalsound": "Human vocal sounds (VocalSound)",
       "xenocanto": "Bird species (xeno-canto)", "cremad": "Emotion in voices (CREMA-D)", "fma": "Music genre (FMA)"}


def pct(x):
    return "–" if x is None else f"{round(100 * x)}%"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def plain(D):
    T = {}
    P4 = (D.get("p4") or {}).get("models", {})
    E, G7, W8, B = D.get("p6") or {}, D.get("p7") or {}, D.get("p8") or {}, D.get("bridge") or {}
    ds = lambda m, d: P4.get(m, {}).get("datasets", {}).get(d)

    # chart rows: probe vs zero-shot per data set, both models
    rows = []
    for d in ("esc50", "fsd50k", "vocalsound", "xenocanto"):
        g, c = ds("gemma", d), ds("clap", d)
        if g and c:
            rows.append({"id": d, "name": DSN[d], "chance": g["chance"],
                         "gemma": {"probe": g["probe"]["acc"], "zs": g["zero_shot"]["acc"]},
                         "clap": {"probe": c["probe"]["acc"], "zs": c["zero_shot"]["acc"]}})
    if E.get("gemma"):
        rows.append({"id": "cremad", "name": DSN["cremad"], "chance": E["gemma"]["emotion"]["chance"],
                     "gemma": {"probe": E["gemma"]["emotion"]["acc"], "zs": E["gemma"]["zero_shot"]["raw"]["acc"]},
                     "clap": {"probe": E["clap"]["emotion"]["acc"], "zs": E["clap"]["zero_shot"]["raw"]["acc"]}})
    if G7.get("gemma"):
        gg, gc = G7["gemma"]["genre"], G7["clap"]["genre"]
        rows.append({"id": "fma", "name": DSN["fma"], "chance": gg["chance"],
                     "gemma": {"probe": gg["probe"]["acc"], "zs": gg["zero_shot"][gg["best_form"]]["report"]["acc"]},
                     "clap": {"probe": gc["probe"]["acc"], "zs": gc["zero_shot"][gc["best_form"]]["report"]["acc"]}})
    T["pz_rows"] = rows
    R = {r["id"]: r for r in rows}

    esc = R.get("esc50")

    def sibl(m, g):
        for d, groups in P4.get(m, {}).get("siblings", {}).items():
            if g in groups:
                return groups[g]
    lau_g, lau_c, owl_g, owl_c = sibl("gemma", "laughter_types"), sibl("clap", "laughter_types"), sibl("gemma", "cuckoo_vs_owls"), sibl("clap", "cuckoo_vs_owls")
    owls_g = sibl("gemma", "owls")
    cls = (D.get("p4") or {}).get("classes", [])
    gem_esc = sorted([c for c in cls if c["model"] == "gemma" and c["dataset"] == "esc50"], key=lambda c: -(c["probe"] - c["zero_shot"]))[:5]
    canon = (G7.get("gemma") or {}).get("canonical_classical") or {}
    canon_c = (G7.get("clap") or {}).get("canonical_classical") or {}

    # hero
    T["p_tagline"] = ("We took audio where every clip already has a correct label (everyday sounds, laughter types, bird species, "
                      "actors speaking in different emotions, music by genre) and measured what two AI audio models can tell apart, "
                      "and whether typing a <b>word</b> can find it.")
    H = []
    if esc:
        H.append(f"<b>Words find sounds badly in Gemma, well in CLAP.</b> On everyday sounds, typing the word finds the right sound "
                 f"<span class=\"em\">{pct(esc['gemma']['zs'])}</span> of the time with Gemma, {pct(esc['clap']['zs'])} with CLAP (guessing: {pct(esc['chance'])}).")
    if lau_g and owl_g:
        H.append(f"<b>Laughter types are only roughly separable</b>: giggle vs chuckle {pct(lau_g['probe']['acc'])} (guessing 50%). "
                 f"Owl species are told apart at least as well, so the model isn't specially tuned to human sounds.")
    if E.get("gemma"):
        H.append(f"<b>Emotion is in there, but unnameable.</b> A trained probe finds {pct(E['gemma']['emotion_recall']['ANG'])} of angry clips; "
                 f"typing \"an angry voice\" finds {pct(E['gemma']['zero_shot']['raw']['recall']['ANG'])}. That's why the podcast search's \"heated argument\" failed.")
    cl = (W8.get("gemma") or {}).get("classical")
    if cl:
        P = cl["perturbations"]
        gap = P["member_score_mean"] - P["non_member_score_mean"]
        dr = -P["delta"]["percussive_only"]["members"]["mean"] / gap if gap > 0 else None
        H.append(f"<b>To the model, \"classical\" means tonal sound without drums.</b> Keep only the drums and the classical signal vanishes "
                 f"({'all of it' if dr is not None and dr >= 1 else pct(dr)} gone); changing tempo or pitch does almost nothing.")
    T["p_headlines"] = H
    T["p_hero_note"] = ("Each pair of bars: <b>solid</b> = is the information in the model (probe), <b>light</b> = can a typed word find it (zero-shot). "
                        "Grey tick = guessing. Click a row for examples.")

    # story
    T["p_story"] = [
        "<b>The sister project, Vibe Search</b>, built a podcast search where you type a <i>vibe</i> (\"two friends laughing about football\") "
        "and it finds 20-second clips that <i>sound</i> like that. It used <b>EmbeddingGemma 2</b>, a Google model that turns both sounds and text "
        "into lists of 768 numbers (an <i>embedding</i>). If laughter and the words \"people laughing\" get similar numbers, you can search sound with words.",
        f"It worked for some searches and failed for others ({(B.get('listening_total') or ['–', '–'])[0]} of {(B.get('listening_total') or ['–', '–'])[1]} results sounded right to a listener). "
        "It found people <i>talking over each other</i> but not <i>hostility</i>, and words and sounds sat in separate regions of the space.",
        "<b>Anand's question:</b> \"With words, a dozen synonyms of <i>love</i> cluster tightly. What's the audio equivalent? Can it tell 50 kinds of laughter apart? "
        "A cuckoo from an owl? Is it tuned to human sounds or to nature?\"",
        "<b>So this project</b> swapped messy podcasts for <b>datasets where every clip already has a correct label</b>, and measured exactly what the model can "
        "and can't tell apart, and whether <i>words</i> can find what the <i>sound</i> side knows.",
    ]

    # data lead
    T["p_data_lead"] = ("All public, already labelled; nothing was labelled by hand. Every test uses <b>different recordings, speakers or artists</b> "
                        "than the probe was trained on, so a model can't cheat by recognising the same microphone or person. For music, tracks with "
                        "singing were removed with a speech-recognition model (Whisper), otherwise lyrics would give the genre away.")

    # finding 1
    if esc:
        ex = "; ".join(f"{c['label'].replace('_', ' ')} (probe {pct(c['probe'])}, word {pct(c['zero_shot'])})" for c in gem_esc)
        T["p_f1"] = [
            f"<b>What we ran:</b> both models on 50 everyday sounds (guessing = {pct(esc['chance'])}). Gemma's probe gets {pct(esc['gemma']['probe'])}, but the word finds "
            f"only {pct(esc['gemma']['zs'])}. CLAP: {pct(esc['clap']['probe'])} and {pct(esc['clap']['zs'])}.",
            f"<b>Examples (Gemma):</b> {ex}. The sound side knows them; typing the word finds almost none.",
            "<b>Meaning:</b> Gemma's \"ears\" are fine, but its link from <i>words to sound effects</i> is very weak. CLAP was trained for exactly that link. "
            "<b>Bad news for Gemma as a sound-effect search engine; good news for CLAP.</b>",
            f"<b>Birds are the exception for both:</b> the probe separates 17 species ({pct(R['xenocanto']['gemma']['probe'])} / {pct(R['xenocanto']['clap']['probe'])}), "
            f"but species names barely work ({pct(R['xenocanto']['gemma']['zs'])} / {pct(R['xenocanto']['clap']['zs'])}).",
        ]
    # finding 2
    if lau_g and owl_g:
        lg = lau_g["per_class"]
        T["p_f2"] = [
            f"<b>Giggle vs chuckle:</b> {pct(lau_g['probe']['acc'])} (Gemma) and {pct(lau_c['probe']['acc'])} (CLAP), where guessing gets 50%. Real, but coarse: nowhere near \"50 kinds of laughter\". "
            "Only these two laughter types exist as labels in usable public data.",
            f"<b>Cuckoo vs 6 owl species:</b> {pct(owl_g['probe']['acc'])} / {pct(owl_c['probe']['acc'])} across 7 species (guessing {pct(owl_g['chance'])}). Relative to guessing, that's <i>better</i> than laughter.",
            "<b>Answer to Anand:</b> the space is <span class=\"em\">not</span> specially tuned to human expression over wildlife.",
            f"<b>But words differ:</b> typing \"giggle\" finds giggles {pct(lg['Giggle']['zero_shot'])} of the time, while owl species names barely beat guessing "
            f"({pct(owls_g['zero_shot']['acc']) if owls_g else '–'}). The model hears bird species but has no vocabulary for them.",
        ]
    # finding 3
    if E.get("gemma"):
        g, c, p = E["gemma"], E["clap"], E.get("prosody", {})
        T["p_f3"] = [
            f"<b>What we ran:</b> {E['n']['actors']} actors saying the same {E['n']['sentences']} sentences in 6 emotions (guessing ≈ {pct(g['emotion']['chance'])}).",
            f"<b>Probe on emotion: {pct(g['emotion']['acc'])} (Gemma), {pct(c['emotion']['acc'])} (CLAP)</b>, above simple loudness/pitch measurements ({pct(p.get('emotion', {}).get('acc'))}). "
            f"Anger is easiest: the probe finds {pct(g['emotion_recall']['ANG'])} of angry clips.",
            f"<b>Typing \"an angry voice\" finds {pct(g['zero_shot']['raw']['recall']['ANG'])}</b> (CLAP: {pct(c['zero_shot']['raw']['recall']['ANG'])}). Both models call almost everything \"neutral\".",
            "<b>\"Encoded but unnameable\"</b> = the emotion is <i>in</i> the numbers (a probe trained on examples reads it), but no text query pulls it out.",
            f"<b>Bonus:</b> Gemma knows <i>which sentence</i> was said ({pct(g['sentence']['acc'])}) but barely <i>who</i> said it ({pct(g['speaker']['acc'])} of {E['n']['actors']} people). "
            f"CLAP: who {pct(c['speaker']['acc'])}. Gemma is a \"words\" model with ears; CLAP hears the voice itself.",
        ]
        T["p_emo_tiles"] = [[pct(g["emotion_recall"]["ANG"]), "angry clips found by a probe (Gemma)"],
                            [pct(g["zero_shot"]["raw"]["recall"]["ANG"]), "angry clips found by typing \"an angry voice\""],
                            [pct(g["sentence"]["acc"]), "Gemma: which sentence was said"],
                            [pct(g["speaker"]["acc"]), f"Gemma: who said it (of {E['n']['actors']})"]]
    # finding 4
    if G7.get("gemma"):
        gg, gc = G7["gemma"]["genre"], G7["clap"]["genre"]
        bz = gg["zero_shot"][gg["best_form"]]
        prec = bz["precision_at_25"]
        good = sorted(prec, key=lambda w: -prec[w]["precision"] / max(prec[w]["base_rate"], 1e-9))
        Rg = G7["gemma"]["region"]
        rz = Rg.get("zero_shot", {}).get(Rg.get("best_form", ""), {})
        T["p_f4"] = [
            f"<b>What we ran:</b> {G7['gemma']['n']['instrumental']:,} instrumental tracks, {len(gg['words'])} genres (guessing {pct(gg['chance'])}). "
            f"Probe {pct(gg['probe']['acc'])} (Gemma) / {pct(gc['probe']['acc'])} (CLAP); words {pct(bz['report']['acc'])} / {pct(gc['zero_shot'][gc['best_form']]['report']['acc'])}. "
            "Genre is partly in the sound; words partly reach it.",
            "<b>Words that work:</b> " + ", ".join(f"{w} ({pct(prec[w]['precision'])} of the 25 nearest tracks, random would be {pct(prec[w]['base_rate'])})" for w in ("rock", "folk", "classical") if w in prec)
            + ". <b>Words that don't:</b> " + ", ".join(w for w in good[-3:]) + ".",
            f"<b>Real classical:</b> {pct(canon.get('musopen_predicted_classical'))} (Gemma) / {pct(canon_c.get('musopen_predicted_classical'))} (CLAP) of Musopen's Beethoven/Mozart recordings are labelled \"classical\".",
        ]
        if rz:
            T["p_f4"].append(f"<b>Regions</b> (Brazilian, Indian…) are weak: the right region is in the top 3 {pct(rz['top3_report'])} of the time (guessing: {pct(3 / len(Rg['words']))}).")
        T["p_music_tiles"] = [[pct(gg["probe"]["acc"]), f"genre probe, Gemma (guessing {pct(gg['chance'])})"], [pct(bz["report"]["acc"]), "genre word finds the genre, Gemma"],
                              [pct(canon.get("musopen_predicted_classical")), "real classical labelled \"classical\""], [pct(bz["top3_report"]), "right genre in the word's top 3"]]
    if cl:
        P = cl["perturbations"]
        gap = P["member_score_mean"] - P["non_member_score_mean"]
        sh = lambda k: -P["delta"][k]["members"]["mean"] / gap
        inst = cl.get("instrument_fixed", {}).get("piano", {})
        instc = ((W8.get("clap") or {}).get("classical") or {}).get("instrument_fixed", {}).get("piano", {})
        T["p_f4b"] = [
            "<b>We edited the audio and re-measured</b> how \"classical\" each classical track looked to the model.",
            f"<b>Keep only the drums</b> → the classical signal disappears ({'all of it' if sh('percussive_only') >= 1 else pct(sh('percussive_only'))} gone). <b>Remove the drums</b> → <i>any</i> track looks more classical.",
            f"<b>Change tempo or pitch</b> → almost no effect ({pct(abs(sh('tempo_x0.75')))}–{pct(abs(sh('pitch_+3')))} of the signal).",
            "<b>So to the model, classical = tonal, drum-free, softer sound</b>, not the composition.",
            f"<b>It's more than \"piano\":</b> it tells classical piano from jazz/pop piano (AUC {f2(inst.get('auc'))} Gemma, {f2(instc.get('auc'))} CLAP; 0.5 = guessing, 1.0 = perfect).",
        ]
    # finding 5
    if B.get("rows"):
        T["p_f5"] = [
            "The podcast searches that worked (\"friends laughing\", \"bedtime story voice\") succeed because of <b>speaking style and the sound of a show</b>, which these labelled datasets don't measure.",
            "The ones that failed (heated argument, whispering, rain, music) sit exactly where we found weaknesses: <b>emotion is unnameable</b>, and <b>fine sounds are weak in Gemma</b>.",
        ]
    # which model + business
    if esc and E.get("gemma"):
        T["p_which"] = [
            ["Search sound effects, animals or music by typing words", "CLAP", f"its words actually reach the sounds ({pct(esc['clap']['zs'])} vs {pct(esc['gemma']['zs'])})"],
            ["Search speech by style or content (podcasts, calls)", "Gemma", f"it understands spoken content ({pct(E['gemma']['sentence']['acc'])} sentence ID); the podcast search worked on it"],
            ["Recognise who is speaking", "CLAP (or classic voice features)", f"Gemma barely encodes the speaker ({pct(E['gemma']['speaker']['acc'])})"],
            ["Detect emotion, hostility, bird species", "Neither model's words: train a small probe", "the information is there; typing words doesn't get it"],
        ]
    T["p_business"] = [
        f"<b>Sound search for audio libraries</b> (sound effects, stock audio, archives): CLAP + text search is accurate enough today ({pct(esc['clap']['zs']) if esc else '–'} on everyday sounds).",
        "<b>Probe detectors for things words can't find</b>: angry calls in a call centre, bird species for ecology, laughter in content. Collect a few hundred labelled examples, "
        "train a tiny classifier on the embeddings (seconds), deploy. <span class=\"em\">When search fails, a small trained probe often works.</span>",
        "<b>Music tagging and filtering</b> for background-music selection: works for clear genres (rock, classical, folk), not fuzzy ones; region only as a rough filter.",
        "<b>Vibe Search upgrade</b>: send sound-effect queries to CLAP and speech/style queries to Gemma; add probes for emotion.",
        "<b>Cheap to run</b>: small models on a laptop CPU, nothing retrained. <b>Caveats:</b> internal study on public datasets; ESC-50 and MTG-Jamendo are non-commercial / research-only; "
        "some results rest on small samples (59 clips for giggle vs chuckle, 16 tracks per genre for the \"what makes classical\" edits).",
    ]
    T["p_glossary"] = [
        "<b>Embedding</b>: a list of numbers (768 for Gemma, 512 for CLAP) representing a clip or a sentence; similar things get similar numbers.",
        "<b>Probe</b>: a tiny classifier trained on embeddings to test whether information is <i>in</i> them. The big model is never changed.",
        "<b>Zero-shot</b>: label each clip by the nearest typed word, no training (how a search box works).",
        "<b>Chance / guessing</b>: 1 ÷ number of options. 60% is great with 50 options, poor with 2.",
        "<b>Knows but can't name</b>: probe high, zero-shot low.",
        "<b>Modality gap</b>: words and sounds sit in separate regions of the space even when they mean the same thing.",
        "<b>Centring</b>: subtract each side's average to pull words and sounds together; helps category lookups slightly, can hurt some sounds.",
        "<b>AUC</b>: how well a score separates two groups (0.5 = coin flip, 1.0 = perfect).",
        "<b>Whisper</b>: speech-to-text model; used only to detect singing. <b>Perturbation</b>: deliberately editing audio (remove drums, change tempo) to see what the model reacts to.",
    ]
    return T
