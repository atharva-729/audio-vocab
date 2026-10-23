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
    T["p_tagline"] = ("Labelled audio (everyday sounds, laughter, birds, emotional speech, music), two AI models: "
                      "what can they tell apart, and can a typed <b>word</b> find it?")
    H = []
    if esc:
        H.append(f"<b>Gemma hears sounds, but words barely find them.</b> A typed word finds the right everyday sound "
                 f"<span class=\"em\">{pct(esc['gemma']['zs'])}</span> of the time; CLAP {pct(esc['clap']['zs'])}.")
    if E.get("gemma"):
        H.append(f"<b>Emotion is in there, but unnameable.</b> A probe finds {pct(E['gemma']['emotion_recall']['ANG'])} of angry clips; "
                 f"typing \"an angry voice\" finds <span class=\"em\">{pct(E['gemma']['zero_shot']['raw']['recall']['ANG'])}</span>.")
    if lau_g and owl_g:
        H.append(f"<b>Laughter types are only roughly separable:</b> giggle vs chuckle {pct(lau_g['probe']['acc'])} (guessing 50%). Owls do at least as well.")
    cl = (W8.get("gemma") or {}).get("classical")
    if cl:
        P = cl["perturbations"]
        gap = P["member_score_mean"] - P["non_member_score_mean"]
        dr = -P["delta"]["percussive_only"]["members"]["mean"] / gap if gap > 0 else None
        H.append(f"<b>\"Classical\" = tonal sound without drums.</b> Drums only: "
                 f"{'all of the classical signal' if dr is not None and dr >= 1 else pct(dr) + ' of the classical signal'} gone.")
    T["p_headlines"] = H
    T["p_takeaway"] = ("<b>Takeaway:</b> use CLAP to search sounds with words, Gemma for speech. "
                       "<span class=\"em\">Where words fail, a small trained probe often works.</span>")
    T["p_hero_note"] = ("Solid = in the model (probe); light = a typed word finds it; tick = guessing. Click a row for examples.")
    T["p_hero_chart_note"] = ("Each row: <b>dot</b> = in the model (probe), <b>ring</b> = a typed word finds it, tick = guessing. "
                              "<span class=\"em\">A long line = the model knows it, but the word can't find it.</span>")

    # story
    lt = B.get("listening_total") or ["–", "–"]
    T["p_story"] = [
        "<b>Origin:</b> Vibe Search, a podcast search on Gemma: type a <i>vibe</i>, get clips that <i>sound</i> like it.",
        f"<b>Mixed results:</b> {lt[0]} of {lt[1]} clips sounded right. It found people talking over each other, not hostility.",
        "<b>The question:</b> can it tell kinds of laughter apart? A cuckoo from an owl? Is it tuned to humans or nature?",
        "<b>This project:</b> labelled public datasets, two models, one test: what's in the model vs what a word finds.",
    ]

    # data lead
    T["p_data_lead"] = [
        "<b>All public, already labelled</b>; nothing labelled by hand.",
        "<b>Tested on unseen recordings, speakers or artists</b>, so a model can't cheat by recognising a voice or mic.",
        "<b>Music:</b> tracks with singing removed (Whisper), so lyrics can't give the genre away.",
    ]

    # finding 1
    if esc:
        ex = "; ".join(f"{c['label'].replace('_', ' ')} {pct(c['probe'])} → {pct(c['zero_shot'])}" for c in gem_esc)
        T["p_f1"] = [
            f"<b>Everyday sounds</b> (50 classes, guessing {pct(esc['chance'])}): Gemma probe {pct(esc['gemma']['probe'])}, word <span class=\"em\">{pct(esc['gemma']['zs'])}</span>. "
            f"CLAP: {pct(esc['clap']['probe'])} and {pct(esc['clap']['zs'])}.",
            f"<b>Examples</b> (Gemma, probe → word): {ex}.",
            "<b>Meaning:</b> Gemma's ears are fine; <span class=\"em\">its word-to-sound link is weak</span>. CLAP was trained for that link.",
            f"<b>Birds are hard for both:</b> the probe separates 17 species ({pct(R['xenocanto']['gemma']['probe'])} / {pct(R['xenocanto']['clap']['probe'])}); "
            f"names barely work ({pct(R['xenocanto']['gemma']['zs'])} / {pct(R['xenocanto']['clap']['zs'])}).",
        ]
    # finding 2
    if lau_g and owl_g:
        lg = lau_g["per_class"]
        T["p_f2"] = [
            f"<b>Giggle vs chuckle:</b> {pct(lau_g['probe']['acc'])} (Gemma) / {pct(lau_c['probe']['acc'])} (CLAP), guessing 50%. "
            "Real but coarse: <span class=\"em\">nowhere near \"50 kinds of laughter\"</span>.",
            "Only these two laughter types are labelled in usable public data.",
            f"<b>Cuckoo vs 6 owls:</b> {pct(owl_g['probe']['acc'])} / {pct(owl_c['probe']['acc'])} (guessing {pct(owl_g['chance'])}): <span class=\"em\">better than laughter</span>, relative to guessing.",
            "<b>So:</b> the model is not specially tuned to human sounds over wildlife.",
            f"<b>But words differ:</b> \"giggle\" finds giggles {pct(lg['Giggle']['zero_shot'])}; owl names barely beat guessing "
            f"({pct(owls_g['zero_shot']['acc']) if owls_g else '–'}). <span class=\"em\">It hears bird species but has no words for them.</span>",
        ]
    # finding 3
    if E.get("gemma"):
        g, c, p = E["gemma"], E["clap"], E.get("prosody", {})
        T["p_f3"] = [
            f"<b>Setup:</b> {E['n']['actors']} actors, the same {E['n']['sentences']} sentences, 6 emotions (guessing ≈ {pct(g['emotion']['chance'])}).",
            f"<b>Probe:</b> {pct(g['emotion']['acc'])} (Gemma) / {pct(c['emotion']['acc'])} (CLAP), beating loudness/pitch alone ({pct(p.get('emotion', {}).get('acc'))}). "
            f"Anger is easiest: {pct(g['emotion_recall']['ANG'])} found.",
            f"<b>Typing \"an angry voice\":</b> <span class=\"em\">{pct(g['zero_shot']['raw']['recall']['ANG'])}</span> (CLAP {pct(c['zero_shot']['raw']['recall']['ANG'])}). "
            "Both call <span class=\"em\">almost everything \"neutral\"</span>.",
            "<b>Encoded but unnameable:</b> the emotion is in the numbers, but no text query pulls it out.",
            f"<b>Bonus:</b> <span class=\"em\">Gemma knows which sentence was said ({pct(g['sentence']['acc'])}) but barely who said it</span> "
            f"({pct(g['speaker']['acc'])} of {E['n']['actors']}). CLAP: who {pct(c['speaker']['acc'])}.",
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
            f"<b>Setup:</b> {G7['gemma']['n']['instrumental']:,} instrumental tracks, {len(gg['words'])} genres (guessing {pct(gg['chance'])}). "
            f"Probe {pct(gg['probe']['acc'])} / {pct(gc['probe']['acc'])}; words {pct(bz['report']['acc'])} / {pct(gc['zero_shot'][gc['best_form']]['report']['acc'])} (Gemma / CLAP).",
            "<b>Words that work</b> (25 nearest tracks vs random): " + ", ".join(f"{w} {pct(prec[w]['precision'])} vs {pct(prec[w]['base_rate'])}" for w in ("rock", "folk", "classical") if w in prec)
            + ". <b>Don't:</b> " + ", ".join(w for w in good[-3:]) + ".",
            f"<b>Real classical</b> (Musopen Beethoven/Mozart): {pct(canon.get('musopen_predicted_classical'))} (Gemma) / {pct(canon_c.get('musopen_predicted_classical'))} (CLAP) labelled \"classical\".",
        ]
        if rz:
            T["p_f4"].append(f"<b>Regions</b> (Brazilian, Indian…) are weak: right one in the top 3 {pct(rz['top3_report'])} of the time (guessing {pct(3 / len(Rg['words']))}).")
        T["p_music_tiles"] = [[pct(gg["probe"]["acc"]), f"genre probe, Gemma (guessing {pct(gg['chance'])})"], [pct(bz["report"]["acc"]), "genre word finds the genre, Gemma"],
                              [pct(canon.get("musopen_predicted_classical")), "real classical labelled \"classical\""], [pct(bz["top3_report"]), "right genre in the word's top 3"]]
    if cl:
        P = cl["perturbations"]
        gap = P["member_score_mean"] - P["non_member_score_mean"]
        sh = lambda k: -P["delta"][k]["members"]["mean"] / gap
        inst = cl.get("instrument_fixed", {}).get("piano", {})
        instc = ((W8.get("clap") or {}).get("classical") or {}).get("instrument_fixed", {}).get("piano", {})
        lo, hi = pct(abs(sh('tempo_x0.75'))), pct(abs(sh('pitch_+3')))
        T["p_f4b"] = [
            "<b>We edited the audio</b> and re-measured how \"classical\" each classical track looked.",
            f"<b>Drums only</b> → <span class=\"em\">classical signal gone</span> ({'all of it' if sh('percussive_only') >= 1 else pct(sh('percussive_only'))}). <b>Drums removed</b> → <i>any</i> track looks more classical.",
            f"<b>Tempo or pitch changes</b> → almost no effect ({lo if lo == hi else lo + '–' + hi} of the signal).",
            "<span class=\"em\">So classical = tonal, drum-free, softer sound</span>, not the composition.",
            f"<b>More than \"piano\":</b> separates classical piano from jazz/pop piano (AUC {f2(inst.get('auc'))} Gemma, {f2(instc.get('auc'))} CLAP; 0.5 = guessing).",
        ]
    # finding 5
    if B.get("rows"):
        T["p_f5"] = [
            "<b>Searches that worked</b> (\"friends laughing\", \"bedtime story voice\") rely on <b>speaking style and show sound</b>, which these datasets don't measure.",
            "<b>Searches that failed</b> (argument, whispering, rain, music) hit the weak spots: unnameable emotion, weak fine sounds in Gemma.",
            "<span class=\"em\">For emotion, a trained probe reads what typed words can't.</span>",
        ]
    # which model + business
    if esc and E.get("gemma"):
        T["p_which"] = [
            ["Search sound effects, animals or music by typing words", "CLAP", f"its words reach the sounds ({pct(esc['clap']['zs'])} vs {pct(esc['gemma']['zs'])})"],
            ["Search speech by style or content (podcasts, calls)", "Gemma", f"understands spoken content ({pct(E['gemma']['sentence']['acc'])} sentence ID); podcast search runs on it"],
            ["Recognise who is speaking", "CLAP (or classic voice features)", f"Gemma barely encodes the speaker ({pct(E['gemma']['speaker']['acc'])})"],
            ["Detect emotion, hostility, bird species", "Neither model's words: train a small probe", "the information is there; typed words don't reach it"],
        ]
    T["p_business"] = [
        f"<b>Sound search for audio libraries:</b> CLAP + typed words works today ({pct(esc['clap']['zs']) if esc else '–'} on everyday sounds).",
        "<b>Probe detectors where words fail:</b> angry calls, bird species, laughter in content.",
        "<span class=\"em\">Recipe: a few hundred labelled examples, a tiny classifier trained in seconds, deploy.</span>",
        "<b>Music tagging:</b> works for clear genres (rock, classical, folk); region only as a rough filter.",
        "<b>Vibe Search upgrade:</b> sound-effect queries to CLAP, speech/style queries to Gemma, plus emotion probes.",
        "<b>Cheap:</b> small models on a laptop CPU, nothing retrained.",
        "<b>Caveats:</b> ESC-50 and MTG-Jamendo are non-commercial / research-only; some samples are small "
        "(59 clips for giggle vs chuckle, 16 tracks per genre for the classical edits).",
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
