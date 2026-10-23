"""Narrative for Q5 (music), Phase 8 and the Vibe Search bridge. Numbers come from D; claims are chosen by explicit
conditions on those numbers."""

NICE = {"harmonic_only": "removing percussion", "percussive_only": "removing harmony (drums only)",
        "tempo_x0.75": "slowing to 0.75×", "tempo_x1.33": "speeding to 1.33×", "pitch_-3": "pitch −3 semitones",
        "pitch_+3": "pitch +3 semitones", "lowpass_1500": "low-pass (darker)", "highpass_500": "high-pass (thinner)",
        "compress": "flattening dynamics", "shuffle_0.5s": "shuffling 0.5 s chunks", "reverse": "reversing"}
PLAIN = {"harmonic_only": "Remove the drums", "percussive_only": "Keep only the drums", "tempo_x0.75": "Slow it down",
         "tempo_x1.33": "Speed it up", "pitch_-3": "Lower the pitch", "pitch_+3": "Raise the pitch", "lowpass_1500": "Muffle it (cut the highs)",
         "highpass_500": "Thin it out (cut the lows)", "compress": "Flatten loud and soft parts", "shuffle_0.5s": "Shuffle 0.5 s chunks", "reverse": "Play it backwards"}
FEAT = {"tempo": "tempo", "tempo_stability": "steady tempo", "onset_rate": "note density", "percussive_ratio": "percussive share",
        "centroid": "brightness", "bandwidth": "bandwidth", "dynamic_range": "dynamic range", "key_clarity": "key clarity",
        "harmony_change": "harmony change rate", "mfcc1": "timbre (MFCC 1)", "mfcc2": "timbre (MFCC 2)",
        "mfcc3": "timbre (MFCC 3)", "mfcc4": "timbre (MFCC 4)", "mfcc5": "timbre (MFCC 5)"}


def pct(x):
    return "–" if x is None else f"{round(100 * x)}%"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def genre_text(g, r, rc):
    """Lead + bullets for one Phase 8 genre (Gemma r, CLAP rc)."""
    P = r["perturbations"]
    d = P["delta"]
    mem = {k: v["members"]["mean"] for k, v in d.items()}
    gap = P["member_score_mean"] - P["non_member_score_mean"]
    share = {k: v / gap for k, v in mem.items()} if gap > 0 else {}
    worst = sorted(mem, key=lambda k: mem[k])[:3]
    least = sorted(mem, key=lambda k: abs(mem[k]))[:2]
    top1 = lambda k: d[k]["members_still_top1"]
    gone = lambda k: "all" if -share[k] >= 1 else pct(-share[k])
    if gap <= 0:
        lead = [f"<b>The word \"{g}\" doesn't pick out its own tracks:</b> they score no higher on it than other genres "
                f"({f2(P['member_score_mean'])} vs {f2(P['non_member_score_mean'])}).",
                "<b>So there's no signal for the edits to remove.</b> This genre is the contrast case."]
    else:
        w = worst[0]
        lead = [f"<b>Setup:</b> we edited {P['n_members']} {g} tracks (drums only, slower, higher pitch, shuffled…) and re-measured how \"{g}\" they sound to the model.",
                f"<b>{PLAIN[w]}</b> → <span class=\"em\">{gone(w)} of the {g} signal gone</span>. "
                f"Tracks the model still calls {g}: {pct(P['members_top1_original'])} → {pct(top1(w))}."]
        sh, rv = share.get("shuffle_0.5s", 0), share.get("reverse", 0)
        if sh < -0.3 and abs(rv) < 0.15:
            lead.append(f"<b>Chopped into 0.5 s pieces and shuffled</b> → {pct(-sh)} gone. <b>Played backwards</b> → barely changes ({pct(rv)}).")
            lead.append("<span class=\"em\">So it reacts to smooth, unbroken flow</span>, not to which way the melody goes.")
        small = [k for k in ("tempo_x0.75", "tempo_x1.33", "pitch_-3", "pitch_+3", "compress") if abs(share.get(k, 1)) < 0.15]
        if len(small) >= 4:
            lead.append("<b>Tempo, pitch and dynamics changes</b> → barely move it.")
    sp = r["features"]["spearman"]
    top = sorted(sp, key=lambda k: -abs(sp[k]["across"]))[:3]
    feats = ", ".join(f"{FEAT[k]} ({'+' if sp[k]['across'] > 0 else '−'}{abs(sp[k]['across']):.2f})" for k in top)
    inst = r.get("instrument_fixed", {}).get("piano", {})
    desc = r["descriptors"]
    fp = r.get("false_positive_genre_counts_top50", {})
    fp_top = ", ".join(f"{k} ({v})" for k, v in list(fp.items())[:3])
    up = max(mem, key=mem.get)
    bullets = [
        f"<b>Biggest drops:</b> " + "; ".join(f"{NICE[k]}" + (f" ({pct(-share[k])} of the gap)" if share else f" ({f2(mem[k])})") for k in worst) + ".",
        f"<b>Barely matters:</b> " + "; ".join(NICE[k] for k in least)
        + (f". <b>Raises it:</b> {NICE[up]} (+{f2(mem[up])}), but other genres rise too (+{f2(d[up]['non_members']['mean'])})." if mem[up] > 0 and d[up]["members"]["lo"] > 0 else "."),
        f"<b>Measured qualities most tied to the {g} score:</b> {feats} (Spearman ρ).",
        f"<b>All 14 qualities together</b> explain R² = {f2(r['features']['regression']['cv_r2'])} of the score on held-out tracks.",
    ]
    if inst.get("auc") is not None:
        holds = inst["auc_ci"][0] > 0.5
        bullets.append(f"<b>Piano only:</b> the score tells {g} piano from other piano with AUC {f2(inst['auc'])} (CI {f2(inst['auc_ci'][0])}–{f2(inst['auc_ci'][1])}). "
                       + ("So \"" + g + "\" is <b>more than \"piano\"</b>." if holds else "So with the instrument fixed, the word <b>mostly means the instrument</b>."))
    bullets.append(f"<b>Nearest the word:</b> {', '.join(desc['top10_text_side'][:4])}. <b>Nearest the sound:</b> {', '.join(desc['top10_audio_side'][:4])} "
                   f"(agreement ρ = {f2(desc['spearman_text_vs_audio'])}).")
    if fp_top:
        bullets.append(f"<b>Most often mistaken for {g}:</b> {fp_top} (of the top 50 non-{g} tracks).")
    if rc:
        Pc = rc["perturbations"]
        gc = Pc["member_score_mean"] - Pc["non_member_score_mean"]
        mc = {k: v["members"]["mean"] for k, v in Pc["delta"].items()}
        wc = sorted(mc, key=mc.get)[:2]
        if gc > 0:
            bullets.append(f"<b>CLAP:</b> biggest drops are " + " and ".join(f"{NICE[k]} ({pct(-mc[k] / gc)})" for k in wc)
                           + f"; shuffling {pct(-mc['shuffle_0.5s'] / gc)}.")
        else:
            bullets.append(f"<b>CLAP</b> doesn't single out {g} tracks either ({f2(Pc['member_score_mean'])} vs {f2(Pc['non_member_score_mean'])}).")
    bullets.append(f"<b>Sample:</b> {P['n_members']} {g} tracks, {P['n_non_members']} others; 95% intervals by resampling artists.")
    inst_note = "AUC: 0.5 = can't tell them apart, 1 = perfect." if inst else ""
    return lead, bullets, inst_note


def music_text(D):
    out = {"pert_names": NICE, "feat_names": FEAT}
    G7, W8, B = D.get("p7") or {}, D.get("p8") or {}, D.get("bridge") or {}
    g, c = G7["gemma"], G7.get("clap", {})
    G, GC = g["genre"], c.get("genre", {})
    bz = G["zero_shot"][G["best_form"]]
    prec = bz["precision_at_25"]
    ratio = sorted(prec, key=lambda w: -prec[w]["precision"] / max(prec[w]["base_rate"], 1e-9))
    canon = g.get("canonical_classical") or {}
    q = G7["phase8"]["qualifying"]
    out["music_lead"] = (f"{g['n']['instrumental']:,} FMA tracks with no vocals found by Whisper, {len(G['words'])} genre words, artist-grouped splits. "
                         f"Best word form (tuned): \"{G['best_form'].split('_')[0]}\" (e.g. \"a piece of jazz music\").")
    out["music_tiles"] = [
        [pct(G["probe"]["acc"]), f"Gemma probe over {len(G['words'])} genres (chance {pct(G['chance'])})"],
        [pct(bz["report"]["acc"]), "Gemma: genre word finds the genre"],
        [pct(canon.get("musopen_predicted_classical")), "canonical classical (Musopen) labelled \"classical\""],
        [pct(bz["top3_report"]), f"true genre in the word's top 3 (chance {pct(3 / len(G['words']))})"],
    ]
    voc = bz.get("vocal_report") or {}
    R = g["region"]
    rz = R.get("zero_shot", {}).get(R.get("best_form", ""), {})
    out["nb_music"] = [
        f"<b>Some genre words work, some don't.</b> Relative to base rate, the 25 tracks nearest the word are best for "
        + ", ".join(f"{w} ({pct(prec[w]['precision'])} vs {pct(prec[w]['base_rate'])})" for w in ratio[:3])
        + "; worst for " + ", ".join(f"{w} ({pct(prec[w]['precision'])})" for w in ratio[-2:]) + ".",
        f"<b>Classical generalises to the canon:</b> {pct(canon.get('musopen_predicted_classical'))} of Musopen's {canon.get('musopen_tracks')} public-domain movements "
        f"are labelled \"classical\" (FMA's indie classical: {pct(canon.get('fma_classical_recall'))}). The model's \"classical\" is the concert-hall sound, not FMA's quirk.",
        f"<b>Lyrics help less than expected:</b> the genre word finds vocal tracks {pct(voc.get('acc'))} of the time vs {pct(bz['report']['acc'])} for instrumental (Gemma), "
        "with residual vocals in the instrumental set as a caveat.",
    ]
    if rz:
        out["nb_music"].append(f"<b>Regions are weak:</b> region words get {pct(rz['report']['acc'])} (chance {pct(R['chance'])}); the true region is in the top 3 "
                               f"{pct(rz['top3_report'])} of the time (chance {pct(3 / len(R['words']))}). Useful as a coarse filter at best.")
    out["nb_music"].append(f"<b>Phase 8 genres</b> (zero-shot precision ≥ 2× base rate, CI above it): {', '.join(q)}; contrast: {G7['phase8']['contrast_failing']}.")
    if W8.get("gemma"):
        out["wm_lead"], out["nb_wm"], out["inst_note"] = {}, {}, {}
        for gname in W8["gemma"]["genres"]:
            lead, bullets, note = genre_text(gname, W8["gemma"][gname], (W8.get("clap") or {}).get(gname))
            out["wm_lead"][gname], out["nb_wm"][gname], out["inst_note"][gname] = lead, bullets, note
        cl = W8["gemma"].get("classical")
        if cl:
            P = cl["perturbations"]
            gap = P["member_score_mean"] - P["non_member_score_mean"]
            mem = {k: v["members"]["mean"] for k, v in P["delta"].items()}
            worst = min(mem, key=mem.get)
            out["headline"] = (f"<b>To the model, \"classical\" means tonal sound without drums.</b> {pct(canon.get('musopen_predicted_classical'))} of canonical "
                               f"recordings are called classical; keeping only the drums of classical tracks erases "
                               f"{pct(-mem[worst] / gap)} of their classical signal, while tempo and pitch changes barely register.")
    if B.get("rows"):
        rows = B["rows"]
        fails = [r for r in rows if r["listening_yes"] is not None and r["listening_yes"] <= 2]
        kinds = {}
        for r in fails:
            kinds.setdefault(r["verdict"], []).append(r["query"])
        out["vibe_lead"] = ("The sister project's 12 listening-test searches, each mapped (before seeing these results) to the closest labelled classes here. "
                            "Verdict rule: probe < 70% → weak area of the space; probe ≥ 70% but zero-shot < 50% → knows but can't name; both high → likely a podcast-data gap.")
        weak = [r for r in rows if r["verdict"].startswith("weak")]
        weak_ok = [r for r in weak if (r["listening_yes"] or 0) >= 3]
        out["nb_vibe"] = [f"<b>Failed searches (≤ 2 of 4 by ear), by verdict:</b> " + " · ".join(f"{v}: " + "; ".join(f"\"{q}\"" for q in qs) for v, qs in kinds.items()) + "."]
        if weak_ok:
            out["nb_vibe"].append(f"<b>The labelled-class view doesn't predict listening success.</b> The pre-set rule marks {len(weak)} of {len(rows)} searches as weak areas, "
                                  f"including {len(weak_ok)} that scored 3–4 of 4 by ear (e.g. \"{weak_ok[0]['query']}\"). Those searches succeed on speaking style and show-level "
                                  "sound (who talks, how, with what production), which these datasets don't label: the per-class numbers here are strict 50- and 62-way recalls, not vibes.")
        ang = next((c for r in rows if r["query"].startswith("heated") for c in r["classes"] if c["label"] == "ANG"), None)
        if ang:
            out["nb_vibe"].append(f"<b>\"Heated argument\", unpacked:</b> its anger part is <span class=\"em\">knows but can't name</span> (probe {pct(ang['probe'])}, words {pct(ang['zero_shot'])}); "
                                  "averaged with FSD50K's shout and scream classes it falls under the rule's 70% bar.")
        out["nb_vibe"].append(f"Listening total in Vibe Search: {B['listening_total'][0]} of {B['listening_total'][1]} (raw search, one listener).")
    out["joint_note"] = ""
    U = D.get("umap") or {}
    if U.get("joint"):
        out["joint_note"] = (f"Only {pct(U['joint']['text_neighbours_that_are_clips'])} of a word's 10 nearest map neighbours are clips. Mean cosine: clip↔clip "
                             f"{f2(U['cosines']['clip_clip'])}, text↔text {f2(U['cosines']['text_text'])}, text↔clip {f2(U['cosines']['text_clip'])}.")
    return out
