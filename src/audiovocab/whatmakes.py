"""Phase 8 (Q5 part 2): what is the model responding to when it ties music to a genre word?

For each genre fixed in reports/phase7_genre.json ("qualifying" + "contrast_failing"):
  1. measured features vs genre score: Spearman within and across genres; standardised regression (CV R^2);
  2. perturbations (perturb.py) on ~150 non-ND instrumental FMA clips: change in genre score, members vs not;
  3. instrument held fixed (MTG-Jamendo piano tracks across genres): does the genre score still separate them?
  4. descriptor words ranked by closeness to the genre word (text side) and to the genre's audio centroid;
  5. false positives: non-members with the highest genre scores.
Genre score = cos(clip, genre word) - mean cos(clip, other genre words), best word form from Phase 7.
Writes reports/phase8_whatmakes.json.
"""
import json

import librosa
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from . import centering, data, embed, genre, metrics
from .audio import decode
from .config import cfg, path
from .perturb import PERTURBATIONS

N_PERTURB = 150
FEATURES = ["tempo", "tempo_stability", "onset_rate", "percussive_ratio", "centroid", "bandwidth", "dynamic_range",
            "key_clarity", "harmony_change", "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MTG_GENRE = {"classical": "classical", "jazz": "jazz", "electronic": "electronic", "pop": "pop", "rock": "rock",
             "folk": "folk", "hip-hop": "hiphop", "blues": "blues", "country": "country", "soul": "soul",
             "easy listening": "easylistening"}


def music_features(rows):
    f = path("data") / "music_features.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["clip_id"])
    todo = rows[~rows.clip_id.isin(have.clip_id)]
    out = []
    for i, r in enumerate(todo.itertuples(), 1):
        sr = 22050
        x = decode(path("raw") / r.source, sr, start=float(r.start) or None, dur=float(r.dur))
        if len(x) < sr:
            continue
        tempo, beats = librosa.beat.beat_track(y=x, sr=sr)
        bt = librosa.frames_to_time(beats, sr=sr)
        ioi = np.diff(bt)
        h, p = librosa.effects.hpss(x)
        rms = librosa.feature.rms(y=x)[0]
        db = 20 * np.log10(rms + 1e-6)
        chroma = librosa.feature.chroma_stft(y=h, sr=sr)
        cm = chroma.mean(1)
        prof = [np.corrcoef(cm, np.roll(P, k))[0, 1] for P in (MAJOR, MINOR) for k in range(12)]
        hop = int(0.5 * sr / 512)
        cs = chroma[:, ::max(hop, 1)]
        cs = cs / (np.linalg.norm(cs, axis=0, keepdims=True) + 1e-9)
        mf = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=6).mean(1)
        out.append({"clip_id": r.clip_id, "tempo": float(np.atleast_1d(tempo)[0]),
                    "tempo_stability": float(1 - min(1, np.std(ioi) / np.mean(ioi))) if len(ioi) >= 3 else np.nan,
                    "onset_rate": len(librosa.onset.onset_detect(y=x, sr=sr)) / (len(x) / sr),
                    "percussive_ratio": float((p ** 2).sum() / ((h ** 2).sum() + (p ** 2).sum() + 1e-12)),
                    "centroid": float(np.log(librosa.feature.spectral_centroid(y=x, sr=sr).mean() + 1)),
                    "bandwidth": float(np.log(librosa.feature.spectral_bandwidth(y=x, sr=sr).mean() + 1)),
                    "dynamic_range": float(np.percentile(db, 95) - np.percentile(db, 10)),
                    "key_clarity": float(np.nanmax(prof)),
                    "harmony_change": float(np.mean(1 - (cs[:, 1:] * cs[:, :-1]).sum(0))) if cs.shape[1] > 1 else np.nan,
                    **{f"mfcc{k}": float(mf[k]) for k in range(1, 6)}})
        if i % 200 == 0:
            pd.concat([have, pd.DataFrame(out)]).to_parquet(f, index=False)
            print(f"features {i}/{len(todo)}", flush=True)
    allf = pd.concat([have, pd.DataFrame(out)], ignore_index=True)
    allf.to_parquet(f, index=False)
    return allf.set_index("clip_id")


def setup(model):
    g7 = json.loads((path("reports") / "phase7_genre.json").read_text())
    prefix = g7[model]["prefix"]
    words = g7[model]["genre"]["words"]
    form, tag = g7["gemma"]["genre"]["best_form"].rsplit("_", 1)  # the Gemma choice is used for both models
    df, X = genre.music_table(model)
    W = genre.word_vectors(model, prefix, "genre", words, form)
    t, T = data.texts(model, prefix)
    gen = T[(t.kind == "generic").values].mean(0)
    a_mean = X[((df.fold == data.tune_fold()) & (df.vocal == False)).values].mean(0)  # noqa: E712
    if tag == "centred":
        X, W = centering.center(X, a_mean), centering.center(W, gen)
    return g7, prefix, words, form, tag, df, X, W, a_mean, gen


def perturb_set(df, words, targets, seed=0):
    inst = df[(df.vocal == False) & df.genre.isin(words) & (df.nd.astype(str) != "True")]  # noqa: E712
    rng = np.random.default_rng(seed)
    per = N_PERTURB // (len(targets) + 1)
    parts = []
    for g in targets:
        m = inst[inst.genre == g]
        parts.append(m.iloc[rng.permutation(len(m))[:per]])
    rest = inst[~inst.genre.isin(targets)]
    parts.append(rest.iloc[rng.permutation(len(rest))[:N_PERTURB - sum(len(p) for p in parts)]])
    return pd.concat(parts)


def run(models=("gemma", "clap")):
    A = cfg()["analysis"]
    out = {"perturbations": {k: v["what"] for k, v in PERTURBATIONS.items()}}
    for model in models:
        g7, prefix, words, form, tag, df, X, W, a_mean, gen = setup(model)
        targets = [w for w in g7["phase8"]["qualifying"] + [g7["phase8"]["contrast_failing"]] if w]
        inst = (df.vocal == False).values & df.genre.isin(words).values  # noqa: E712
        feats = music_features(df[inst])
        res = {"genres": targets, "form": form, "centred": tag == "centred"}
        # perturbation set and embeddings
        P = perturb_set(df, words, targets)
        m = embed.load(model)
        keys0 = [embed.clip_key(r) for r in P.to_dict("records")]
        E0 = embed.embed_cached(m, keys0, embed.load_window)
        Ep = {}
        for name in PERTURBATIONS:
            Ep[name] = embed.embed_cached(m, [{**k, "perturb": name} for k in keys0], embed.load_window)
            print(model, "perturb", name, flush=True)
        del m
        if tag == "centred":
            E0 = centering.center(E0, a_mean)
            Ep = {k: centering.center(v, a_mean) for k, v in Ep.items()}
        for g in targets:
            j = words.index(g)
            sc = genre.genre_score(X, W, j)
            r = {}
            # 1. features
            sub = df[inst].assign(score=sc[inst]).join(feats, on="clip_id")
            sub = sub.dropna(subset=FEATURES)
            corr = {}
            for ftr in FEATURES:
                within = sub[sub.genre == g]
                corr[ftr] = {"across": round(float(spearmanr(sub[ftr], sub.score).statistic), 4),
                             "within": round(float(spearmanr(within[ftr], within.score).statistic), 4)}
            Z = StandardScaler().fit_transform(sub[FEATURES].values)
            yv = sub.score.values
            pred = cross_val_predict(LinearRegression(), Z, yv, groups=sub.group_id.values, cv=GroupKFold(5))
            coef = LinearRegression().fit(Z, yv).coef_
            r["features"] = {"spearman": corr, "n": int(len(sub)),
                             "regression": {"cv_r2": round(float(1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()), 4),
                                            "coef_std": {f: round(float(c), 4) for f, c in zip(FEATURES, coef)}},
                             "member_vs_rest_mean": {f: [round(float(sub[sub.genre == g][f].mean()), 3),
                                                         round(float(sub[sub.genre != g][f].mean()), 3)] for f in FEATURES}}
            # 2. perturbations
            s0 = genre.genre_score(E0, W, j)
            mem = (P.genre == g).values
            pr = {}
            for name in PERTURBATIONS:
                d = genre.genre_score(Ep[name], W, j) - s0
                gm = P.group_id.values
                pr[name] = {"members": dict(zip(("mean", "lo", "hi"), [round(v, 4) for v in metrics.bootstrap_ci(d[mem], gm[mem], n=A["bootstrap"])])),
                            "non_members": dict(zip(("mean", "lo", "hi"), [round(v, 4) for v in metrics.bootstrap_ci(d[~mem], gm[~mem], n=A["bootstrap"])])),
                            "members_still_top1": round(float((np.argmax(Ep[name][mem] @ W.T, 1) == j).mean()), 4)}
            r["perturbations"] = {"n_members": int(mem.sum()), "n_non_members": int((~mem).sum()),
                                  "member_score_mean": round(float(s0[mem].mean()), 4),
                                  "non_member_score_mean": round(float(s0[~mem].mean()), 4),
                                  "members_top1_original": round(float((np.argmax(E0[mem] @ W.T, 1) == j).mean()), 4),
                                  "delta": pr}
            # 3. instrument held fixed (MTG piano)
            r["instrument_fixed"] = instrument_fixed(model, prefix, words, form, tag, g, a_mean, gen, A)
            # 4. descriptors
            r["descriptors"] = descriptors(model, prefix, W[j], X[inst & (df.genre == g).values], g)
            # 5. false positives
            nonm = np.flatnonzero(inst & (df.genre != g).values)
            top = nonm[np.argsort(-sc[nonm])[:12]]
            meta = fma_meta()
            r["false_positives"] = [{"track": df.item_id.iloc[i], "genre": df.genre.iloc[i],
                                     "region": df.region.iloc[i], "score": round(float(sc[i]), 4),
                                     **meta.get(int(df.item_id.iloc[i]), {}),
                                     **({k: round(float(feats.loc[df.clip_id.iloc[i], k]), 3) for k in ("tempo", "percussive_ratio", "centroid")}
                                        if df.clip_id.iloc[i] in feats.index else {})} for i in top]
            r["false_positive_genre_counts_top50"] = pd.Series(df.genre.values[nonm[np.argsort(-sc[nonm])[:50]]]).value_counts().to_dict()
            res[g] = r
        out[model] = res
    (path("reports") / "phase8_whatmakes.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def fma_meta():
    import zipfile
    z = zipfile.ZipFile(path("raw") / "meta" / "fma" / "fma_metadata.zip")
    t = pd.read_csv(z.open("fma_metadata/tracks.csv"), index_col=0, header=[0, 1])
    return {int(i): {"title": str(a), "artist": str(b), "fma_genres": str(c)}
            for i, a, b, c in zip(t.index, t[("track", "title")], t[("artist", "name")], t[("track", "genres")])}


def instrument_fixed(model, prefix, words, form, tag, g, a_mean, gen, A):
    mg = MTG_GENRE.get(g)
    df, X = data.subset(model, "mtg")
    flags = pd.read_parquet(path("data") / "vocal_flags.parquet")
    flags = flags[flags.dataset == "mtg"].set_index("item_id")
    df = df.assign(vocal=df.item_id.map(flags.vocal))
    W = genre.word_vectors(model, prefix, "genre", words, form)
    if tag == "centred":
        X, W = centering.center(X, a_mean), centering.center(W, gen)
    sc = genre.genre_score(X, W, words.index(g))
    ok = (df.instr_tag.astype(str) == "True").values & (df.vocal == False).values  # noqa: E712
    out = {}
    for ins in cfg_instruments():
        has = df.instruments.astype(str).str.contains(f"'{ins}'").values & ok
        if mg is None:
            continue
        gen_ = df.mtg_genres.astype(str)
        mem = has & gen_.str.contains(f"'{mg}'").values
        non = has & ~gen_.str.contains(f"'{mg}'").values
        if mem.sum() < 10 or non.sum() < 10:
            out[ins] = {"members": int(mem.sum()), "non_members": int(non.sum()), "note": "too few tracks"}
            continue
        from sklearn.metrics import roc_auc_score
        yy = np.r_[np.ones(mem.sum()), np.zeros(non.sum())]
        ss = np.r_[sc[mem], sc[non]]
        gg = np.r_[df.group_id.values[mem], df.group_id.values[non]]
        rng = np.random.default_rng(0)
        ug = np.unique(gg)
        boots = []
        for _ in range(A["bootstrap"]):
            pick = np.isin(gg, rng.choice(ug, len(ug)))
            if len(np.unique(yy[pick])) == 2:
                boots.append(roc_auc_score(yy[pick], ss[pick]))
        other_genres = pd.Series([x for s in gen_[non] for x in eval(s)]).value_counts().head(6).to_dict()
        out[ins] = {"members": int(mem.sum()), "non_members": int(non.sum()), "auc": round(float(roc_auc_score(yy, ss)), 4),
                    "auc_ci": [round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)],
                    "non_member_genres": other_genres}
    return out


def cfg_instruments():
    import yaml
    from .config import ROOT
    return yaml.safe_load((ROOT / "data" / "taxonomy.yaml").read_text(encoding="utf-8"))["music"]["mtg_instruments"]


def descriptors(model, prefix, wvec, Xg, g):
    t, T = data.texts(model, prefix)
    m = (t.kind == "descriptor").values
    D, names = T[m], t.text[m].tolist()
    c = Xg.mean(0)
    c /= np.linalg.norm(c)
    text_side = D @ wvec
    audio_side = D @ c
    rt, ra = np.argsort(-text_side), np.argsort(-audio_side)
    return {"spearman_text_vs_audio": round(float(spearmanr(text_side, audio_side).statistic), 4),
            "top10_text_side": [names[i] for i in rt[:10]], "top10_audio_side": [names[i] for i in ra[:10]],
            "bottom5_audio_side": [names[i] for i in ra[-5:]],
            "scores": {n: [round(float(a), 4), round(float(b), 4)] for n, a, b in zip(names, text_side, audio_side)}}
