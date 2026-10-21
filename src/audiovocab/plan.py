"""Phase 2: selection plan from metadata only. Counts every class/genre/region the plan would use, per dataset.

Writes reports/phase2_plan.json (all numbers) and data/selection/*.parquet (the exact candidate lists Phase 3 will
download). Deterministic: fixed seeds, fixed caps, no results involved.
"""
import ast
import json
import zipfile

import numpy as np
import pandas as pd
import yaml

from .config import ROOT, path

SEED = 0
CAPS = {
    "fsd_per_class": 40, "fsd_min_class": 25,
    "vocalsound_per_class": 100,
    "xc_per_species": 40, "xc_per_recordist": 2,
    "fma_genre_instr_tagged": 200, "fma_genre_untagged": 150, "fma_per_artist_genre": 3,
    "fma_region_per_artist": 5, "fma_calibration_each": 100,
    "musopen_per_work": 4,
}
MTG_SHARDS = ["53", "86"]
VOCAL_SUBGENRES = ["Rap", "Singer-Songwriter", "Jazz: Vocal", "Easy Listening: Vocal"]


def meta(*parts):
    return path("raw").joinpath("meta", *parts)


def taxonomy():
    return yaml.safe_load((ROOT / "data" / "taxonomy.yaml").read_text(encoding="utf-8"))


def _cap_per_group(df, group, n, seed=SEED):
    return df.sample(frac=1, random_state=seed).groupby(group, sort=False).head(n)


# ---------- sound datasets ----------

def esc50(tax):
    df = pd.read_csv(meta("esc50", "esc50.csv"))
    df["family"] = df.category.map(lambda c: tax["esc50"][c]["family"])
    return df.assign(dataset="esc50", label=df.category, group_id=df.src_file.astype(str), split=df.fold.astype(str))


def fsd50k(tax):
    gt = zipfile.ZipFile(meta("fsd50k", "FSD50K.ground_truth.zip"))
    md = zipfile.ZipFile(meta("fsd50k", "FSD50K.metadata.zip"))
    ev = pd.read_csv(gt.open("FSD50K.ground_truth/eval.csv"))
    info = json.load(md.open("FSD50K.metadata/eval_clips_info_FSD50K.json"))
    classes = tax["fsd50k"]
    labels = ev.labels.str.split(",").map(set)
    rows = []
    for c, spec in classes.items():
        rivals = {o for o, s in classes.items() if o != c and set(s["groups"]) & set(spec["groups"])}
        hit = labels.map(lambda L: c in L and not (L & rivals))
        for fname in ev.fname[hit]:
            i = info[str(fname)]
            rows.append({"clip": str(fname), "label": c, "family": spec["family"], "group_id": i["uploader"],
                         "licence": i["license"]})
    df = pd.DataFrame(rows)
    # a clip may qualify for two unrelated classes (e.g. Bark + Car); keep it once, under its rarer class
    freq = df.label.value_counts()
    df = df.assign(_f=df.label.map(freq)).sort_values("_f").drop_duplicates("clip").drop(columns="_f")
    df = _cap_per_group(df, "label", CAPS["fsd_per_class"])
    keep = df.label.value_counts()
    dropped = sorted(keep[keep < CAPS["fsd_min_class"]].index)
    df = df[~df.label.isin(dropped)]
    return df.assign(dataset="fsd50k", split="eval"), dropped


def xenocanto(tax):
    from . import meta as m
    rows = []
    for name, spec in tax["xenocanto"].items():
        g, s = spec["sci"].split()
        recs = []
        for q in ("A", "B"):
            j = m.xeno_canto(f"gen:{g} sp:{s} q:{q} len:5-60")
            recs += [dict(r, q=q) for r in j.get("recordings", [])]
        df = pd.DataFrame(recs)
        df = df[~df.lic.str.rstrip("/").str.split("/").str[-2].str.contains("nd")]
        df = df.sample(frac=1, random_state=SEED).sort_values("q", kind="stable")   # A first, random within
        df = df.groupby("rec", sort=False).head(CAPS["xc_per_recordist"]).head(CAPS["xc_per_species"])
        rows.append(pd.DataFrame({"clip": df.id.astype(str), "label": name, "family": spec["family"],
                                  "group_id": df.rec, "licence": df.lic, "quality": df.q, "length": df.length,
                                  "url": df.file}))
    return pd.concat(rows).assign(dataset="xenocanto")


def cremad(tax):
    tree = json.loads(meta("cremad", "tree.json").read_text(encoding="utf-8"))
    wavs = [x["path"] for x in tree["tree"] if x["path"].startswith("AudioWAV/") and x["path"].endswith(".wav")]
    p = pd.DataFrame([w.split("/")[1][:-4].split("_") for w in wavs], columns=["actor", "sentence", "emotion", "level"])
    return p.assign(dataset="cremad", clip=[w.split("/")[1][:-4] for w in wavs], label=p.emotion,
                    family="speech_emotion", group_id=p.actor)


# ---------- music ----------

def _fma_tables():
    z = zipfile.ZipFile(meta("fma", "fma_metadata.zip"))
    genres = pd.read_csv(z.open("fma_metadata/genres.csv"), index_col=0)
    tracks = pd.read_csv(z.open("fma_metadata/tracks.csv"), index_col=0, header=[0, 1])
    return genres, tracks


def fma(tax):
    genres, t = _fma_tables()
    title, top = genres.title.to_dict(), genres.top_level.to_dict()
    gall = t[("track", "genres_all")].map(ast.literal_eval)
    gmap, rmap = tax["music"]["genres"], tax["music"]["regions"]
    tops = gall.map(lambda g: {title[top[x]] for x in g if x in top} & set(gmap))
    instr_id = genres.index[genres.title == "Instrumental"][0]
    base = pd.DataFrame({
        "clip": t.index.astype(str), "artist": t[("artist", "id")].astype(str).values,
        "licence": t[("track", "license")].fillna("").values, "duration": t[("track", "duration")].values,
        "instr_tag": gall.map(lambda g: instr_id in g).values,
        "genre": tops.map(lambda s: gmap[next(iter(s))] if len(s) == 1 else None).values,
    }, index=t.index)
    region_ids = {i: rmap[title[i]] for i in genres.index if title[i] in rmap}
    # most specific region: deepest subgenre present (a child beats its parent)
    depth = {i: 0 for i in genres.index}
    for i in genres.index:
        p, d = genres.loc[i, "parent"], 0
        while p:
            d, p = d + 1, genres.loc[p, "parent"]
        depth[i] = d
    base["region"] = gall.map(lambda g: (lambda r: region_ids[max(r, key=depth.get)] if r else None)(
        [x for x in g if x in region_ids])).values
    base = base[base.duration >= 30]
    nd = base.licence.str.contains("NoDeriv|No Deriv", case=False)
    base["nd"] = nd

    # genre pools
    pools = []
    for g in sorted(set(gmap.values())):
        m = base[base.genre == g]
        tagged = _cap_per_group(m[m.instr_tag], "artist", CAPS["fma_per_artist_genre"]).head(CAPS["fma_genre_instr_tagged"])
        untagged = _cap_per_group(m[~m.instr_tag], "artist", CAPS["fma_per_artist_genre"]).head(CAPS["fma_genre_untagged"])
        pools += [tagged.assign(pool="genre"), untagged.assign(pool="genre")]
    regions = _cap_per_group(base[base.region.notna()], "artist", CAPS["fma_region_per_artist"]).assign(pool="region")
    cand = pd.concat(pools + [regions]).drop_duplicates("clip")

    # Whisper calibration set: artists outside the candidate pools, tagged instrumental vs vocal subgenres
    vocal_ids = set(genres.index[genres.title.isin(VOCAL_SUBGENRES)])
    outside = base[~base.artist.isin(set(cand.artist))]
    vocal = outside[gall.loc[outside.index].map(lambda g: bool(vocal_ids & set(g))).values & ~outside.instr_tag]
    instr = outside[outside.instr_tag]
    calib = pd.concat([
        _cap_per_group(instr, "artist", 1).head(CAPS["fma_calibration_each"]).assign(pool="calib_instrumental"),
        _cap_per_group(vocal, "artist", 1).head(CAPS["fma_calibration_each"]).assign(pool="calib_vocal"),
    ])
    return cand.assign(dataset="fma"), calib.assign(dataset="fma")


def musopen():
    j = json.loads(meta("musopen", "MusopenCollectionAsFlac.json").read_text(encoding="utf-8"))
    fl = pd.DataFrame([f for f in j["files"] if f["name"].endswith(".flac")])
    fl["work"] = fl.name.str.split("/").str[0]
    fl["composer"] = fl.work.str.split("_").str[0]
    fl = fl.sort_values("name").groupby("work").head(CAPS["musopen_per_work"])
    return fl.assign(dataset="musopen", clip=fl.name, label="classical", group_id=fl.work,
                     length=fl.length.astype(float), size=fl["size"].astype(int))


def mtg(tax):
    df = pd.read_csv(meta("mtg", "raw_30s_cleantags.tsv"), sep="\t", usecols=range(5), header=0,
                     names=["track", "artist", "album", "path", "duration"])
    tags = {}
    for line in meta("mtg", "raw_30s_cleantags.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        p = line.split("\t")
        tags[p[0]] = p[5:]
    df["folder"] = df.path.str.split("/").str[0]
    df = df[df.folder.isin(MTG_SHARDS)].copy()
    df["genres"] = df.track.map(lambda t: [x.split("---")[1] for x in tags[t] if x.startswith("genre---")])
    df["instruments"] = df.track.map(lambda t: [x.split("---")[1] for x in tags[t] if x.startswith("instrument---")])
    df["tag_instrumental"] = df.instruments.map(lambda i: bool(i) and not ({"voice", "vocals", "choir"} & set(i)))
    cells = {}
    for ins in tax["music"]["mtg_instruments"]:
        for g in tax["music"]["mtg_genres"]:
            m = df[df.tag_instrumental & df.instruments.map(lambda i: ins in i) & df.genres.map(lambda x: g in x)]
            cells[f"{ins}|{g}"] = {"tracks": len(m), "artists": int(m.artist.nunique())}
    return df.assign(dataset="mtg"), cells


# ---------- report ----------

def _count(df, col="label", group="group_id"):
    g = df.groupby(col)
    return {k: {"clips": int(len(v)), "groups": int(v[group].nunique())} for k, v in g}


def run():
    tax = taxonomy()
    sel = path("data") / "selection"
    sel.mkdir(parents=True, exist_ok=True)
    out = {"caps": CAPS, "mtg_shards": MTG_SHARDS}

    e = esc50(tax)
    out["esc50"] = {"clips": len(e), "classes": int(e.label.nunique()), "per_family": e.family.value_counts().to_dict()}
    f, f_dropped = fsd50k(tax)
    out["fsd50k"] = {"clips": len(f), "classes": int(f.label.nunique()), "dropped_lt_min": f_dropped,
                     "per_class": _count(f), "licences": f.licence.str.extract(r"licenses/([^/]+)|(zero)")
                     .bfill(axis=1)[0].value_counts().to_dict()}
    x = xenocanto(tax)
    out["xenocanto"] = {"clips": len(x), "per_species": _count(x),
                        "quality": x.quality.value_counts().to_dict(),
                        "licences": x.licence.str.rstrip("/").str.split("/").str[-2].value_counts().to_dict()}
    c = cremad(tax)
    out["cremad"] = {"clips": len(c), "actors": int(c.actor.nunique()), "sentences": int(c.sentence.nunique()),
                     "per_emotion": c.label.value_counts().to_dict()}
    out["vocalsound"] = {"clips_available": 21024, "speakers": 3365, "classes": list(tax["vocalsound"]),
                         "planned": CAPS["vocalsound_per_class"] * len(tax["vocalsound"]),
                         "note": "audio arrives via the bundle; per-class counts verified on import"}

    cand, calib = fma(tax)
    g = cand[cand.pool == "genre"]
    r = cand[cand.pool == "region"]
    out["fma"] = {
        "candidates": len(cand), "calibration": len(calib),
        "genre_pool": {k: {"tracks": int(len(v)), "artists": int(v.artist.nunique()),
                           "instr_tagged": int(v.instr_tag.sum()), "non_nd": int((~v.nd).sum())}
                       for k, v in g.groupby("genre")},
        "region_pool": {k: {"tracks": int(len(v)), "artists": int(v.artist.nunique()),
                            "instr_tagged": int(v.instr_tag.sum()), "non_nd": int((~v.nd).sum())}
                        for k, v in r.groupby("region")},
    }
    mo = musopen()
    out["musopen"] = {"movements": len(mo), "works": int(mo.work.nunique()), "composers": sorted(mo.composer.unique()),
                      "hours": round(float(mo.length.sum()) / 3600, 1), "gb": round(float(mo["size"].sum()) / 1e9, 2)}
    mt, cells = mtg(tax)
    out["mtg"] = {"tracks_in_shards": len(mt), "tag_instrumental": int(mt.tag_instrumental.sum()),
                  "artists": int(mt.artist.nunique()), "cells": cells}

    for name, df in [("esc50", e), ("fsd50k", f), ("xenocanto", x), ("cremad", c), ("fma", cand),
                     ("fma_calibration", calib), ("musopen", mo), ("mtg", mt)]:
        df.reset_index(drop=True).astype({k: str for k in df.columns if df[k].dtype == object}).to_parquet(
            sel / f"{name}.parquet", index=False)
    f_out = path("reports") / "phase2_plan.json"
    f_out.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
                     encoding="utf-8")
    return out
