# Dataset licences

Verified 2026-10-22 from each dataset's own licence text (files in `data/raw/meta/`, fetched by `audiovocab.meta`).
This is a working summary for an internal prototype, not legal advice. **Flag to Anand before any client-facing
or commercial use:** several sources are non-commercial (NC), and this is an internal study at a company.

## How licences are applied in this project

- **Analysis** (embedding, probes, kNN, statistics) runs locally on all chosen clips. Only derived numbers and
  vectors leave the machine; no audio is redistributed.
- **Perturbations** (Phase 8: filtering, time-stretch, shuffling) create altered copies of audio. They are used only
  on clips whose licence allows adaptation (no **ND**), and the altered audio is never shared.
- **Public page playback:** only clips with **CC0, Public Domain Mark, CC BY or CC BY-SA** licences, each with its
  attribution next to the player. Everything else is shown as labels, not sound. Local playback is fine.
- **Excluded outright:** ND recordings from xeno-canto (per the README). Datasets with unclear licences (GTZAN), and
  YouTube-ID datasets (AudioSet, MusicCaps).

## Per dataset

| Dataset | Licence (verified) | Source of truth | Used? | Page playback |
|---|---|---|---|---|
| **ESC-50** | **CC BY-NC 3.0** for the dataset; the ESC-10 subset is CC BY 3.0. Per-clip Freesound attributions are in the repo's `LICENSE` (2,059 lines). | `ESC-50/LICENSE`, README "License" | Yes, all 2,000 clips | Only ESC-10 clips (CC BY), with attribution |
| **FSD50K** | Per clip: eval set CC0 4,914 / CC BY 3,489 / CC BY-NC 1,425 / Sampling+ 403. The dataset as a whole is CC BY 4.0. | Zenodo record 4060432 description; `eval_clips_info_FSD50K.json` | Yes, a class subset of the **eval** split | CC0 and CC BY clips only, credited to the Freesound uploader |
| **VocalSound** | **CC BY-SA 4.0** | `vocalsound/README.md` (licence badge and statement) | Yes (via bundle) | Yes, with attribution (Gong et al., ICASSP 2022) |
| **xeno-canto** | Per recording: mostly CC BY-NC-SA 4.0; some CC0, BY, BY-SA, BY-NC; ND excluded | API v3 `lic` field per recording | Yes, non-ND only | CC0, BY and BY-SA recordings only; NC ones stay internal |
| **CREMA-D** | **ODbL 1.0** (database) + **DbCL 1.0** (contents) | `CREMA-D/LICENSE.txt` | Yes, all 7,442 clips | Allowed under DbCL, with attribution. Faces aren't involved (audio only). |
| RAVDESS | CC BY-NC-SA 4.0 | Zenodo record 1188976 | **Not used**: it was only a backup, and CREMA-D is reachable | — |
| **FMA** (`fma_large`, 30 s clips) | Metadata **CC BY 4.0**. Audio: per track, as chosen by the artist. 40% of tracks are NoDerivatives, and most are NC. | `fma/README.md` "Acknowledgments and Licenses"; `tracks.csv` `track.license` | Yes, a selected subset | CC BY / CC BY-SA / CC0 tracks only, credited to the artist |
| **MTG-Jamendo** | Metadata **CC BY-NC-SA 4.0**. Audio: per track CC. **"Made available solely for non-commercial research and academic use"**; commercial use needs Jamendo's permission. | `mtg/README.md` "License" | Yes (2 shards via bundle), internal analysis only | **No.** Labels only, because of the research-only clause. |
| **Musopen** (archive.org "Musopen Collection as FLAC") | **Public Domain Mark 1.0** | archive.org item metadata `licenseurl` | Yes, canonical classical check | Yes, crediting Musopen |
| Song Describer | CC BY-SA 4.0 (captions); audio per track | Zenodo record 10072001 | **Not used** in v1 (optional in the README) | — |
| GTZAN | Unclear (copyrighted recordings, no licence) | — | **Excluded** (README rule) | — |
| AudioSet, MusicCaps | YouTube ID lists | — | **Excluded** (README rule) | — |

## Attribution text (for the page footer)

- **ESC-50:** K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015. CC BY-NC
  3.0. Clips from Freesound.org; per-clip credits in the dataset's LICENSE file.
- **FSD50K:** E. Fonseca, X. Favory, J. Pons, F. Font, X. Serra, "FSD50K: an Open Dataset of Human-Labeled Sound
  Events", IEEE/ACM TASLP 2022. Dataset CC BY 4.0; clips from Freesound.org under their own CC licences.
- **VocalSound:** Y. Gong, J. Yu, J. Glass, "VocalSound: A Dataset for Improving Human Vocal Sounds Recognition",
  ICASSP 2022. CC BY-SA 4.0.
- **xeno-canto:** recordings from xeno-canto.org, each credited to its recordist with its CC licence (XC id shown).
- **CREMA-D:** H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, R. Verma, "CREMA-D: Crowd-sourced
  Emotional Multimodal Actors Dataset", IEEE Trans. Affective Computing 2014. ODbL 1.0 / DbCL 1.0.
- **FMA:** M. Defferrard, K. Benzi, P. Vandergheynst, X. Bresson, "FMA: A Dataset for Music Analysis", ISMIR 2017.
  Metadata CC BY 4.0; audio under each artist's licence.
- **MTG-Jamendo:** D. Bogdanov, M. Won, P. Tovstogan, A. Porter, X. Serra, "The MTG-Jamendo Dataset for Automatic
  Music Tagging", ICML 2019 ML4MD workshop. Metadata CC BY-NC-SA 4.0; non-commercial research use.
- **Musopen:** recordings by Musopen (musopen.org), Public Domain Mark, via the Internet Archive.
