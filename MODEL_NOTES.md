# Model notes: how EmbeddingGemma 2 and CLAP handle a clip and a sentence

Every claim cites its source. Paths are relative to the sister's model folder
(`podcast-vibe-search/models/google--embeddinggemma-2/`, written **[M]**) or to `.venv/Lib/site-packages/` (written
**[T]** for `transformers/` and **[ST]** for `sentence_transformers/`), with transformers 5.20.2 and
sentence-transformers 6.1.0. "Model card" means `[M] README.md`. **Measured** means verified in this repo; the result
file is named. Anything the sources don't say is marked **not stated**.

## EmbeddingGemma 2

### Audio path

| Question | Answer | Source |
|---|---|---|
| Sample rate | **16 kHz mono.** | Model card l.254 ("mono at 16 kHz"); `[M] preprocessor_config.json` `sampling_rate: 16000` |
| Features | **128-bin log-mel spectrogram**, 0–8 kHz, 20 ms Hann frames (320 samples), **10 ms hop** (160), FFT 512, magnitude (not power) → HTK mel → `log(mel + 0.001)`. No per-bin normalisation, no pre-emphasis, no dither. A half-frame of zeros is prepended ("semicausal"). | `[M] preprocessor_config.json`; `[T] models/gemma4/feature_extraction_gemma4.py` l.181–182 (padding), l.203–208 (rFFT → \|·\| → mel → log) |
| Loudness | Not normalised anywhere: a gain of g dB adds a constant to the log-mel (`log(a·x) = log a + log x`, before the floor). Whether the network is invariant is an empirical question; see "Behaviour tests". | Same lines; no normalisation step exists in the extractor |
| **Maximum length** | **30 s. Longer audio is silently truncated** (only the first 30 s is used; no chunking, no warning). The feature extractor defaults to `max_length=480_000` samples with `truncation=True`. sentence-transformers passes audio with only `padding=True`, so the default applies. **Measured:** 30, 31, 40 and 60 s inputs all give 750 audio tokens (`reports/phase1_audio_tokens.json`). The behaviour test confirms 40 s and 60 s embeddings are identical to 30 s. | `[T] feature_extraction_gemma4.py` l.230–231; `[ST] base/modules/transformer.py` l.1259–1261 |
| The model card says "~327 seconds" | That's the 8,192-token context divided by 25 tokens/s. **It doesn't hold in the transformers/sentence-transformers path**, which truncates at 30 s first. | Model card l.237 vs the above |
| `audio_seq_length: 280` in `processor_config.json` | **Not a limit in this path.** It's used only by `_compute_audio_num_tokens` (a placeholder-count helper for vLLM). The real token count comes from the feature mask, uncapped. **Measured:** 11.2 s gives 280 tokens, and 20 s gives 500. | `[M] processor_config.json` l.3; `[T] models/embedding_gemma2/processing_embedding_gemma2.py` l.320–331 (real path), l.415–453 (capped helper) |
| Tokens per second | **25** (one per 40 ms): 10 ms mel frames → two stride-2 convolutions (×4). **Measured:** 1 s gives 25, 10 s gives 250, and 20 s gives 500. | `processing_embedding_gemma2.py` l.126–129, l.439–450; `phase1_audio_tokens.json` |
| Audio encoder | 12-layer conformer ("gemma4_audio"), hidden size 1024, output 1536. Attention is **local and causal**: chunks of 12 tokens, each seeing the 12 previous tokens and no future ones (about 1 s of context per layer). Long-range mixing therefore happens mostly in the text backbone. | `[M] config.json` audio_config (l.346–348, 356, 369, 372); `[T] models/gemma4/modeling_gemma4.py` l.271–274 |
| Into the shared model | Each audio token → RMSNorm (no scale) → linear 1536→512, then **scattered into the text sequence** in place of `<|audio|>` placeholders. The sequence is `<bos><boa>` + N audio tokens + `<eoa><eos>`, so a 20 s clip is 504 positions. | `[T] models/embedding_gemma2/modeling_embedding_gemma2.py` l.575–591, l.819–837; `phase1_audio_tokens.json` (`seq_len` = N + 4) |
| Backbone | The same 24-layer Gemma text transformer (hidden size 512) that encodes text. Attention is **bidirectional**: a 512-position sliding window (≈ 20 s of audio) on most layers, and full attention on every 6th layer (4 of 24). So every position can reach the whole clip. | `modeling_embedding_gemma2.py` l.539–541, l.839–845; `[M] config.json` text_config `sliding_window: 512`, `layer_types` |
| Pooling | **Mean over all positions**, including `<bos>/<boa>/<eoa>/<eos>`. Each position is projected 512→768 first (equivalent to projecting after the mean), then the vector is L2-normalised. | `[M] 1_Pooling/config.json` (`mean`, `include_prompt: true`); `modeling_embedding_gemma2.py` l.504–505; `[M] modules.json` (Normalize) |
| Output | 768-d, L2-normalised. Matryoshka-trained, so it can be truncated to 512/256/128 (we use 768). | Model card l.42–45, l.167–174 |
| Prefix for audio | **None.** "Pass images, video, and audio without any prefix." | Model card l.131 |
| Precision | bfloat16 or float32; **never float16** (NaNs). We use float32 because bf16 is about 12× slower on this CPU (sister's measurement). | Model card l.188–194; `ENVIRONMENT.md` |

### Text path

| Question | Answer | Source |
|---|---|---|
| Tokeniser | Gemma SentencePiece, vocabulary 262,144. | Model card l.60; `[M] config.json` `vocab_size` |
| **Task prefixes** | **Queries:** `task: search result \| query: {q}` (prompt name `SearchQuery`). **Documents:** `title: {title or none} \| text: {content}` (`Document`). **Symmetric tasks:** `task: classification \| query:`, `task: clustering \| query:`, `task: sentence similarity \| query:`. "Omitting it still works but reduces precision." | Model card l.131–152; `[M] config_sentence_transformers.json` `prompts` |
| Default prompt | **None** (`default_prompt_name: null`): with no `prompt_name`, no prefix is added. | `[M] config_sentence_transformers.json` |
| Prefix tokens in the vector | Yes. Mean pooling includes the prompt tokens (`include_prompt: true`), so the prefix contributes to the embedding. | `[M] 1_Pooling/config.json` |
| Max length | 8,192 tokens (model card). The tokenizer itself sets no limit (`model_max_length` is unset, ~1e30); sentence-transformers caps it at the model's position limit. Irrelevant for our short texts. | Model card l.46, l.234; `[M] tokenizer_config.json` |
| Which prefix to use here | Audio is the "document" side of a retrieval task. So text queries use **`SearchQuery`**, as the sister did and as the model card recommends for asymmetric tasks. The wording test measures how much the choice moves the vector. | Model card l.139–144 |

### The shared space

| Question | Answer | Source |
|---|---|---|
| Architecture | **One shared backbone**, not two towers: audio becomes soft tokens inside the text transformer's sequence. The only modality-specific parts are the audio encoder and its 1536→512 projection. | `modeling_embedding_gemma2.py` l.644–660, l.819–845 |
| Training objective | **Not stated** (no loss, temperature or pair counts given). The card says only that pre-training used "paired text, image, video, and audio data to align representations across different modalities". | Model card l.258–265 |
| Training data | Speech in many languages, "environmental sounds, and acoustic events"; cutoff January 2025. Music: **not stated**. Dataset names: **not stated**. | Model card l.258–265 |
| Modality gap | **Not stated** in the docs. The sister measured it: mean cosine clip↔clip 0.71, text↔text 0.69, text↔clip 0.57. | Sister `PROJECT_STATUS.md` §7 |
| Benchmarks | Audio retrieval: MSEB MRR@10 69.54; MAEB 49.39. | Model card l.87–88 |

## CLAP (`laion/clap-htsat-unfused`), briefly

| Question | Answer | Source |
|---|---|---|
| Audio input | **48 kHz**, 64-bin log-mel (50 Hz–14 kHz, 1024-point FFT, hop 480 = 10 ms). | `[M-clap] preprocessor_config.json` |
| Max length | **10 s.** The processor's default for longer audio is `rand_trunc`, a *random* 10 s crop (not deterministic!). Shorter audio is repeat-padded (`repeatpad`). The unfused model has `enable_fusion: false`. **Our wrapper** (copied from the sister) splits audio into fixed 10 s windows, embeds each, and averages: deterministic, nothing dropped. | `[M-clap] preprocessor_config.json` (`max_length_s: 10`, `truncation: rand_trunc`, `padding: repeatpad`); `[M-clap] config.json` audio_config; `src/audiovocab/embed.py` `Clap.audio` |
| Audio encoder and pooling | HTS-AT (Swin transformer, depths 2/2/6/2) → **average pool** over the final feature map → projection to 512 → L2-normalise. | `[T] models/clap/modeling_clap.py` l.759, l.894, l.1462, l.1503ff. |
| Text | RoBERTa tokeniser (vocabulary 50,265, max 512 tokens). The **first token** (`<s>`) is pooled (dense + tanh) → projection → L2-normalise. **No task prefixes.** | `[M-clap] tokenizer_config.json`; `modeling_clap.py` l.1234–1244, l.1459, l.1496–1497 |
| Shared space | **Two towers** joined by a contrastive (InfoNCE-style) loss with a learned temperature, trained on LAION-Audio-630K with keyword-to-caption augmentation. | Model card abstract (`[M-clap] README.md`); `config.json` `logit_scale_init_value` |
| Output | 512-d. | `config.json` `projection_dim: 512` |

`[M-clap]` = `podcast-vibe-search/models/laion--clap-htsat-unfused/`.

## Behaviour tests

On 30 of the sister's podcast clips (one per show, seed 0) plus synthetic signals; `src/audiovocab/modelnotes.py`,
full numbers in `reports/phase1_behaviour.json`. The tables are filled in below from that file.

<!-- behaviour tables: generated by audiovocab.notes_md, do not edit below -->

### Length (n = 30 clips; cosine to the same clip at 30 s, mean and min)

| Length (s) | 2 | 5 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| gemma | 0.762 (0.701) | 0.845 (0.763) | 0.911 (0.827) | 0.968 (0.930) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| clap | 0.830 (0.529) | 0.923 (0.813) | 0.963 (0.910) | 0.988 (0.968) | 1.000 (1.000) | 0.992 (0.973) | 0.978 (0.910) |

| | Gemma | CLAP |
|---|---|---|
| Next, non-overlapping 20 s window vs this 20 s window (content drift baseline), mean | 0.811 | 0.906 |
| 40 s vs 30 s embedding, max abs difference | 0.000 | 0.030 |
| 60 s vs 30 s embedding, max abs difference | 0.000 | 0.055 |

- **Gemma truncates at 30 s** (difference exactly 0). CLAP keeps changing past 30 s, because our wrapper averages extra 10 s windows.
- Shorter clips drift away from the 30 s vector, but even a 10 s clip stays closer to it than the next 20 s of the same show is to this one. **Content moves the vector more than length does.** Phase 3 therefore uses windows of ≤ 10 s for sounds and 10 s windows for music.

### Loudness (10 s clips, cosine to 0 dB; mean and min)

| Gain | -12 dB | -6 dB | 0 dB | 6 dB |
|---|---|---|---|---|
| gemma | 0.998 (0.980) | 0.999 (0.994) | 1.000 (1.000) | 0.999 (0.995) |
| clap | 0.920 (0.865) | 0.978 (0.963) | 1.000 (1.000) | 0.978 (0.963) |

- **Gemma is close to loudness-invariant**; CLAP drops more at −12 dB. +6 dB was applied without clipping protection.

### Silence and noise (10 s)

| | Gemma | CLAP |
|---|---|---|
| Clip↔clip cosine, mean | 0.706 | 0.557 |
| silence ↔ clips, mean cosine | 0.666 | 0.072 |
| clips for which silence is nearer than their nearest real clip (of 30) | 0 | 0 |
| white ↔ clips, mean cosine | 0.658 | 0.117 |
| clips for which white is nearer than their nearest real clip (of 30) | 0 | 0 |
| pink ↔ clips, mean cosine | 0.671 | 0.084 |
| clips for which pink is nearer than their nearest real clip (of 30) | 1 | 0 |

- **Silence is not a hub** in either model: no clip has silence as its nearest neighbour.
- In Gemma **everything is close to everything** (clip↔clip about 0.7, and even silence↔speech about 0.67): the space is a narrow cone. Absolute cosines are therefore not comparable across models; only rankings and contrasts are. CLAP spreads clips much more widely (silence↔speech ≈ 0).

### Text wording (Gemma; 8 concepts × word / phrase / description)

| Prefix | Same text with vs without prefix | Same concept, different wording | Different concepts, same form | Text↔audio: word | phrase | description |
|---|---|---|---|---|---|---|
| none | — | 0.831 | 0.697 | 0.663 | 0.653 | 0.637 |
| SearchQuery | 0.893 | 0.833 | 0.663 | 0.580 | 0.572 | 0.557 |
| Document | 0.882 | 0.851 | 0.710 | 0.590 | 0.596 | 0.577 |
| Classification | 0.774 | 0.851 | 0.708 | 0.528 | 0.543 | 0.548 |
| CLAP (no prefixes) | — | 0.635 | 0.218 | -0.004 | -0.050 | 0.022 |

- **The prefix is a big move**: the same text with `SearchQuery` has cosine 0.893 with the bare text; `Classification` moves it further (0.774). `SearchQuery` and `Document` are nearly the same direction (0.971).
- **Prefixes push text away from audio** in raw cosine (bare text sits closest). That's a shift of the whole text cloud (the modality gap), not necessarily worse ranking. Which form ranks clips best is measured as zero-shot accuracy on a tuning split in Phase 4, and the choice is fixed before the reported split is scored.
- Wording matters less than concept: same-concept rewordings stay closer than different concepts in the same form, for every prefix.

### Embedding cost (CPU, 12 cores)

| Clip length | 1 s | 2 s | 5 s | 10 s |
|---|---|---|---|---|
| gemma: s per clip | 0.72 | 1.04 | 2.29 | 3.90 |
| clap: s per clip | 0.24 | 0.26 | 0.37 | 0.36 |

Measured while downloads were also running, so these are upper bounds. A clean single 20 s clip (Phase 0 parity run) took **4.19 s with Gemma** and **0.58 s with CLAP**. Gemma costs about 10× CLAP. There's a fixed per-clip overhead, so many short clips cost more per second of audio than a few long ones.
