# Cross-check: did podcast-vibe-search use EmbeddingGemma 2 correctly?

Read-only review of the sister repo at commit `4facfd4`, against `MODEL_NOTES.md`. Nothing in the sister repo was
changed; whether to act on anything here is the user's call.

## Verdict: no problems found

| Check | What the sister did | What the model needs | OK? |
|---|---|---|---|
| **Clip length vs the max length** | 20 s segments (`config.yaml` `segment_s: 20`; `audio.py` `segment_bounds`) | Audio over **30 s** is silently truncated; 20 s = 500 audio tokens, all used. **Measured:** 20 s → 500 tokens (`reports/phase1_audio_tokens.json`). | **Yes:** nothing truncated |
| The `audio_seq_length: 280` setting (11.2 s) in `processor_config.json` | Not used | Applies only to a vLLM placeholder helper, not the sentence-transformers path. **Measured:** 20 s gives 500 tokens, not 280. | **Yes** |
| **Audio prefix** | None (`embed.py` `Gemma.audio` passes `{"array", "sampling_rate"}` only) | "Pass … audio without any prefix" (model card l.131) | **Yes** |
| Sample rate | Decoded at 16 kHz mono with ffmpeg (`audio.py` `decode`, `Gemma.sr = 16000`) | 16 kHz mono | **Yes** |
| **Search queries** | `Gemma.text(texts)` with default `prompt_name="SearchQuery"`, giving `task: search result \| query: …` (`rank.py` l.49, l.121) | Asymmetric retrieval: query prefix on queries | **Yes** |
| Zero-shot feature prompts (laughter/music/crowd) | Embedded with `SearchQuery` (`textemb.py`, `features.py` l.70–72) and compared to audio | Same as queries | **Yes** |
| Transcript embeddings (topic signal) | `"title: none \| text: {transcript}"`, with `prompt_name=None` (`textemb.py`) | Document format `title: none \| text: …` (model card l.133, l.144) | **Yes** (and the parity check reproduces these vectors exactly) |
| Centring text mean | 77 `GENERIC_PHRASES` embedded with `SearchQuery`, like the queries they centre | Consistent with queries | **Yes** |
| Precision | float32 on CPU | bf16 or fp32, never fp16 | **Yes** |
| CLAP (Phase 1 model pick only) | Fixed 10 s windows, averaged | Max 10 s; the processor's default *random* crop was avoided | **Yes** |

## Notes that don't change any result

- **The model card's "~327 s of audio" is not achievable** through transformers/sentence-transformers. It's
  capped at 30 s by the feature extractor. The sister never needed more.
- **Mean pooling includes the prompt tokens** (`include_prompt: true`). So the `SearchQuery` prefix shifts every query
  vector by a shared component. That's part of the modality gap the sister measured (text↔text 0.69 vs text↔clip
  0.57), and its centring removed it. The Phase 1 wording test (`MODEL_NOTES.md`) measures how big that shift is.
