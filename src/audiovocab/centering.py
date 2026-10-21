"""Closing the modality gap: subtract each modality's mean vector before comparing (from podcast-vibe-search).

In EmbeddingGemma 2's shared space, text vectors sit closer to other text and audio to other audio than text to
audio. Subtracting an audio mean (over a clip set) and a text mean (over a fixed, neutral phrase set) and
re-normalising lines the two clouds up. Here the means are passed in explicitly, because each analysis computes its
audio mean on its own tuning split rather than one global catalogue.

The text mean uses GENERIC_PHRASES: everyday sound/voice/setting vocabulary, deliberately not any test query, so it
can't be tuned to them.
"""
import numpy as np

GENERIC_PHRASES = [
    "a man talking", "a woman talking", "two people talking", "a group conversation", "an interview", "a monologue",
    "a lecture", "a phone call", "people laughing", "someone crying", "shouting", "whispering", "singing", "humming",
    "applause", "a crowd", "a stadium", "a quiet room", "a busy street", "a cafe", "rain", "wind", "thunder", "birds",
    "ocean waves", "a fire crackling", "traffic", "footsteps", "a door closing", "typing", "piano music",
    "guitar music", "drums", "electronic music", "orchestral music", "jazz", "a radio jingle", "an advertisement",
    "silence", "static noise", "a fast speaker", "a slow speaker", "a deep voice", "a high voice", "an old man",
    "a young child", "an excited voice", "a calm voice", "an angry voice", "a sad voice", "a nervous voice",
    "a confident voice", "a funny story", "a scary story", "a sad story", "a bedtime story", "a news report",
    "a weather report", "a sports commentary", "a cooking show", "a science explanation", "a history lesson",
    "a meditation", "a prayer", "a speech", "a debate", "an argument", "a joke", "gossip", "a recipe", "an audiobook",
    "a podcast intro", "a game show", "a quiz", "a love story", "a crime report", "a business update",
]


def _l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def means(audio_emb, encoder):
    """Audio mean over the given clips + text mean over GENERIC_PHRASES."""
    return {"audio": audio_emb.mean(0), "text": encoder.text(GENERIC_PHRASES).mean(0)}


def center(x, mean, strength=1.0):
    """Subtract `strength` × mean (0 = raw, 1 = full centring), then re-normalise."""
    return x if not strength else _l2(x - strength * mean)
