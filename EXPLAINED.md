# Audio Vocab, explained in plain words

## 1. The story: why we did this

**The sister project (Vibe Search)** built a search engine for podcasts where you type a *vibe* ("two friends
laughing about football") and it finds 20-second clips that *sound* like that. It used **EmbeddingGemma 2**, a new
Google model that turns both **sounds** and **text** into lists of 768 numbers (an "embedding"). If a clip of
laughter and the words "people laughing" get similar numbers, you can search sound with words.

It worked for some searches and failed for others: 33 of 48 results sounded right to a listener. Two puzzles came
out of it:
- It found people *talking over each other* but not *hostility* ("heated argument" failed).
- Words and sounds sat in separate regions of the space (the "modality gap").

**Anand's question:** "With words, a dozen synonyms of *love* cluster tightly. What's the audio equivalent? Can it
tell 50 kinds of laughter apart? Can it tell a cuckoo from an owl? Is it tuned to human sounds or to nature?"

**So this project:** instead of messy podcasts, we took **datasets where every clip already has a correct label**
(this is a dog, this is a giggle, this actor is speaking angrily, this track is jazz). Then we measured exactly
what the model can and can't tell apart, and whether *words* can find what the *sound* side knows.

## 2. The two models, and why there are two

- **EmbeddingGemma 2 (Gemma):** the model the sister project used, from Google (2025). It's basically a *text*
  model with an audio "ear" attached: audio is turned into pseudo-words and fed through the same brain that reads
  text. It's strong on **speech**.
- **CLAP:** an older model (LAION, 2023), trained specifically on **sound clips paired with captions** ("a dog
  barking", "rain on a roof"). Its whole job is matching sounds to short descriptions.

**Why use CLAP at all?** As a **second opinion**. If both models show the same thing, it's a fact about audio AI in
general. If only one does, it's a quirk of that model. And it turned out to matter a lot: the two are good at
*different* things.

## 3. The two key measurements: "probe" vs "zero-shot"

This is the core idea of the whole project. Each model turns a clip into 768 numbers. We asked two different
questions of those numbers.

### Probe = "Is the information *in* there?"
A **probe** is a tiny, simple classifier trained *on top of* the model's numbers. Show it 30 labelled examples of
"giggle" and 30 of "chuckle"; it learns where the line between them falls; then we test it on clips it has
**never seen, from different people**.
- If the probe is accurate, the model's numbers **contain** the difference.
- The model itself is never changed. The probe only reads its numbers.
- It's like asking: "does this photo contain enough detail to tell twins apart, if someone studies examples?"

### Zero-shot = "Can a *word* find it?"
No training at all. We type the word ("a person giggling", "a person chuckling"), turn it into numbers with the
same model, and label each clip by whichever word it's closest to. That's exactly how a search box works.
- If zero-shot is accurate, **words can reach** that sound.

### The gap between them is the main finding
- **Probe high, zero-shot high:** the model knows it *and* you can search for it with words.
- **Probe high, zero-shot low:** "**knows but can't name**". The information is in the sound numbers, but typing
  the word doesn't find it.
- **Probe low:** the model doesn't really hear the difference at all.

### How to read the numbers
- **Accuracy** = the % of test clips labelled correctly.
- **Chance** = what you'd get by guessing. With 2 options chance is 50%; with 50 options it's 2%. **Always compare
  against chance**: 60% is great with 50 options and poor with 2.
- Everything is tested on **different recordings / speakers / artists** than it was trained on, so the model can't
  cheat by recognising the same microphone or person.

## 4. The data (all public, already labelled; we labelled nothing by hand)

| Dataset | What's in it | Why we used it |
|---|---|---|
| **ESC-50** | 2,000 five-second clips of 50 everyday sounds (dog, rain, chainsaw, sneeze…) | The standard "can it hear everyday sounds" test |
| **FSD50K** | 2,399 clips in 62 finer classes: **giggle vs chuckle**, scream vs shout, whisper, car vs truck… | The fine distinctions, including the only laughter *types* with labels |
| **VocalSound** | 600 clips: laughter, sigh, cough, throat-clear, sneeze, sniff (one person each) | Human non-speech sounds |
| **xeno-canto** | 680 bird recordings: **cuckoo, 6 owl species**, doves, songbirds, crows | Anand's "cuckoo vs owl" question |
| **CREMA-D** | 7,442 clips: **91 actors saying the same 12 sentences in 6 emotions** (angry, happy, sad…) | Lets us separate *what* is said from *how* it's said |
| **FMA** | 4,192 music tracks with genre (jazz, rock, classical…) and region (Brazilian, Indian…) | "Can a genre word find music?" |
| **Musopen** | 116 real classical recordings (Beethoven, Mozart…) | Checks that "classical" means real classical, not just FMA's amateur version |
| **MTG-Jamendo** | 1,135 tracks tagged with genre **and** instrument | "Piano in classical vs piano in jazz": is "classical" just "piano"? |

For music we removed tracks with singing (using a speech-recognition model, **Whisper**). Otherwise lyrics would
give the genre away (Portuguese words → "Brazilian").

## 5. What we found, with examples

### Finding 1: Gemma hears sounds well, but words barely reach them. CLAP does both.
**What we ran:** both models on ESC-50 (50 everyday sounds; chance = 2%).

| | Probe (is it in there?) | Zero-shot (can a word find it?) |
|---|---|---|
| Gemma | 83% | **25%** |
| CLAP | 97% | **90%** |

**Example:** for "thunderstorm" Gemma's probe is 100% right, but typing "thunderstorm" finds about 3% of them. The
same happens for toilet flush, rooster, chainsaw and vacuum: the probe gets ~90–100%, and the word finds ~0%.
**Meaning:** Gemma's sound side is fine, but its link from words to sounds is very weak for sound effects. CLAP was
trained for exactly that link. **Bad news for Gemma as a sound-effect search engine; good news for CLAP.**

### Finding 2: Laughter types are only roughly distinguishable, and not "better" than birds.
**What we ran:** giggle vs chuckle (FSD50K, chance 50%), and cuckoo vs 6 owl species (chance 14%).
- Giggle vs chuckle: **69%** for both models. It's real, but coarse; nowhere near "50 kinds of laughter".
- Cuckoo vs owls: **73–74%** across 7 species. Relative to chance, that's *better* than the laughter result.
- **Answer to Anand:** the space is *not* specially tuned to human expression over wildlife.
- **But words differ a lot:** typing "giggle" finds giggles **90–97%** of the time, while typing an owl species
  name barely beats guessing (~22%). The model "hears" bird species but has no vocabulary for them.

### Finding 3: Emotion is encoded but unnameable.
**What we ran:** CREMA-D (same sentences in 6 emotions; chance ≈ 17%).
- **Probe on emotion: 56–58%**, well above chance and above simple loudness/pitch measurements (44%). Anger is
  easiest: the probe finds **74%** of angry clips.
- **Zero-shot:** typing "an angry voice" finds **0–2%** of angry clips. Both models call almost everything
  "neutral".
- **"Encoded but unnameable" means:** the emotion is *in* the numbers (a probe trained on examples can read it), but
  no text query can pull it out. This explains the sister project's "heated argument" failure: the anger signal
  exists in the audio, but the search words can't reach it.

**Bonus finding:** Gemma knows *what sentence* was said almost perfectly (**100%**) but *who* said it barely at all
(**6%**, of 91 people). CLAP gets 48% on who. So Gemma is a "words" model with ears; CLAP hears the voice itself.

### Finding 4: Genre words partly work; "classical" means "tonal sound without drums".
**What we ran:** instrumental FMA tracks, 11 genres (chance 9%).
- Probe: 41% (Gemma), 49% (CLAP). Words: 27% / 37%. Genre is partly in the sound; words partly reach it.
- Words that work well: **rock** (84% of the 25 nearest tracks are rock, where 18% would be random), **folk**,
  **classical**. Words that don't: easy listening, soul, blues.
- **Real classical:** 99–100% of Musopen's Beethoven/Mozart recordings get labelled "classical".
- **What makes it "classical"? We edited the audio and re-measured:**
  - keep only the drums → the "classical" signal **disappears completely**;
  - remove the drums → *any* track sounds more classical;
  - change tempo or pitch → **almost no effect**.

  **So to the model, classical = tonal, drum-free, softer sound**, not the composition. It's also more than just
  "piano": it can tell classical piano from jazz/pop piano (AUC 0.75–0.83, where 0.5 = guessing and 1.0 =
  perfect).
- **Regions** (Brazilian, Indian…): weak. The right region is in the top 3 about half the time (guessing: 27%).

### Finding 5: Back to Vibe Search.
The podcast searches that worked ("friends laughing", "bedtime story voice") succeed because of **speaking style
and the sound of a show**, which these labelled datasets don't measure. The ones that failed (heated argument,
whispering, rain, music) sit exactly where we found weaknesses: emotion is unnameable, and fine sounds are weak in
Gemma.

## 6. Good or bad? Which model is better?

| Use case | Better choice | Why |
|---|---|---|
| Search sound effects / animals / music by typing words | **CLAP** | Its words actually reach the sounds (90% vs 25%) |
| Search *speech* by style or content (podcasts, calls) | **Gemma** | It understands spoken content (100% sentence ID); the sister got 33/48 |
| Recognise the speaker's identity | **CLAP** (or classic voice features) | Gemma barely encodes who is speaking |
| Detect emotion / hostility / bird species | **Neither model's words. Train a small probe** on labelled examples | The info is there; typing words doesn't get it |

## 7. Business sense: is anything deployable?

**Yes, a few concrete things, all cheap** (small models, runs on a laptop CPU, no training of the big model):

1. **Sound search for audio libraries** (sound effects, stock audio, archives): CLAP + text search is accurate
   enough today (~90% on everyday sounds).
2. **"Probe" detectors for things words can't find:** angry/hostile calls in a call centre, specific bird species
   for ecology, laughter in content moderation. Collect a few hundred labelled examples, train a tiny classifier
   on the embeddings (seconds), deploy. This is the main practical lesson: **when search fails, a small trained
   probe often works.**
3. **Music tagging and filtering by genre, for background-music selection:** works for clear genres (rock,
   classical, folk), not for fuzzy ones (easy listening, soul). Region only as a rough filter.
4. **Vibe Search improvements:** route sound-effect queries to CLAP and speech/style queries to Gemma, and add
   probes for emotion.

**Caveats to say out loud:**
- These are internal-study results on specific public datasets.
- Two datasets (ESC-50, MTG-Jamendo) are non-commercial or research-only: fine for this study, but check before
  any client product.
- Some results rest on small samples (e.g. 59 clips for giggle vs chuckle, 16 tracks per genre in the "what makes
  classical" test).

## 8. Glossary

- **Embedding:** a list of numbers (768 for Gemma, 512 for CLAP) that represents a clip or a sentence. Similar
  things get similar numbers.
- **Cosine similarity:** how close two embeddings point in the same direction (1 = same, 0 = unrelated).
- **Probe:** a tiny classifier trained on embeddings to test whether information is *in* them.
- **Zero-shot:** classifying by comparing to word embeddings, with no training (how a search box works).
- **Chance:** the accuracy of random guessing (1 ÷ number of options).
- **Knows but can't name:** probe high, zero-shot low.
- **Modality gap:** words and sounds sit in separate regions of the space, even when they mean the same thing.
- **Centring:** subtracting the average of each side to pull words and sounds together. It helps category lookups
  slightly and can hurt some sounds.
- **AUC:** how well a score separates two groups (0.5 = coin flip, 1.0 = perfect).
- **Whisper:** speech-to-text model; used here only to detect singing.
- **Perturbation:** deliberately editing the audio (remove drums, change tempo…) to see what the model reacts to.
