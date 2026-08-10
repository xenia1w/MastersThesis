# Masters Thesis — Improving Accuracy in Automatic Speech Recognition via Integration of Acoustic and Linguistic Features

## Overview

This project investigates speaker adaptation for automatic speech recognition (ASR)
along two independent axes, one per research question:

- **Part 1 — Acoustic adaptation (RQ1).** Can a speaker's *acoustic* profile be
  injected into an ASR model to reduce WER?
- **Part 2 — Linguistic adaptation (RQ2).** Can a call-level *textual* profile —
  domain vocabulary, named entities, speaking style — do the same through prompting?

The two parts use different models, datasets and metrics, and are evaluated
separately.

---

## Part 1 — Acoustic Adaptation (RQ1)

Datasets: L2-ARCTIC and the Speech Accent Archive.
ASR backbone: `facebook/wav2vec2-base-960h`; profile extractor: `facebook/wav2vec2-base`.
Sharing transformer weights between the two keeps the profile and its receiver in
one latent space — the reason wav2vec2 was chosen over WavLM, which has no
official CTC head.

**Feature extraction** (`src/acoustic_feature_extraction/`)
Exploratory characterisation of speaker embeddings using WavLM: perturbation
sensitivity, and how stable an embedding is as a function of how much audio it is
built from. Note that WavLM is used *here only* — the profile that conditions the
ASR model is extracted with wav2vec2 (see below).

**LoRA speaker adaptation** (`src/asr_adaptation/`)
Fine-tune with per-speaker LoRA adapters on labeled L2-ARCTIC utterances. Measure
WER improvement and how much adaptation data is actually needed.

**Acoustic profile injection via FiLM** (`src/asr_adaptation/`)
FiLM (Feature-wise Linear Modulation) conditioning: a speaker centroid extracted
from the Wav2Vec2 encoder is fed through an MLP producing per-layer (γ, β) pairs,
applied after each encoder layer's final norm. LoRA adapters and the FiLM MLP are
trained jointly. Includes a wrong-speaker control to verify the model genuinely
uses speaker-specific information, and a layer sweep to find which encoder layer
yields the most discriminative profile.

Guides: [`acoustic_feature_extraction/README.md`](src/acoustic_feature_extraction/README.md)
· [`asr_adaptation/README.md`](src/asr_adaptation/README.md)

---

## Part 2 — Linguistic Adaptation (RQ2)

Dataset: *Earnings21* (43 calls). Model: Whisper Medium. Code:
`src/linguistic_prompting/`.

An LLM builds a call-level context profile, which is injected two different ways:

- **Prompt-time conditioning** — the profile is passed as Whisper's `initial_prompt`,
  softly biasing decoding toward the call's vocabulary. Four content strategies
  (`metadata_only`, `transcript_only`, `transcript_plus_knowledge`,
  `transcript_metadata_knowledge`) cross an orthogonal format axis (keyword `list`
  vs. natural-language `prose`).
- **Post-hoc correction** — decoding is left untouched and the LLM instead repairs
  the raw hypothesis, in `blind`, `context`, or `profile` mode.

Evaluation uses a fixed-window design: the profile is built from `[0:00, 5:00]` and
scored on the disjoint `[5:00, 15:00]` window against hand-annotated references
(`manual_annotation.csv`), so no profile ever sees its own evaluation audio.
Metrics are WER and entity-EER.

Guide: [`linguistic_prompting/README.md`](src/linguistic_prompting/README.md)

---

## Setup

```bash
# Install all dependencies (including dev)
uv sync --dev

# Verify everything is working
uv run pytest
uv run ty check
```

**Required data** (place in `data/raw/` — not committed to git):
- `data/raw/l2arctic_release_v5.0.zip` — L2-ARCTIC v5 dataset (part 1)
- `data/raw/archive.zip` — Speech Accent Archive (part 1)
- `data/raw/earnings21/` — Earnings21 audio, `.nlp` references and file metadata (part 2)

**Part 2 also needs** an API key in `.env` — `KISSKI_API_KEY` (or `SAIA_API_KEY`) —
for profile building and post-hoc correction, which run locally against the
academic-cloud endpoint rather than on the cluster.

---

## Running Tests

```bash
# All tests
uv run pytest

# With verbose output
uv run pytest -v

# A specific test file
uv run pytest tests/test_wer.py -v

# A specific test by name
uv run pytest tests/test_wer.py::test_exact_match_is_zero -v

# Type checking
uv run ty check
```

---

## Project Structure

```
MastersThesis/
├── main.py                            # Part 1 entry point (acoustic features)
├── manual_annotation.csv              # Part 2: hand-annotated window boundaries
├── pyproject.toml
├── src/
│   ├── acoustic_feature_extraction/   # Part 1 — see its README
│   ├── asr_adaptation/                # Part 1 — see its README
│   │   ├── data/
│   │   ├── inference/
│   │   ├── metrics/
│   │   ├── models/
│   │   │   ├── wav2vec_lora.py        # plain LoRA adapter
│   │   │   └── film_lora.py           # FiLM-conditioned LoRA
│   │   ├── pipeline/
│   │   │   ├── baseline_eval.py
│   │   │   ├── lora_train.py
│   │   │   ├── data_size_analysis.py
│   │   │   ├── film_train.py          # FiLM+LoRA training
│   │   │   └── film_wrong_speaker.py  # wrong-speaker control
│   │   └── slurm/
│   └── linguistic_prompting/          # Part 2 — see its README
│       ├── models/                    # prompt templates, profile building, constants
│       ├── pipeline/                  # profile/eval window transcription, scoring, post-hoc
│       ├── comparison/                # baseline-vs-prompted analysis + word-level diffs
│       └── slurm/
├── tests/                             # All unit tests
└── data/
    ├── raw/                           # Raw datasets (not in git)
    ├── cache/huggingface/             # HuggingFace model cache
    └── processed/                     # Pipeline outputs
        ├── l2arctic_minimal_embeddings/
        ├── saa_minimal_embeddings/
        ├── perturbations/
        ├── perturbation_embeddings/
        ├── perturbation_sensitivity/
        ├── asr_adaptation/
        └── linguistic_prompting/
            ├── profiles/              # LLM-built profiles, one subdir per strategy
            ├── v2/                    # fixed-window runs + scores
            └── figures/
```
