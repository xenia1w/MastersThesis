# Masters Thesis — Improving Accuracy in Automatic Speech Recognition via Integration of Acoustic and Linguistic Features

## Overview

This project investigates speaker adaptation for automatic speech recognition (ASR)
using acoustic features extracted from non-native English speakers.

**Phase 1 — Acoustic Feature Extraction** (`src/acoustic_feature_extraction/`)
Extract WavLM speaker embeddings from L2-ARCTIC and Speech Accent Archive datasets,
analyse perturbation sensitivity, and study embedding stability.

**Phase 2 — ASR Speaker Adaptation** (`src/asr_adaptation/`)
Fine-tune `wav2vec2-base-960h` with per-speaker LoRA adapters using labeled
L2-ARCTIC utterances. Measure WER improvement and how much data is needed.

**Phase 3 — Acoustic Profile Injection via FiLM** (`src/asr_adaptation/`)
Extend Phase 2 with FiLM (Feature-wise Linear Modulation) conditioning: a speaker
centroid extracted from the Wav2Vec2 encoder is fed through an MLP that produces
per-layer (γ, β) pairs applied after each encoder layer's final norm. Both LoRA
adapters and the FiLM MLP are trained jointly. Includes a wrong-speaker control
experiment to verify the model uses speaker-specific information, and a layer sweep
to identify which encoder layer yields the most discriminative acoustic profile.

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
- `data/raw/l2arctic_release_v5.0.zip` — L2-ARCTIC v5 dataset
- `data/raw/archive.zip` — Speech Accent Archive

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
├── main.py                          # Phase 1 entry point (acoustic features)
├── pyproject.toml
├── src/
│   ├── acoustic_feature_extraction/ # Phase 1 — see its README
│   └── asr_adaptation/              # Phases 2 & 3 — see its README
│       ├── data/
│       ├── inference/
│       ├── metrics/
│       ├── models/
│       │   ├── wav2vec_lora.py      # Phase 2: plain LoRA adapter
│       │   └── film_lora.py         # Phase 3: FiLM-conditioned LoRA
│       ├── pipeline/
│       │   ├── baseline_eval.py
│       │   ├── lora_train.py
│       │   ├── data_size_analysis.py
│       │   ├── film_train.py        # Phase 3: FiLM+LoRA training
│       │   └── film_wrong_speaker.py # Phase 3: wrong-speaker control
│       └── slurm/
├── tests/                           # All unit tests
└── data/
    ├── raw/                         # Raw dataset zips (not in git)
    ├── cache/huggingface/           # HuggingFace model cache
    └── processed/                   # Pipeline outputs
        ├── l2arctic_minimal_embeddings/
        ├── saa_minimal_embeddings/
        ├── perturbations/
        ├── perturbation_embeddings/
        ├── perturbation_sensitivity/
        └── asr_adaptation/
```

---

## Phase Guides

- **Phase 1:** [`src/acoustic_feature_extraction/README.md`](src/acoustic_feature_extraction/README.md)
- **Phases 2 & 3:** [`src/asr_adaptation/README.md`](src/asr_adaptation/README.md)
