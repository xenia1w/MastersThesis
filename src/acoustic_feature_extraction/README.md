# Phase 1 — Acoustic Feature Extraction

Extract WavLM speaker embeddings from L2-ARCTIC and Speech Accent Archive,
generate audio perturbations, and analyse embedding stability and sensitivity.

All commands are run from the **project root**.

---

## Pipelines

### 1. Extract Speaker Embeddings

**L2-ARCTIC** (3 speakers × 3 utterances by default):
```bash
uv run python main.py --dataset l2arctic
```

**L2-ARCTIC** with a custom limit:
```bash
uv run python main.py --dataset l2arctic --max-items 10
```

**Speech Accent Archive:**
```bash
uv run python main.py --dataset saa
```

**SAA** with a quick subset for testing:
```bash
uv run python main.py --dataset saa --max-items 20
```

Output: `data/processed/l2arctic_minimal_embeddings/` or `data/processed/saa_minimal_embeddings/`
Each speaker gets a subdirectory of `.pt` files (PyTorch tensors + metadata).

---

### 2. Generate Perturbed Audio

Applies 5 acoustic perturbation types to each utterance:
- `rate_p10` / `rate_m10` — time stretch ±10%
- `pitch_p2st` / `pitch_m2st` — pitch shift ±2 semitones
- `pause_ins` — insert 0.2s silence at 35% and 70% of the utterance

```bash
uv run python -m src.acoustic_feature_extraction.pipeline.generate_perturbations
```

Output: `data/processed/perturbations/{dataset}/` + `manifest.csv`

---

### 3. Extract Perturbation Embeddings

Extracts WavLM embeddings for each original and perturbed audio file.

```bash
uv run python -m src.acoustic_feature_extraction.pipeline.extract_perturbation_embeddings
```

Output: `data/processed/perturbation_embeddings/{dataset}/` + `manifest_embeddings.csv`

---

### 4. Analyse Perturbation Sensitivity

Computes cosine similarity between original and perturbed embeddings for each
of the 3 embedding types (mean-pooled, mean+std, x-vector).

```bash
uv run python -m src.acoustic_feature_extraction.pipeline.analyze_perturbation_sensitivity
```

Output: `data/processed/perturbation_sensitivity/{dataset}/`
- `sensitivity_detail.csv` — per-utterance cosine similarities
- `sensitivity_aggregate.csv` — per-perturbation-type statistics
- `sensitivity_summary.json` — overall summary

---

### 5. Speaker Stability Analysis

Plots how quickly each speaker's embedding converges as more utterances are added.

```bash
uv run python -m src.acoustic_feature_extraction.pipeline.speaker_stability
```

Output: `data/processed/stability/`

---

## Tests

```bash
# All Phase 1 tests
uv run pytest tests/test_generate_perturbations.py \
               tests/test_analyze_perturbation_sensitivity.py \
               tests/test_extract_perturbation_embeddings.py \
               tests/test_incremental_embeddings.py \
               tests/test_saa_segmentation.py -v
```

---

## Embedding Formats

Each `.pt` file contains a dict loadable with `torch.load()`:

| Key | Shape | Description |
|-----|-------|-------------|
| `utterance_embedding` | `[768]` or `[512]` | Mean-pooled or x-vector embedding |
| `utterance_embedding_meanstd` | `[1536]` or `[1024]` | Mean+std concatenated, L2-normalised |
| `frame_representations` | `[T, 768]` | Frame-level representations at ~50 Hz |
| `model_name` | str | Model identifier |

Models used:
- `microsoft/wavlm-base-plus` → 768-dim frames
- `microsoft/wavlm-base-plus-sv` → 512-dim x-vectors (speaker verification head)
