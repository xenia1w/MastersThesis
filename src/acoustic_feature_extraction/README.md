# Part 1 — Acoustic Feature Extraction (RQ1)

Extract WavLM speaker embeddings from L2-ARCTIC and Speech Accent Archive,
generate audio perturbations, and analyse embedding stability and sensitivity.

All commands are run from the **project root**.

> ### WavLM here, wav2vec2 for adaptation
>
> This module is the **exploratory** stage: it characterises speaker embeddings
> (how stable they are, how they respond to perturbation) using WavLM. WavLM is
> **not** the profile extractor used in the adaptation experiments.
>
> The acoustic profile that actually conditions the ASR model is extracted with
> **`facebook/wav2vec2-base`** — see `asr_adaptation/data/wav2vec2_speaker_embeddings.py`,
> which is the default (`--profile-extractor wav2vec2`). The reason is representational
> coherence: the ASR backbone is `facebook/wav2vec2-base-960h`, so extractor and
> backbone share transformer weights and therefore a latent space. WavLM has no
> official CTC head, which would have put the profile in a different space from the
> receiver.
>
> WavLM remains available for adaptation as a non-default fallback
> (`--profile-extractor wavlm`) for ablation. The two modules share pooling code:
> `wav2vec2_speaker_embeddings.py` imports `mean_std_pool` from
> `features/utterance_embedding.py` here.

---

## Pipelines

### 1. Extract Speaker Embeddings

**L2-ARCTIC** — a fixed sample of 3 speakers (ABA, ASI, BWC) × 3 utterances
(`arctic_a0001`–`a0003`), hardcoded as `default_samples()`:
```bash
uv run python main.py --dataset l2arctic
```

**Speech Accent Archive** — every recording in the archive:
```bash
uv run python main.py --dataset saa
```

**SAA** with a quick subset for testing:
```bash
uv run python main.py --dataset saa --max-items 20
```

> `--max-items` only ever *truncates* the sample list. For SAA that's a useful
> subset of the whole archive; for L2-ARCTIC the default list is just 9 samples, so
> anything above 9 has no effect. To extract from more L2-ARCTIC speakers, edit
> `default_samples()` in `pipeline/l2arctic_minimal.py` or pass `samples=` to
> `run_acoustic_pipeline()` directly.

Common flags: `--model-name` (default `microsoft/wavlm-base-plus-sv`), `--outer-zip`
and `--save-root` to override the default paths, plus `--include-missing` /
`--no-validate-files` for SAA metadata handling.

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

Output: `data/processed/stability/{dataset}_speaker_stability/`, including
`speaker_stability_all.csv`.

---

### 6. Figures

Two plotting modules turn the analysis CSVs into the figures used in the thesis.

```bash
# Stability curves (reads speaker_stability_all.csv)
uv run python -m src.acoustic_feature_extraction.plots.plot_speaker_stability

# Perturbation sensitivity, both datasets
uv run python -m src.acoustic_feature_extraction.plots.plot_perturbation_sensitivity
```

Both write `png` + `pdf` (override with `--formats`) alongside a `*_stats.csv` of the
plotted values, under `src/acoustic_feature_extraction/plots/`. Use `--dataset
l2arctic|saa|all` to restrict the sensitivity plot, and `--csv-path` / `--out-dir`
to point the stability plot at a different run.

---

## Tests

```bash
# All feature-extraction tests
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
- `microsoft/wavlm-base-plus-sv` → 512-dim x-vectors (speaker verification head).
  **This is the default** for `main.py`.
- `microsoft/wavlm-base-plus` → 768-dim frames. Select with
  `--model-name microsoft/wavlm-base-plus`.

Which model you run determines which shapes appear in the table above: the `-sv`
head yields `[512]` / `[1024]`, the plain encoder `[768]` / `[1536]`.
