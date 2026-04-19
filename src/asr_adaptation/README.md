# Phase 2 — ASR Speaker Adaptation

Fine-tune `microsoft/wavlm-base-plus` with per-speaker LoRA adapters using
labeled L2-ARCTIC utterances. Evaluate WER improvement and data efficiency.

All commands are run from the **project root**.

---

## Pipelines

### 1. Baseline WER Evaluation (Ticket #4)

Evaluates the **unadapted** model to establish a WER baseline before any fine-tuning.

**Single speaker (local, for quick testing):**
```bash
uv run python -m src.asr_adaptation.pipeline.baseline_eval \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation/baseline_wer \
    --cache-dir     data/cache/huggingface \
    --speaker       ABA
```

**All L2-ARCTIC speakers + SAA (on cluster):**
```bash
sbatch src/asr_adaptation/slurm/run_baseline.sh
```

Output:
- `data/processed/asr_adaptation/baseline_wer/l2arctic_baseline.csv`
- `data/processed/asr_adaptation/baseline_wer/saa_baseline.csv`

CSV columns: `speaker_id, utterance_id, native_language, reference, hypothesis, wer`

**Quick sanity check after running:**
```bash
# Preview first rows
head -5 data/processed/asr_adaptation/baseline_wer/l2arctic_baseline.csv

# Average WER across all utterances for a speaker
awk -F',' 'NR>1 {sum+=$6; count++} END {printf "Avg WER: %.3f (%d utterances)\n", sum/count, count}' \
    data/processed/asr_adaptation/baseline_wer/l2arctic_baseline.csv
```

---

### 2. Per-Speaker LoRA Fine-Tuning (Ticket #5)

Fine-tunes a speaker-specific LoRA adapter on their labeled utterances and
measures adapted WER on a held-out test set.

- Train/eval split: first 500 utterances (shuffled by seed) for training, last 100 held out
- Saves LoRA weights + a per-utterance CSV with baseline and adapted WER side-by-side

**Single speaker (local):**
```bash
uv run python -m src.asr_adaptation.pipeline.lora_train \
    --speaker       ABA \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation \
    --cache-dir     data/cache/huggingface
```

Optional flags: `--n-train 100`, `--n-epochs 10`, `--seed 0`

**All speakers in parallel (on cluster):**
```bash
sbatch --array=0-17 src/asr_adaptation/slurm/run_lora_speaker.sh
```

Output:
- `data/processed/asr_adaptation/lora_weights/{speaker_id}/` — saved LoRA adapter weights
- `data/processed/asr_adaptation/adaptation_results/{speaker_id}.csv`

CSV columns: `speaker_id, utterance_id, n_train, reference, hypothesis_baseline, hypothesis_adapted, wer_baseline, wer_adapted, wer_delta`

**Quick sanity check after running:**
```bash
# Average baseline vs adapted WER for ABA
awk -F',' 'NR>1 {b+=$7; a+=$8; c++} END {printf "Baseline: %.3f  Adapted: %.3f\n", b/c, a/c}' \
    data/processed/asr_adaptation/adaptation_results/ABA.csv
```

---

### 3. Data Size Sweep — RQ1.3 (Ticket #6)

Trains LoRA on N utterances (N ∈ {1, 5, 10, 20, 50, 100, 200}), repeated across 3 seeds,
to find how much data is needed before adaptation becomes effective.

Each job handles one `(speaker, N, seed)` combination and writes its own CSV file,
avoiding write conflicts when hundreds of jobs run simultaneously on the cluster.

**Single combination locally (good for testing):**
```bash
uv run python -m src.asr_adaptation.pipeline.data_size_analysis run \
    --speaker       ABA \
    --n-train       10 \
    --seed          0 \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation/data_size_curves \
    --cache-dir     data/cache/huggingface
```

Output: `data/processed/asr_adaptation/data_size_curves/ABA_0010_seed0.csv`

**Full sweep — 18 speakers × 7 N-values × 3 seeds = 378 jobs (on cluster):**
```bash
sbatch --array=0-377 src/asr_adaptation/slurm/run_data_size.sh
```

**After all jobs finish — merge into per-speaker summary files:**
```bash
uv run python -m src.asr_adaptation.pipeline.data_size_analysis merge \
    --output-dir data/processed/asr_adaptation/data_size_curves
```

Output per speaker: `data/processed/asr_adaptation/data_size_curves/{speaker_id}_wer_vs_n.csv`

CSV columns: `speaker_id, n_train, seed, wer_baseline, wer_adapted, wer_delta`

---

### 4. Acoustic Correlation Analysis (Ticket #7)

*Not yet implemented — coming after tickets #5 and #6.*

Correlates WavLM embedding distance (from Phase 1) with WER improvement
from LoRA adaptation. Connects Phase 1 to Phase 2 analytically.

```bash
uv run python -m src.asr_adaptation.pipeline.acoustic_correlation \
    --embeddings-dir    data/processed/l2arctic_minimal_embeddings \
    --adaptation-csv    data/processed/asr_adaptation/adaptation_results/l2arctic_adapted.csv \
    --output-dir        data/processed/asr_adaptation/acoustic_correlation
```

---

## SLURM Cluster Workflow

```bash
# 1. Push latest code to cluster via git, then on the cluster:
git pull

# 2. First time only — set up venv on cluster:
uv sync

# 3. Download model weights (first time only, run interactively):
uv run python -c "
from transformers import WavLMForCTC, Wav2Vec2Processor
WavLMForCTC.from_pretrained('microsoft/wavlm-base-plus', cache_dir='data/cache/huggingface')
Wav2Vec2Processor.from_pretrained('microsoft/wavlm-base-plus', cache_dir='data/cache/huggingface')
"

# 4. Submit jobs
mkdir -p logs
sbatch src/asr_adaptation/slurm/run_baseline.sh
sbatch --array=0-17  src/asr_adaptation/slurm/run_lora_speaker.sh
sbatch --array=0-377 src/asr_adaptation/slurm/run_data_size.sh

# 5. Monitor
squeue -u $USER
tail -f logs/baseline_<jobid>.out

# 6. Copy results back to laptop
rsync -avz username@cluster:~/MastersThesis/data/processed/asr_adaptation/ \
    data/processed/asr_adaptation/
```

> Before submitting, fill in the 3 TODOs in each SLURM script:
> `--partition`, `--account`, and any `module load` commands your cluster needs.

---

## Tests

```bash
# All Phase 2 tests
uv run pytest tests/test_l2arctic_transcriptions.py \
               tests/test_wavlm_lora.py \
               tests/test_wer.py \
               tests/test_baseline_eval.py \
               tests/test_lora_train.py \
               tests/test_data_size_analysis.py -v

# All tests in the whole project
uv run pytest -v
```

---

## Architecture

```
src/asr_adaptation/
├── data/
│   └── l2arctic_transcriptions.py  # Load audio + transcripts from nested zip
├── inference/
│   └── transcribe.py               # Shared transcription utility (chunked)
├── metrics/
│   └── wer.py                      # compute_wer() with text normalization
├── models/
│   └── wavlm_lora.py             # build_lora_model(), save/load adapter
├── pipeline/
│   ├── baseline_eval.py            # Ticket #4
│   ├── lora_train.py               # Ticket #5 ✓
│   ├── data_size_analysis.py       # Ticket #6 ✓
│   └── acoustic_correlation.py     # Ticket #7 (TODO)
└── slurm/
    ├── run_baseline.sh
    ├── run_lora_speaker.sh
    └── run_data_size.sh
```

## L2-ARCTIC Speakers

The full dataset has 18 speakers used in the SLURM array jobs:

```
Index:   0    1    2    3     4     5    6    7     8
Speaker: ABA  ASI  BWC  EBVS  ERMS  HJK  HKK  HQTV  LXC

Index:   9      10   11   12   13    14   15    16   17
Speaker: MBMPS  NCC  NJS  PNV  RRBI  SKA  SVBI  THV  TNI
```
