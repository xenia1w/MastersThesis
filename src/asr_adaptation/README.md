# Phase 2 — ASR Speaker Adaptation

Fine-tune `facebook/wav2vec2-base-960h` with per-speaker LoRA adapters using
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

*Not yet implemented — coming next.*

Fine-tunes a speaker-specific LoRA adapter on their labeled utterances and
measures adapted WER on a held-out test set.

**Single speaker (local):**
```bash
uv run python -m src.asr_adaptation.pipeline.lora_train \
    --speaker       ABA \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation \
    --cache-dir     data/cache/huggingface
```

**All speakers in parallel (on cluster):**
```bash
sbatch --array=0-17 src/asr_adaptation/slurm/run_lora_speaker.sh
```

Output:
- `data/processed/asr_adaptation/lora_weights/{speaker_id}/` — saved LoRA adapter
- `data/processed/asr_adaptation/adaptation_results/l2arctic_adapted.csv`

---

### 3. Data Size Sweep — RQ1.3 (Ticket #6)

*Not yet implemented — coming next.*

Trains LoRA on N utterances (varying N) to find how much data is needed
before adaptation becomes effective. Answers RQ1.3.

**Single (speaker, N, seed) locally:**
```bash
uv run python -m src.asr_adaptation.pipeline.data_size_analysis \
    --speaker       ABA \
    --n-train       10 \
    --seed          0 \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation/data_size_curves \
    --cache-dir     data/cache/huggingface
```

**Full sweep — 18 speakers × 7 N-values × 3 seeds = 378 jobs (on cluster):**
```bash
sbatch --array=0-377 src/asr_adaptation/slurm/run_data_size.sh
```

Output: `data/processed/asr_adaptation/data_size_curves/{speaker_id}_wer_vs_n.csv`

---

### 4. Prosodic Correlation Analysis (Ticket #7)

*Not yet implemented — coming after tickets #5 and #6.*

Correlates WavLM embedding distance (from Phase 1) with WER improvement
from LoRA adaptation. Connects Phase 1 to Phase 2 analytically.

```bash
uv run python -m src.asr_adaptation.pipeline.prosodic_correlation \
    --embeddings-dir    data/processed/l2arctic_minimal_embeddings \
    --adaptation-csv    data/processed/asr_adaptation/adaptation_results/l2arctic_adapted.csv \
    --output-dir        data/processed/asr_adaptation/prosodic_correlation
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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h', cache_dir='data/cache/huggingface')
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h', cache_dir='data/cache/huggingface')
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
               tests/test_wav2vec_lora.py \
               tests/test_wer.py \
               tests/test_baseline_eval.py -v

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
│   └── wav2vec_lora.py             # build_lora_model(), save/load adapter
├── pipeline/
│   ├── baseline_eval.py            # Ticket #4
│   ├── lora_train.py               # Ticket #5 (TODO)
│   ├── data_size_analysis.py       # Ticket #6 (TODO)
│   └── prosodic_correlation.py     # Ticket #7 (TODO)
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
