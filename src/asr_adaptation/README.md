# Phases 2 & 3 — ASR Speaker Adaptation

Fine-tune `facebook/wav2vec2-base-960h` with per-speaker LoRA adapters using
labeled L2-ARCTIC utterances. Phase 3 extends Phase 2 by adding FiLM
(Feature-wise Linear Modulation) speaker conditioning on top of LoRA.

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

---

### 4. FiLM-Conditioned LoRA Training (Ticket #38)

Trains a FiLM+LoRA model: a two-layer MLP maps the speaker's acoustic centroid
(1536-dim Wav2Vec2 mean+std embedding) to per-layer (γ, β) modulation parameters
applied after each encoder `final_layer_norm`. LoRA adapters and the FiLM MLP
are trained jointly; the MLP is zero-initialised so training starts from the
pre-trained LoRA baseline.

**Single speaker (local):**
```bash
uv run python -m src.asr_adaptation.pipeline.film_train \
    --speaker-id    ABA \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation \
    --cache-dir     data/cache/huggingface
```

Optional flags: `--n-train 200`, `--n-epochs 10`, `--profile-layer -1`, `--no-profile` (LoRA-only ablation)

**All 24 speakers in parallel (on cluster):**
```bash
sbatch --array=0-23 src/asr_adaptation/slurm/run_film_speaker.sh
```

Output:
- `data/processed/asr_adaptation/film_lora_weights/{speaker_id}/` — saved LoRA + FiLM MLP weights
- `data/processed/asr_adaptation/film_adaptation_results/{speaker_id}.csv`

CSV columns: `speaker_id, utterance_id, n_train, reference, hypothesis_baseline, hypothesis_adapted, wer_baseline, wer_adapted, wer_delta`

---

### 5. Wrong-Speaker Control Experiment (Ticket #38)

Loads a trained FiLM+LoRA checkpoint for speaker A and re-transcribes their
eval set using speaker B's centroid. If WER increases, FiLM has learned
speaker-specific content; if WER is unchanged, FiLM only helps structurally.

Requires the FiLM training step (above) to have completed first.

**Single pair (local):**
```bash
uv run python -m src.asr_adaptation.pipeline.film_wrong_speaker \
    --speaker-id        ABA \
    --wrong-speaker-id  ASI \
    --l2arctic-zip      data/raw/l2arctic_release_v5.0.zip \
    --checkpoint-dir    data/processed/asr_adaptation \
    --output-dir        data/processed/asr_adaptation \
    --cache-dir         data/cache/huggingface
```

**All 24 speakers (on cluster):**
```bash
sbatch --array=0-23 src/asr_adaptation/slurm/run_film_wrong_speaker.sh
```

Output:
- `data/processed/asr_adaptation/film_wrong_speaker_results/{speaker_id}_vs_{wrong_speaker_id}.csv`

CSV columns: `speaker_id, utterance_id, wrong_speaker_id, reference, hypothesis_correct, hypothesis_wrong, wer_correct, wer_wrong, wer_delta`

---

### 6. Profile Layer Sweep (Ticket #38)

Tests which Wav2Vec2 encoder layer produces the most speaker-discriminative
acoustic profile by training FiLM on layers 4, 6, and 9 (layer -1 covered
by the main FiLM job). Runs on 6 representative speakers.

Submit after the main FiLM job (`JOBID`) has finished:
```bash
sbatch --dependency=afterok:JOBID --export=ALL,PROFILE_LAYER=4 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
sbatch --dependency=afterok:JOBID --export=ALL,PROFILE_LAYER=6 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
sbatch --dependency=afterok:JOBID --export=ALL,PROFILE_LAYER=9 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
```

Output: `data/processed/asr_adaptation/film_layer{N}_results/{speaker_id}.csv`

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

# 4. Submit Phase 2 jobs
mkdir -p logs
sbatch src/asr_adaptation/slurm/run_baseline.sh
sbatch --array=0-23  src/asr_adaptation/slurm/run_lora_speaker.sh
sbatch --array=0-23  src/asr_adaptation/slurm/run_lora_speaker_wav2vec2.sh   # Wav2Vec2 profile variant
sbatch --array=0-23  src/asr_adaptation/slurm/run_lora_speaker_no_profile.sh # LoRA-only ablation

# 5. Submit Phase 3 jobs
FILM_JOB=$(sbatch --array=0-23 --parsable src/asr_adaptation/slurm/run_film_speaker.sh)
sbatch --array=0-23 --dependency=afterok:$FILM_JOB src/asr_adaptation/slurm/run_film_wrong_speaker.sh
# Layer sweep (optional, 6 speakers × 3 layers):
sbatch --dependency=afterok:$FILM_JOB --export=ALL,PROFILE_LAYER=4 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
sbatch --dependency=afterok:$FILM_JOB --export=ALL,PROFILE_LAYER=6 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
sbatch --dependency=afterok:$FILM_JOB --export=ALL,PROFILE_LAYER=9 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh

# 6. Monitor
squeue -u $USER
tail -f logs/film_<jobid>.out

# 7. Copy results back to laptop
rsync -avz username@cluster:~/MastersThesis/data/processed/asr_adaptation/ \
    data/processed/asr_adaptation/
```

---

## Tests

```bash
# All Phase 2 tests
uv run pytest tests/test_l2arctic_transcriptions.py \
               tests/test_wav2vec_lora.py \
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
│   ├── l2arctic_transcriptions.py   # Load audio + transcripts from nested zip
│   ├── speaker_embeddings.py        # WavLM-based speaker centroid computation
│   └── wav2vec2_speaker_embeddings.py # Wav2Vec2-based speaker centroid
├── inference/
│   └── transcribe.py                # Shared transcription utility (chunked)
├── metrics/
│   └── wer.py                       # compute_wer() with text normalization
├── models/
│   ├── wav2vec_lora.py              # Phase 2: build_lora_model(), save/load adapter
│   └── film_lora.py                 # Phase 3: FiLMConditionedLoraModel, build/load/save
├── pipeline/
│   ├── baseline_eval.py             # Ticket #4
│   ├── lora_train.py                # Ticket #5 ✓
│   ├── data_size_analysis.py        # Ticket #6 ✓
│   ├── film_train.py                # Ticket #38 — FiLM+LoRA training
│   ├── film_wrong_speaker.py        # Ticket #38 — wrong-speaker control inference
│   └── results_analysis.py          # Aggregate results analysis utilities
└── slurm/
    ├── run_baseline.sh
    ├── run_lora_speaker.sh
    ├── run_lora_speaker_wav2vec2.sh  # LoRA + Wav2Vec2 profile (ablation)
    ├── run_lora_speaker_no_profile.sh # LoRA-only, no FiLM (ablation)
    ├── run_data_size.sh
    ├── run_film_speaker.sh           # Phase 3: FiLM+LoRA, all 24 speakers
    ├── run_film_wrong_speaker.sh     # Phase 3: wrong-speaker control
    └── run_film_layer_sweep.sh       # Phase 3: profile layer sweep (layers 4/6/9)
```

## L2-ARCTIC Speakers

The full dataset has 24 speakers used in the SLURM array jobs:

```
Index:    0    1    2    3     4     5    6    7     8
Speaker:  ABA  ASI  BWC  EBVS  ERMS  HJK  HKK  HQTV  LXC

Index:    9      10   11   12   13    14   15    16   17
Speaker:  MBMPS  NCC  NJS  PNV  RRBI  SKA  SVBI  THV  TNI

Index:    18    19    20    21    22    23
Speaker:  TXHC  YBAA  YDCK  YKWK  ZHAA  TLV
```
