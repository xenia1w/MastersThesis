#!/bin/bash
# =============================================================================
# run_film_wrong_speaker.sh
# Wrong-speaker control experiment: load the trained FiLM+LoRA for each speaker
# and run inference on their test set using the *next* speaker's centroid
# (circular rotation), then compare WER to the correct-centroid baseline.
#
# Interpretation:
#   wer_delta > 0  →  wrong centroid hurt WER  →  FiLM uses speaker content ✓
#   wer_delta ≈ 0  →  model ignores profile    →  FiLM helps structurally only
#
# Submit all 24 speaker pairs:
#   sbatch --array=0-23 src/asr_adaptation/slurm/run_film_wrong_speaker.sh
#
# Submit a single pair for a quick sanity check (e.g. ABA ← ASI):
#   sbatch --array=0-0  src/asr_adaptation/slurm/run_film_wrong_speaker.sh
#
# Monitor: squeue -u $USER
# =============================================================================

#SBATCH --job-name=film-wrong-spk
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-23
#SBATCH --partition=gpu
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/film_wrong_%x_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/film_wrong_%x_%j.err

module load cuda/12.8

set -euo pipefail

# PROFILE_LAYER controls which layer the wrong centroid is extracted from.
# Defaults to -1 (last layer) to match the original FiLM training.
# Override via: --export=ALL,PROFILE_LAYER=6
PROFILE_LAYER="${PROFILE_LAYER:--1}"

# Resolve checkpoint/output dir from layer
if [[ "$PROFILE_LAYER" == "-1" ]]; then
    DATA_DIR="data/processed/asr_adaptation_film"
else
    DATA_DIR="data/processed/asr_adaptation_film_layer${PROFILE_LAYER}"
fi

SPEAKERS=(ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI THV TNI TXHC YBAA YDCK YKWK ZHAA TLV)
N=${#SPEAKERS[@]}

# Each speaker gets the *next* speaker's centroid (circular)
SPEAKER="${SPEAKERS[$SLURM_ARRAY_TASK_ID]}"
WRONG_IDX=$(( (SLURM_ARRAY_TASK_ID + 1) % N ))
WRONG_SPEAKER="${SPEAKERS[$WRONG_IDX]}"

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

mkdir -p logs "${DATA_DIR}/film_wrong_speaker_results"

echo "=== Wrong-speaker control (layer=${PROFILE_LAYER}): $SPEAKER (model) ← $WRONG_SPEAKER (centroid) started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.film_wrong_speaker \
    --speaker-id        "$SPEAKER" \
    --wrong-speaker-id  "$WRONG_SPEAKER" \
    --l2arctic-zip      data/raw/l2arctic_release_v5.0.zip \
    --checkpoint-dir    "$DATA_DIR" \
    --output-dir        "$DATA_DIR" \
    --cache-dir         data/cache/huggingface \
    --profile-layer     "$PROFILE_LAYER"

echo "=== Wrong-speaker control (layer=${PROFILE_LAYER}): $SPEAKER ← $WRONG_SPEAKER finished: $(date) ==="
