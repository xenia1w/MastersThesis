#!/bin/bash
# =============================================================================
# run_baseline.sh
# Evaluate unadapted wav2vec2-base-960h on all L2-ARCTIC and SAA speakers.
# Outputs WER CSVs to data/processed/asr_adaptation/baseline_wer/.
#
# Submit with:  sbatch src/asr_adaptation/slurm/run_baseline.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=asr-baseline
#SBATCH --time=04:00:00          # Adjust if needed — ~10 min per speaker
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

#SBATCH --partition=gpu_short

#SBATCH --account=qu

#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/baseline_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/baseline_%j.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

# Point HuggingFace to the project's local model cache (avoids re-downloading)
export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=0      # set to 1 once models are cached on the cluster

source .venv/bin/activate

mkdir -p logs data/processed/asr_adaptation/baseline_wer

echo "=== Baseline evaluation started: $(date) ==="
echo "Running on host: $(hostname)"

python -m src.asr_adaptation.pipeline.baseline_eval \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --saa-zip       data/raw/archive.zip \
    --output-dir    data/processed/asr_adaptation/baseline_wer \
    --cache-dir     data/cache/huggingface

echo "=== Baseline evaluation finished: $(date) ==="
