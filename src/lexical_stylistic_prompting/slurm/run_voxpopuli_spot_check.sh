#!/bin/bash
# =============================================================================
# run_voxpopuli_spot_check.sh
# Run Whisper medium on the first 10 eligible VoxPopuli English speakers to
# check baseline WER before committing to the full dataset.
#
# Usage:
#   sbatch src/lexical_stylistic_prompting/slurm/run_voxpopuli_spot_check.sh
#
# Prerequisites — download the dataset on the login node first (needs internet):
#   uv run python -c "
#   from datasets import load_dataset
#   for split in ('train', 'validation', 'test'):
#       load_dataset('facebook/voxpopuli', 'en', split=split,
#                    cache_dir='data/cache/huggingface', trust_remote_code=True)
#   print('Done')
#   "
# =============================================================================

#SBATCH --job-name=voxpopuli-spot
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/voxpopuli_spot_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/voxpopuli_spot_%j.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

OUTPUT_DIR="data/processed/lexical_stylistic_prompting/voxpopuli_spot_check"
mkdir -p logs "$OUTPUT_DIR"

echo "=== VoxPopuli spot-check started: $(date) ==="

python -m src.lexical_stylistic_prompting.pipeline.voxpopuli_baseline_eval \
    --model        openai/whisper-medium \
    --output-dir   "$OUTPUT_DIR" \
    --cache-dir    data/cache/huggingface \
    --n-speakers   10

echo "=== VoxPopuli spot-check finished: $(date) ==="
