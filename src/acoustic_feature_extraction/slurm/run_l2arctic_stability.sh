#!/bin/bash
# =============================================================================
# run_l2arctic_stability.sh
# Run speaker stability analysis on L2-ARCTIC for all three representations
# (wavlm_base_meanstd, wavlm_sv_xvector, wav2vec2_meanstd) with dense k
# checkpoints for smoother per-speaker curves in the stability plots.
#
# Submit with:  sbatch src/acoustic_feature_extraction/slurm/run_l2arctic_stability.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=stability-l2arctic
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu

#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/stability_l2arctic_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/stability_l2arctic_%j.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=0

source .venv/bin/activate

mkdir -p logs

echo "=== L2-ARCTIC stability started: $(date) ==="
echo "Running on host: $(hostname)"

uv run python -m src.acoustic_feature_extraction.pipeline.speaker_stability \
    --dataset      l2arctic \
    --ordering     chronological \
    --ks           1,2,3,4,5,6,7,8,9,10,12,15,20

echo "=== L2-ARCTIC stability finished: $(date) ==="
