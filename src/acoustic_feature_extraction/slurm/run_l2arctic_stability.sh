#!/bin/bash
# =============================================================================
# run_l2arctic_stability.sh
# Run speaker stability analysis on L2-ARCTIC for the wav2vec2_meanstd
# representation with dense k checkpoints for smoother per-speaker curves.
# Results saved to data/processed/stability/l2arctic_wav2vec2_stability/
#
# Submit with:  sbatch src/acoustic_feature_extraction/slurm/run_l2arctic_stability.sh
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=stability-l2arctic
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
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

python -m src.acoustic_feature_extraction.pipeline.speaker_stability \
    --dataset           l2arctic \
    --ordering          chronological \
    --ks                1,2,3,4,5,6,7,8,9,10,12,15,20 \
    --representations   wav2vec2_meanstd \
    --save-root         data/processed/stability/l2arctic_wav2vec2_stability

echo "=== L2-ARCTIC stability finished: $(date) ==="
