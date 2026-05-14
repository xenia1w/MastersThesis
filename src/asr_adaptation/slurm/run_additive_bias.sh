#!/bin/bash
# =============================================================================
# run_additive_bias.sh
# Per-speaker LoRA + additive bias fine-tuning (wav2vec2 speaker profile).
# Results saved to data/processed/results_lora_additive_bias/
#
# Submit all 24 speakers:
#   sbatch --array=0-23 src/asr_adaptation/slurm/run_additive_bias.sh
#
# Submit a single speaker for testing (ABA = index 0):
#   sbatch --array=0 src/asr_adaptation/slurm/run_additive_bias.sh
#
# Monitor: squeue -u $USER
# =============================================================================

#SBATCH --job-name=asr-lora-additive-bias
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-23
#SBATCH --partition=gpu
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/additive_bias_%x_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/additive_bias_%x_%j.err

module load cuda/12.8

set -euo pipefail

SPEAKERS=(ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI THV TNI TXHC YBAA YDCK YKWK ZHAA TLV)
SPEAKER="${SPEAKERS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

mkdir -p logs \
         data/processed/results_lora_additive_bias/lora_weights \
         data/processed/results_lora_additive_bias/adaptation_results

echo "=== LoRA + additive bias training for speaker $SPEAKER started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.lora_train \
    --speaker            "$SPEAKER" \
    --l2arctic-zip       data/raw/l2arctic_release_v5.0.zip \
    --output-dir         data/processed/results_lora_additive_bias \
    --cache-dir          data/cache/huggingface \
    --profile-extractor  wav2vec2

echo "=== LoRA + additive bias training for speaker $SPEAKER finished: $(date) ==="
