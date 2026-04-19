#!/bin/bash
# =============================================================================
# run_lora_speaker.sh
# Per-speaker LoRA fine-tuning as a SLURM array job.
# Each array task handles one speaker from SPEAKERS list below.
#
# Submit with:
#   sbatch --array=0-23 src/asr_adaptation/slurm/run_lora_speaker.sh
#
#   Or a subset (e.g. first 3 speakers for testing):
#   sbatch --array=0-2  src/asr_adaptation/slurm/run_lora_speaker.sh
#
# Monitor with: squeue -u $USER
# Each job writes to logs/lora_<speaker>_<jobid>.out
# =============================================================================

#SBATCH --job-name=asr-lora
#SBATCH --time=04:00:00          # gpu_short max walltime is 4h; ~30 min per speaker with 500 utterances
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-23
#SBATCH --output=logs/lora_%x_%j.out
#SBATCH --error=logs/lora_%x_%j.err

#SBATCH --partition=gpu

#SBATCH --account=qu

#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/lora_%x_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/lora_%x_%j.err

module load cuda/12.8

set -euo pipefail

# Full list of L2-ARCTIC speakers (24 total)
SPEAKERS=(ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI THV TNI TXHC YBAA YDCK YKWK ZHAA TLV)

# Pick the speaker for this array task
SPEAKER="${SPEAKERS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1      # models should already be cached after baseline run

source .venv/bin/activate

mkdir -p logs \
         data/processed/asr_adaptation/wavlm_lora_weights \
         data/processed/asr_adaptation/wavlm_adaptation_results

echo "=== LoRA training for speaker $SPEAKER started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.lora_train \
    --speaker       "$SPEAKER" \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation/wavlm_lora \
    --cache-dir     data/cache/huggingface

echo "=== LoRA training for speaker $SPEAKER finished: $(date) ==="
