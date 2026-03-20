#!/bin/bash
# =============================================================================
# run_lora_speaker.sh
# Per-speaker LoRA fine-tuning as a SLURM array job.
# Each array task handles one speaker from SPEAKERS list below.
#
# Submit with:
#   sbatch --array=0-17 src/asr_adaptation/slurm/run_lora_speaker.sh
#
#   Or a subset (e.g. first 3 speakers for testing):
#   sbatch --array=0-2  src/asr_adaptation/slurm/run_lora_speaker.sh
#
# Monitor with: squeue -u $USER
# Each job writes to logs/lora_<speaker>_<jobid>.out
# =============================================================================

#SBATCH --job-name=asr-lora
#SBATCH --time=06:00:00          # Adjust — ~30 min per speaker with 500 utterances
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lora_%x_%j.out
#SBATCH --error=logs/lora_%x_%j.err

# TODO: uncomment and set your cluster's partition
# #SBATCH --partition=gpu

# TODO: uncomment if your cluster requires an account
# #SBATCH --account=YOUR_ACCOUNT

# TODO: if your cluster uses environment modules for CUDA, load them here:
# module load cuda/12.1
# module load python/3.13

set -euo pipefail

# Full list of L2-ARCTIC speakers (18 total)
SPEAKERS=(ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI THV TNI)

# Pick the speaker for this array task
SPEAKER="${SPEAKERS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1      # models should already be cached after baseline run

source .venv/bin/activate

mkdir -p logs \
         data/processed/asr_adaptation/lora_weights \
         data/processed/asr_adaptation/adaptation_results

echo "=== LoRA training for speaker $SPEAKER started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.lora_train \
    --speaker       "$SPEAKER" \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation \
    --cache-dir     data/cache/huggingface

echo "=== LoRA training for speaker $SPEAKER finished: $(date) ==="
