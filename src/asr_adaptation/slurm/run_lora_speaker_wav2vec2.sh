#!/bin/bash
# =============================================================================
# run_lora_speaker_wav2vec2.sh
# Per-speaker LoRA fine-tuning using Wav2Vec2 acoustic profile extractor.
# Identical to run_lora_speaker.sh except --profile-extractor wav2vec2 and
# a separate output directory so WavLM results are not overwritten.
#
# Submit all 24 speakers:
#   sbatch --array=0-23 src/asr_adaptation/slurm/run_lora_speaker_wav2vec2.sh
#
# Submit a subset for testing (e.g. first 3 speakers):
#   sbatch --array=0-2  src/asr_adaptation/slurm/run_lora_speaker_wav2vec2.sh
#
# Monitor: squeue -u $USER
# =============================================================================

#SBATCH --job-name=asr-lora-w2v2
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-23
#SBATCH --partition=gpu
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/lora_w2v2_%x_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/lora_w2v2_%x_%j.err

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
         data/processed/asr_adaptation_wav2vec2/lora_weights \
         data/processed/asr_adaptation_wav2vec2/adaptation_results

echo "=== LoRA (wav2vec2 profile) training for speaker $SPEAKER started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.lora_train \
    --speaker            "$SPEAKER" \
    --l2arctic-zip       data/raw/l2arctic_release_v5.0.zip \
    --output-dir         data/processed/asr_adaptation_wav2vec2 \
    --cache-dir          data/cache/huggingface \
    --profile-extractor  wav2vec2

echo "=== LoRA (wav2vec2 profile) training for speaker $SPEAKER finished: $(date) ==="
