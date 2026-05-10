#!/bin/bash
# =============================================================================
# run_film_speaker.sh
# Per-speaker FiLM-conditioned LoRA fine-tuning using Wav2Vec2 acoustic profile.
#
# Submit all 24 speakers:
#   sbatch --array=0-23 src/asr_adaptation/slurm/run_film_speaker.sh
#
# Submit a subset for testing (e.g. first 3 speakers):
#   sbatch --array=0-2  src/asr_adaptation/slurm/run_film_speaker.sh
#
# To use a separate (higher) learning rate for the FiLM MLP:
#   sbatch --array=0-23 --export=ALL,FILM_LR=1e-4 src/asr_adaptation/slurm/run_film_speaker.sh
#
# Monitor: squeue -u $USER
# =============================================================================

#SBATCH --job-name=asr-film-w2v2
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-23
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/film_%x_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/film_%x_%j.err

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
         data/processed/asr_adaptation_film/film_lora_weights \
         data/processed/asr_adaptation_film/film_adaptation_results

echo "=== FiLM training for speaker $SPEAKER started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

FILM_LR_ARG=""
if [[ -n "${FILM_LR:-}" ]]; then
    FILM_LR_ARG="--film-lr ${FILM_LR}"
fi

# shellcheck disable=SC2086
python -m src.asr_adaptation.pipeline.film_train \
    --speaker-id        "$SPEAKER" \
    --l2arctic-zip      data/raw/l2arctic_release_v5.0.zip \
    --output-dir        data/processed/asr_adaptation_film \
    --cache-dir         data/cache/huggingface \
    --profile-extractor wav2vec2 \
    $FILM_LR_ARG

echo "=== FiLM training for speaker $SPEAKER finished: $(date) ==="
