#!/bin/bash
# =============================================================================
# run_film_layer_sweep.sh
# Layer sweep for wav2vec2 profile extraction — tests which encoder layer
# produces the most speaker-discriminative profile for FiLM conditioning.
#
# Runs on a representative subset of 6 speakers (indices 0-5).
# Layer -1 (last) is already covered by run_film_speaker.sh.
# Sweep: layers 4, 6, 9
#
# Submit all three layers (after the main FiLM job JOBID finishes):
#   sbatch --dependency=afterok:JOBID --export=ALL,PROFILE_LAYER=4 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
#   sbatch --dependency=afterok:JOBID --export=ALL,PROFILE_LAYER=6 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
#   sbatch --dependency=afterok:JOBID --export=ALL,PROFILE_LAYER=9 --array=0-5 src/asr_adaptation/slurm/run_film_layer_sweep.sh
#
# Monitor: squeue -u $USER
# =============================================================================

#SBATCH --job-name=asr-film-sweep
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/film_sweep_layer%x_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/film_sweep_layer%x_%j.err

module load cuda/12.8

set -euo pipefail

# PROFILE_LAYER must be passed via --export=ALL,PROFILE_LAYER=N
if [[ -z "${PROFILE_LAYER:-}" ]]; then
    echo "ERROR: PROFILE_LAYER is not set. Pass it via --export=ALL,PROFILE_LAYER=N" >&2
    exit 1
fi

SPEAKERS=(ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI THV TNI TXHC YBAA YDCK YKWK ZHAA TLV)
SPEAKER="${SPEAKERS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

OUTPUT_DIR="data/processed/asr_adaptation_film_layer${PROFILE_LAYER}"

mkdir -p logs \
         "${OUTPUT_DIR}/film_lora_weights" \
         "${OUTPUT_DIR}/film_adaptation_results"

echo "=== FiLM layer sweep (layer=${PROFILE_LAYER}) for speaker $SPEAKER started: $(date) ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.film_train \
    --speaker-id        "$SPEAKER" \
    --l2arctic-zip      data/raw/l2arctic_release_v5.0.zip \
    --output-dir        "$OUTPUT_DIR" \
    --cache-dir         data/cache/huggingface \
    --profile-extractor wav2vec2 \
    --profile-layer     "$PROFILE_LAYER"

echo "=== FiLM layer sweep (layer=${PROFILE_LAYER}) for speaker $SPEAKER finished: $(date) ==="
