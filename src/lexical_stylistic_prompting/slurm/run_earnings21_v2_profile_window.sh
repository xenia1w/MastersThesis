#!/bin/bash
# =============================================================================
# run_earnings21_v2_profile_window.sh
# v2 methodology: one array task per call — transcribe the fixed PROFILE window
# [0:00, 5:00] = [0, 300] s with openai-whisper Whisper Medium, no prompt. The noisy
# transcript feeds the transcript-based profiles (transcript_only / _plus_knowledge /
# _metadata_knowledge). metadata_only does NOT need this step.
#
# Loader-independent (slices audio directly), so it is not affected by the segment builder.
#
# Submit:
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CALL_IDS_FILE=${CALL_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_earnings21_v2_profile_window.sh
#
# Then (locally) build profiles from the transcripts:
#   uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \
#     --data-dir data/raw/earnings21 --strategy transcript_only --n-profile 300 \
#     --profiles-dir    data/processed/lexical_stylistic_prompting/v2/profiles \
#     --transcripts-dir data/processed/lexical_stylistic_prompting/v2/profile_transcripts \
#     --skip-existing
# =============================================================================

#SBATCH --job-name=earnings21-v2-profile-window
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/earnings21_v2_profile_window_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/earnings21_v2_profile_window_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export WHISPER_CACHE="$PROJECT_DIR/data/cache/whisper"

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
WINDOW_SECONDS="${WINDOW_SECONDS:-300}"
PROFILE_TAG="${PROFILE_TAG:-300}"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/v2/profile_transcripts"

mkdir -p logs "$OUTPUT_DIR" "$WHISPER_CACHE"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 v2 profile-window started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}, window: [0, ${WINDOW_SECONDS}]s"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_window_profile \
    --data-dir       data/raw/earnings21 \
    --call-id        "$CALL_ID" \
    --window-seconds "$WINDOW_SECONDS" \
    --profile-tag    "$PROFILE_TAG" \
    --model          medium \
    --download-root  "$WHISPER_CACHE" \
    --output-dir     "$OUTPUT_DIR"

echo "=== Earnings21 v2 profile-window finished: $(date) ==="
