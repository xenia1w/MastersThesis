#!/bin/bash
# =============================================================================
# run_earnings21_profile_window.sh
# One array task per call — transcribe the profile window (first n_profile turns,
# audio [0, split_ts]) with Whisper Medium and NO prompt. Produces the noisy
# transcripts consumed by the transcript_only / transcript_plus_knowledge profile
# builders. Run this ONLY if you prefer the cluster over a local background run;
# afterwards download profile_transcripts/ so the (internet-only) KISSKI build can
# read them.
#
# Submit:
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CALL_IDS_FILE=${CALL_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_earnings21_profile_window.sh
#
# Then download the transcripts to build profiles locally:
#   rsync -avz xenia1w@gateway.hpc.tu-berlin.de:/home/users/x/xenia1w/MastersThesis/\
#     data/processed/lexical_stylistic_prompting/profile_transcripts/ \
#     data/processed/lexical_stylistic_prompting/profile_transcripts/
# =============================================================================

#SBATCH --job-name=earnings21-profile-window
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/earnings21_profile_window_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/earnings21_profile_window_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export WHISPER_CACHE="$PROJECT_DIR/data/cache/whisper"

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/profile_transcripts"

mkdir -p logs "$OUTPUT_DIR" "$WHISPER_CACHE"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 profile-window transcription started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_profile_window \
    --model         medium \
    --download-root "$WHISPER_CACHE" \
    --data-dir      data/raw/earnings21 \
    --output-dir    "$OUTPUT_DIR" \
    --n-profile     20 \
    --call-id       "$CALL_ID"

echo "=== Earnings21 profile-window transcription finished: $(date) ==="
