#!/bin/bash
# =============================================================================
# run_earnings21_fullcall_metadata_only.sh
# One array task per earnings call — full-call metadata_only evaluation. The LLM
# keyword profile is injected into openai-whisper Whisper Medium as initial_prompt
# (carried to every 30 s window). Same code path as the baseline run.
#
# Prerequisites:
#   1. Build profiles locally and upload them:
#      uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \
#          --data-dir data/raw/earnings21 --strategy metadata_only --n-profile 20 --skip-existing
#      rsync -avz data/processed/lexical_stylistic_prompting/profiles/ \
#        xenia1w@gateway.hpc.tu-berlin.de:/home/users/x/xenia1w/MastersThesis/data/processed/lexical_stylistic_prompting/profiles/
#   2. Pre-download Whisper medium weights on a login node (see baseline script).
#
# Submit:
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CALL_IDS_FILE=${CALL_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_earnings21_fullcall_metadata_only.sh
#
# Merge after all tasks complete:
# The glob is numeric (prompted_<callid>.csv) so it never matches the aggregate prompted_all.csv:
#   uv run python -c "
#   import glob, pandas as pd
#   d='data/processed/lexical_stylistic_prompting/earnings21_fullcall_metadata_only'
#   dfs=[pd.read_csv(f) for f in sorted(glob.glob(d+'/prompted_[0-9]*.csv'))]
#   pd.concat(dfs).to_csv(d+'/prompted_all.csv', index=False)
#   print('Merged', len(dfs), 'files')
#   "
# =============================================================================

#SBATCH --job-name=earnings21-fullcall-metadata
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/earnings21_fullcall_metadata_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/earnings21_fullcall_metadata_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export WHISPER_CACHE="$PROJECT_DIR/data/cache/whisper"

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
PROMPT_FORMAT="${PROMPT_FORMAT:-list}"
SUFFIX=""; [ "$PROMPT_FORMAT" = "prose" ] && SUFFIX="_prose"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/earnings21_fullcall_metadata_only${SUFFIX}"
PROFILES_DIR="data/processed/lexical_stylistic_prompting/profiles"

mkdir -p logs "$OUTPUT_DIR" "$WHISPER_CACHE"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 fullcall metadata_only started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval \
    --model         medium \
    --download-root "$WHISPER_CACHE" \
    --data-dir      data/raw/earnings21 \
    --profiles-dir  "$PROFILES_DIR" \
    --output        "${OUTPUT_DIR}/prompted_${CALL_ID}.csv" \
    --strategy      metadata_only \
    --prompt-format "$PROMPT_FORMAT" \
    --n-profile     20 \
    --call-id       "$CALL_ID"

echo "=== Earnings21 fullcall metadata_only finished: $(date) ==="
