#!/bin/bash
# =============================================================================
# run_earnings21_v2_prompted.sh
# v2 methodology: one array task per call — prompted transcription of the EVAL window
# [5:00, 15:00] = [300, 900] s with openai-whisper Whisper Medium, injecting the call's
# profile as initial_prompt. Parametrised by STRATEGY (and PROMPT_FORMAT). Saves only the
# hypothesis; WER/entity-EER are scored locally against hand-annotated references.
#
# Prerequisites:
#   - Profiles built into v2/profiles/<strategy>/<call_id>_300.json (build_earnings21_profiles.py
#     with --profiles-dir v2/profiles --n-profile 300), then uploaded to the cluster.
#   - For transcript_* strategies, run_earnings21_v2_profile_window.sh first (local profile build).
#
# Submit (one strategy per invocation):
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   for S in metadata_only transcript_only transcript_plus_knowledge transcript_metadata_knowledge; do
#     sbatch --array=0-${N} \
#            --export=ALL,CALL_IDS_FILE=${CALL_IDS},STRATEGY=${S},PROMPT_FORMAT=list \
#            src/lexical_stylistic_prompting/slurm/run_earnings21_v2_prompted.sh
#   done
#   # prose variant: set PROMPT_FORMAT=prose (profiles must be built with --prompt-format prose)
#
# Merge per strategy after tasks complete:
#   uv run python -c "
#   import glob, pandas as pd
#   d='data/processed/lexical_stylistic_prompting/v2/earnings21_window_transcript_only'
#   dfs=[pd.read_csv(f) for f in sorted(glob.glob(d+'/prompted_[0-9]*.csv'))]
#   pd.concat(dfs).to_csv(d+'/prompted_all.csv', index=False); print('Merged', len(dfs))
#   "
# =============================================================================

#SBATCH --job-name=earnings21-v2-prompted
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/earnings21_v2_prompted_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/earnings21_v2_prompted_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export WHISPER_CACHE="$PROJECT_DIR/data/cache/whisper"

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
STRATEGY="${STRATEGY:?set STRATEGY (metadata_only|transcript_only|transcript_plus_knowledge|transcript_metadata_knowledge)}"
PROMPT_FORMAT="${PROMPT_FORMAT:-list}"
EVAL_START="${EVAL_START:-300}"
EVAL_END="${EVAL_END:-900}"
PROFILE_TAG="${PROFILE_TAG:-300}"
SUFFIX=""; [ "$PROMPT_FORMAT" = "prose" ] && SUFFIX="_prose"
PROFILES_DIR="data/processed/lexical_stylistic_prompting/v2/profiles"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/v2/earnings21_window_${STRATEGY}${SUFFIX}"

mkdir -p logs "$OUTPUT_DIR" "$WHISPER_CACHE"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 v2 prompted started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}, strategy: ${STRATEGY}, format: ${PROMPT_FORMAT}, window: [${EVAL_START}, ${EVAL_END}]s"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_window_eval \
    --data-dir      data/raw/earnings21 \
    --call-id       "$CALL_ID" \
    --eval-start    "$EVAL_START" \
    --eval-end      "$EVAL_END" \
    --strategy      "$STRATEGY" \
    --profiles-dir  "$PROFILES_DIR" \
    --profile-tag   "$PROFILE_TAG" \
    --prompt-format "$PROMPT_FORMAT" \
    --model         medium \
    --download-root "$WHISPER_CACHE" \
    --output        "${OUTPUT_DIR}/prompted_${CALL_ID}.csv"

echo "=== Earnings21 v2 prompted finished: $(date) ==="
