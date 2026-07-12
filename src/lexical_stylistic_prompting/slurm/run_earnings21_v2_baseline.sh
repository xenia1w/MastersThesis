#!/bin/bash
# =============================================================================
# run_earnings21_v2_baseline.sh
# v2 methodology: fixed wall-clock windows. One array task per call — baseline
# (no prompt) transcription of the EVAL window [5:00, 15:00] = [300, 900] s with
# openai-whisper Whisper Medium. Saves only the hypothesis; WER/entity-EER are
# scored locally afterwards against references cut from hand-annotated 5:00/15:00
# boundaries. Same code path as the prompted v2 runs (differ only by the prompt).
#
# All 44 calls are >= 15 min, so the [300, 900] window fits every call.
#
# Prerequisites (once):
#   Pre-download Whisper medium weights on a login node:
#     XDG_CACHE_HOME=data/cache uv run python -c \
#       "import whisper; whisper.load_model('medium', download_root='data/cache/whisper')"
#
# Submit:
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CALL_IDS_FILE=${CALL_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_earnings21_v2_baseline.sh
#
# Merge after all tasks complete (numeric glob ignores the aggregate):
#   uv run python -c "
#   import glob, pandas as pd
#   d='data/processed/lexical_stylistic_prompting/v2/earnings21_window_baseline'
#   dfs=[pd.read_csv(f) for f in sorted(glob.glob(d+'/baseline_[0-9]*.csv'))]
#   pd.concat(dfs).to_csv(d+'/baseline_all.csv', index=False)
#   print('Merged', len(dfs), 'files')
#   "
# =============================================================================

#SBATCH --job-name=earnings21-v2-baseline
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/earnings21_v2_baseline_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/earnings21_v2_baseline_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export WHISPER_CACHE="$PROJECT_DIR/data/cache/whisper"

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
EVAL_START="${EVAL_START:-300}"
EVAL_END="${EVAL_END:-900}"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/v2/earnings21_window_baseline"

mkdir -p logs "$OUTPUT_DIR" "$WHISPER_CACHE"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 v2 baseline started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}, window: [${EVAL_START}, ${EVAL_END}]s"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_window_eval \
    --data-dir      data/raw/earnings21 \
    --call-id       "$CALL_ID" \
    --eval-start    "$EVAL_START" \
    --eval-end      "$EVAL_END" \
    --strategy      baseline \
    --model         medium \
    --download-root "$WHISPER_CACHE" \
    --output        "${OUTPUT_DIR}/baseline_${CALL_ID}.csv"

echo "=== Earnings21 v2 baseline finished: $(date) ==="
