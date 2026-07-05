#!/bin/bash
# =============================================================================
# run_earnings21_fullcall_baseline.sh
# One array task per earnings call — full-call baseline (no prompt) with
# openai-whisper Whisper Medium. Must be run with the SAME code path as the
# metadata_only run so the two differ only by the injected prompt.
#
# Prerequisites (once):
#   Pre-download Whisper medium weights on a login node so compute nodes run offline:
#     XDG_CACHE_HOME=data/cache uv run python -c \
#       "import whisper; whisper.load_model('medium', download_root='data/cache/whisper')"
#
# Submit:
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CALL_IDS_FILE=${CALL_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_earnings21_fullcall_baseline.sh
#
# Merge after all tasks complete. The glob is numeric (baseline_<callid>.csv) so it never
# matches the aggregate baseline_all.csv — safe to re-run without folding it into itself:
#   uv run python -c "
#   import glob, pandas as pd
#   d='data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline'
#   dfs=[pd.read_csv(f) for f in sorted(glob.glob(d+'/baseline_[0-9]*.csv'))]
#   pd.concat(dfs).to_csv(d+'/baseline_all.csv', index=False)
#   print('Merged', len(dfs), 'files')
#   "
# NOTE: baseline_all.csv REPLACES the old HF baseline. Back it up first if you want to keep it:
#   mv .../earnings21_fullcall_baseline/baseline_all.csv .../baseline_all.hf.bak
# =============================================================================

#SBATCH --job-name=earnings21-fullcall-baseline
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/earnings21_fullcall_baseline_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/earnings21_fullcall_baseline_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export WHISPER_CACHE="$PROJECT_DIR/data/cache/whisper"

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline"

mkdir -p logs "$OUTPUT_DIR" "$WHISPER_CACHE"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 fullcall baseline started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval \
    --model         medium \
    --download-root "$WHISPER_CACHE" \
    --data-dir      data/raw/earnings21 \
    --output        "${OUTPUT_DIR}/baseline_${CALL_ID}.csv" \
    --strategy      baseline \
    --n-profile     20 \
    --call-id       "$CALL_ID"

echo "=== Earnings21 fullcall baseline finished: $(date) ==="
