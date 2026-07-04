#!/bin/bash
# =============================================================================
# run_earnings21_fullcall_baseline.sh
# One array task per earnings call — evaluates audio from segment 20 onwards
# as a single continuous slice (no per-segment RTTM alignment).
#
# Prerequisites (run on login node before submitting):
#   1. Upload earnings21 data if not already present:
#      rsync -avz data/raw/earnings21/ \
#        xenia1w@gateway.hpc.tu-berlin.de:/home/users/x/xenia1w/MastersThesis/data/raw/earnings21/
#
#   2. Pre-download Whisper medium (needs internet — login node only):
#      HF_HOME=data/cache/huggingface \
#      uv run python -c "
#      from transformers import pipeline
#      pipeline('automatic-speech-recognition', model='openai/whisper-medium',
#               cache_dir='data/cache/huggingface')
#      print('Done')
#      "
#
# Submit:
#   CALL_IDS=data/raw/earnings21/call_ids.txt
#   N=$(( $(wc -l < "$CALL_IDS") - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CALL_IDS_FILE=${CALL_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_earnings21_fullcall_baseline.sh
#
# Merge after all tasks complete:
#   uv run python -c "
#   import glob, pandas as pd
#   dfs = [pd.read_csv(f) for f in sorted(glob.glob(
#       'data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline/baseline_*.csv'))]
#   pd.concat(dfs).to_csv(
#       'data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline/baseline_all.csv',
#       index=False)
#   print(f'Merged {len(dfs)} files')
#   "
# =============================================================================

#SBATCH --job-name=earnings21-fullcall-baseline
#SBATCH --time=00:20:00
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

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

CALL_IDS_FILE="${CALL_IDS_FILE:-data/raw/earnings21/call_ids.txt}"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline"

mkdir -p logs "$OUTPUT_DIR"

CALL_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CALL_IDS_FILE")

echo "=== Earnings21 fullcall baseline started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, call: ${CALL_ID}"

python -m src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_baseline_eval \
    --model      openai/whisper-medium \
    --data-dir   data/raw/earnings21 \
    --output     "${OUTPUT_DIR}/baseline_${CALL_ID}.csv" \
    --cache-dir  data/cache/huggingface \
    --n-profile  20 \
    --call-id    "$CALL_ID"

echo "=== Earnings21 fullcall baseline finished: $(date) ==="
