#!/bin/bash
# =============================================================================
# run_tedlium_baseline_chunked_small.sh
# Each array task processes CHUNK_SIZE speakers in a single Python process
# (model loaded once, amortised over the whole chunk).
#
# Usage:
#   CHUNK_SIZE=25
#   TALK_IDS=data/processed/lexical_stylistic_prompting/technical_talk_ids.txt
#   TOTAL=$(wc -l < "$TALK_IDS")
#   N=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE - 1 ))
#   sbatch --array=0-${N} \
#          --export=ALL,CHUNK_SIZE=${CHUNK_SIZE},TALK_IDS_FILE=${TALK_IDS} \
#          src/lexical_stylistic_prompting/slurm/run_tedlium_baseline_chunked.sh
#
# Each task writes one CSV per speaker to --output-dir, so parallel tasks
# never write to the same file.  Merge afterwards with:
#   python -c "
#   import glob, pandas as pd
#   dfs = [pd.read_csv(f) for f in glob.glob('data/processed/lexical_stylistic_prompting/baseline_technical_small/tedlium_baseline_*.csv')]
#   pd.concat(dfs).to_csv('data/processed/lexical_stylistic_prompting/baseline_technical_small/tedlium_baseline_all.csv', index=False)
#   print(f'Merged {len(dfs)} files')
#   "
# =============================================================================

#SBATCH --job-name=tedlium-small
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/tedlium_small_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/tedlium_small_%A_%a.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

CHUNK_SIZE="${CHUNK_SIZE:-25}"
TALK_IDS_FILE="${TALK_IDS_FILE:-data/processed/lexical_stylistic_prompting/technical_talk_ids.txt}"
OUTPUT_DIR="data/processed/lexical_stylistic_prompting/baseline_technical_small"

mkdir -p logs "$OUTPUT_DIR"

# Compute the line range for this task (1-based for sed)
START_LINE=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1 ))
END_LINE=$(( (SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE ))

# Extract this chunk's speaker IDs into a temp file
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT
sed -n "${START_LINE},${END_LINE}p" "$TALK_IDS_FILE" > "$TMPFILE"

N_SPEAKERS=$(wc -l < "$TMPFILE")
echo "=== TED-LIUM chunked baseline (whisper-small) started: $(date) ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}, lines ${START_LINE}-${END_LINE}, speakers: ${N_SPEAKERS}"
cat "$TMPFILE"

python -m src.lexical_stylistic_prompting.pipeline.baseline_eval \
    --model          openai/whisper-small \
    --output-dir     "$OUTPUT_DIR" \
    --cache-dir      data/cache/huggingface \
    --dataset-path   data/processed/lexical_stylistic_prompting/tedlium_technical \
    --speakers-file  "$TMPFILE"

echo "=== TED-LIUM chunked baseline (whisper-small) finished: $(date) ==="
