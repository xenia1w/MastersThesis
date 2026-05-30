#!/bin/bash
# =============================================================================
# run_tedlium_baseline.sh
# Evaluate Whisper medium (no prompting) on TED-LIUM 3 test segments.
# Outputs per-segment WER CSV to data/processed/lexical_stylistic_prompting/baseline/.
#
# Run for a single speaker (recommended — one job per speaker):
#   sbatch --export=SPEAKER_ID=AlGore src/lexical_stylistic_prompting/slurm/run_tedlium_baseline.sh
#
# Run all speakers in parallel (loop over manifest):
#   for spk in $(tail -n +2 data/processed/lexical_stylistic_prompting/tedlium_manifest.csv | cut -d, -f1); do
#       sbatch --export=SPEAKER_ID=$spk src/lexical_stylistic_prompting/slurm/run_tedlium_baseline.sh
#   done
#
# Merge per-speaker CSVs afterwards:
#   head -1 data/processed/lexical_stylistic_prompting/baseline/tedlium_baseline_AlGore.csv > tedlium_baseline.csv
#   tail -n +2 -q data/processed/lexical_stylistic_prompting/baseline/tedlium_baseline_*.csv >> tedlium_baseline.csv
# =============================================================================

#SBATCH --job-name=tedlium-baseline
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/tedlium_baseline_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/tedlium_baseline_%j.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

mkdir -p logs data/processed/lexical_stylistic_prompting/baseline

echo "=== TED-LIUM baseline started: $(date) ==="
echo "Running on host: $(hostname)"
echo "Speaker: ${SPEAKER_ID:-all}"

python -m src.lexical_stylistic_prompting.pipeline.baseline_eval \
    --model      openai/whisper-medium \
    --output-dir data/processed/lexical_stylistic_prompting/baseline \
    --cache-dir  data/cache/huggingface \
    ${SPEAKER_ID:+--speaker "$SPEAKER_ID"}

echo "=== TED-LIUM baseline finished: $(date) ==="
