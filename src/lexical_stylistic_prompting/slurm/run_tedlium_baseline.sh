#!/bin/bash
# =============================================================================
# run_tedlium_baseline.sh
# Evaluate Whisper medium (no prompting) on TED-LIUM 3 speakers.
# Outputs one CSV per speaker to --output-dir.
#
# Env vars (pass via --export):
#   SPEAKER_ID   — single talk ID to evaluate (required)
#   DATASET_PATH — path to pre-filtered HF dataset on disk (optional;
#                  defaults to data/processed/lexical_stylistic_prompting/tedlium_selected)
#   OUTPUT_DIR   — output directory (optional;
#                  defaults to data/processed/lexical_stylistic_prompting/baseline)
#
# Run all 30 selected speakers (final experiment):
#   while IFS= read -r spk; do
#       sbatch --export=SPEAKER_ID="$spk" src/lexical_stylistic_prompting/slurm/run_tedlium_baseline.sh
#   done < src/lexical_stylistic_prompting/data/speaker_selection/speakers_selected.txt
#
# Run all 739 technical speakers (broad evaluation):
#   First prepare the dataset:
#     uv run src/lexical_stylistic_prompting/data/prepare_dataset.py \
#         --speakers-file src/lexical_stylistic_prompting/data/speaker_selection/speakers_technical.txt \
#         --output-dir data/processed/lexical_stylistic_prompting/tedlium_technical \
#         --match-mode base_name
#   Then get the list of talk IDs and submit:
#     python -c "
#     from datasets import load_from_disk
#     ds = load_from_disk('data/processed/lexical_stylistic_prompting/tedlium_technical')
#     ids = sorted(set(ex['speaker_id'] for ex in ds.select_columns(['speaker_id'])))
#     print('\n'.join(ids))
#     " > /tmp/technical_talk_ids.txt
#     while IFS= read -r spk; do
#         sbatch --export=SPEAKER_ID="$spk",DATASET_PATH=data/processed/lexical_stylistic_prompting/tedlium_technical,OUTPUT_DIR=data/processed/lexical_stylistic_prompting/baseline_technical \
#             src/lexical_stylistic_prompting/slurm/run_tedlium_baseline.sh
#     done < /tmp/technical_talk_ids.txt
#
# Merge per-speaker CSVs afterwards:
#   bash src/lexical_stylistic_prompting/slurm/merge_baseline.sh <output_dir>
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

OUTPUT_DIR="${OUTPUT_DIR:-data/processed/lexical_stylistic_prompting/baseline}"
mkdir -p logs "$OUTPUT_DIR"

echo "=== TED-LIUM baseline started: $(date) ==="
echo "Running on host: $(hostname)"
echo "Speaker: ${SPEAKER_ID:-all}"
echo "Dataset: ${DATASET_PATH:-(default)}"
echo "Output:  $OUTPUT_DIR"

python -m src.lexical_stylistic_prompting.pipeline.baseline_eval \
    --model          openai/whisper-medium \
    --output-dir     "$OUTPUT_DIR" \
    --cache-dir      data/cache/huggingface \
    ${SPEAKER_ID:+--speaker "$SPEAKER_ID"} \
    ${DATASET_PATH:+--dataset-path "$DATASET_PATH"}

echo "=== TED-LIUM baseline finished: $(date) ==="
