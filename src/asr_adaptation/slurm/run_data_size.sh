#!/bin/bash
# =============================================================================
# run_data_size.sh
# Data size sweep for RQ1.3: train LoRA on N utterances per speaker,
# vary N across [20, 50, 100, 200, 300, 500], repeat 3 seeds each.
#
# Total tasks = 4 speakers × 6 N-values × 1 seed = 24 jobs
#
# Submit with:
#   sbatch --array=0-23 src/asr_adaptation/slurm/run_data_size.sh
#
#   For a quick test (first 6 tasks = speaker 0, all N-values):
#   sbatch --array=0-5   src/asr_adaptation/slurm/run_data_size.sh
#
# Monitor with: squeue -u $USER
# =============================================================================

#SBATCH --job-name=asr-sweep
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/sweep_%A_%a.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/sweep_%A_%a.err

module load cuda/12.8

set -euo pipefail

SPEAKERS=(RRBI ERMS LXC HQTV)
N_VALUES=(20 50 100 200 300 500)
SEEDS=(0)

N_SPEAKERS=${#SPEAKERS[@]}   # 18
N_N=${#N_VALUES[@]}          # 7
N_SEEDS=${#SEEDS[@]}         # 3

# Decode flat task index into (speaker, n_train, seed)
TASK_ID=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( TASK_ID % N_SEEDS ))
N_IDX=$(( (TASK_ID / N_SEEDS) % N_N ))
SPEAKER_IDX=$(( TASK_ID / (N_SEEDS * N_N) ))

SPEAKER="${SPEAKERS[$SPEAKER_IDX]}"
N_TRAIN="${N_VALUES[$N_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

mkdir -p logs data/processed/asr_adaptation/data_size_curves

echo "=== Data size sweep | speaker=$SPEAKER n_train=$N_TRAIN seed=$SEED | $(date) ==="
echo "Array task: $SLURM_ARRAY_TASK_ID | Host: $(hostname)"

python -m src.asr_adaptation.pipeline.data_size_analysis run \
    --speaker       "$SPEAKER" \
    --n-train       "$N_TRAIN" \
    --seed          "$SEED" \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    data/processed/asr_adaptation/data_size_curves \
    --cache-dir     data/cache/huggingface

echo "=== Done | speaker=$SPEAKER n_train=$N_TRAIN seed=$SEED | $(date) ==="
