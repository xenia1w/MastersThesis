#!/bin/bash
# =============================================================================
# run_profile_discriminability.sh
# Inter-speaker profile discriminability analysis.
#
# Computes pairwise cosine similarities between speaker centroids for all 24
# L2-ARCTIC speakers across multiple encoder layers and both profile extractors
# (wav2vec2 and wavlm).  Outputs a heatmap + CSV per configuration.
#
# A single job — no array needed, runs sequentially per configuration.
#
# Submit:
#   sbatch src/asr_adaptation/slurm/run_profile_discriminability.sh
#
# Monitor: squeue -u $USER
# =============================================================================

#SBATCH --job-name=profile-discrim
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu_short
#SBATCH --account=qu
#SBATCH --chdir=/home/users/x/xenia1w/MastersThesis
#SBATCH --output=/home/users/x/xenia1w/MastersThesis/logs/profile_discrim_%j.out
#SBATCH --error=/home/users/x/xenia1w/MastersThesis/logs/profile_discrim_%j.err

module load cuda/12.8

set -euo pipefail

PROJECT_DIR="/home/users/x/xenia1w/MastersThesis"
cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/cache/huggingface"
export TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

OUTPUT_DIR="data/processed/asr_adaptation/profile_discriminability"
mkdir -p logs "$OUTPUT_DIR"

echo "=== Profile discriminability analysis started: $(date) ==="

# wav2vec2 — last layer (matches FiLM training default)
echo "--- wav2vec2, layer=-1 ---"
python -m src.asr_adaptation.pipeline.profile_discriminability \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    "$OUTPUT_DIR" \
    --extractor     wav2vec2 \
    --profile-layer -1 \
    --cache-dir     data/cache/huggingface

# wav2vec2 — layer sweep (mirrors run_film_layer_sweep.sh)
for LAYER in 4 6 9; do
    echo "--- wav2vec2, layer=${LAYER} ---"
    python -m src.asr_adaptation.pipeline.profile_discriminability \
        --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
        --output-dir    "$OUTPUT_DIR" \
        --extractor     wav2vec2 \
        --profile-layer "$LAYER" \
        --cache-dir     data/cache/huggingface
done

# wav2vec2-base-superb-sid — speaker-ID fine-tuned wav2vec2, same architecture as backbone
# Loads the encoder weights only (classification head is ignored by Wav2Vec2Model.from_pretrained)
echo "--- wav2vec2-base-superb-sid, layer=-1 ---"
python -m src.asr_adaptation.pipeline.profile_discriminability \
    --l2arctic-zip   data/raw/l2arctic_release_v5.0.zip \
    --output-dir     "$OUTPUT_DIR" \
    --extractor      wav2vec2 \
    --profile-layer  -1 \
    --wav2vec2-model superb/wav2vec2-base-superb-sid \
    --cache-dir      data/cache/huggingface

# WavLM — for comparison with the Phase 1 stability analysis
echo "--- wavlm, layer=-1 ---"
python -m src.asr_adaptation.pipeline.profile_discriminability \
    --l2arctic-zip  data/raw/l2arctic_release_v5.0.zip \
    --output-dir    "$OUTPUT_DIR" \
    --extractor     wavlm \
    --profile-layer -1 \
    --cache-dir     data/cache/huggingface

echo "=== Profile discriminability analysis finished: $(date) ==="
