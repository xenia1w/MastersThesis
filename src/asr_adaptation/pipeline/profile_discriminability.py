"""Inter-speaker discriminability analysis for acoustic profiles.

Computes pairwise cosine similarities between speaker centroids to check
whether the profiles used for FiLM conditioning are actually separable
across speakers.  A high mean inter-speaker similarity (e.g. > 0.95) means
the centroids cluster too tightly for FiLM to distinguish speakers, which
would explain a flat wrong-speaker WER result regardless of learning rate.

Outputs:
  {output_dir}/profile_discriminability_{extractor}_layer{layer}.csv
      One row per speaker pair: speaker_a, speaker_b, cosine_similarity
  {output_dir}/profile_discriminability_{extractor}_layer{layer}.png
      Heatmap of the full pairwise similarity matrix + histogram of off-diagonal values

Run locally (fast, CPU, 3 speakers for a quick sanity check):
  uv run python -m src.asr_adaptation.pipeline.profile_discriminability \\
      --l2arctic-zip data/raw/l2arctic_release_v5.0.zip \\
      --output-dir   data/processed/asr_adaptation/profile_discriminability \\
      --speakers     ABA ASI BWC

Run all speakers + multiple layers on cluster:
  sbatch src/asr_adaptation/slurm/run_profile_discriminability.sh
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from src.asr_adaptation.data.l2arctic_transcriptions import list_l2arctic_samples_with_transcripts
from src.asr_adaptation.data.speaker_embeddings import compute_speaker_centroid
from src.asr_adaptation.data.wav2vec2_speaker_embeddings import compute_speaker_centroid_wav2vec2
from src.asr_adaptation.pipeline.lora_train import _split_samples

_ALL_SPEAKERS = [
    "ABA", "ASI", "BWC", "EBVS", "ERMS", "HJK", "HKK", "HQTV", "LXC",
    "MBMPS", "NCC", "NJS", "PNV", "RRBI", "SKA", "SVBI", "THV", "TNI",
    "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA", "TLV",
]
_N_EVAL = 100  # must match training split so centroids are computed from the same utterances


def _compute_centroids(
    speakers: list[str],
    l2arctic_zip: str,
    device: torch.device,
    extractor: str,
    profile_layer: int,
    cache_dir: str | None,
    wavlm_model: str,
) -> dict[str, torch.Tensor]:
    """Return {speaker_id: L2-normalised centroid} for all speakers."""
    centroids: dict[str, torch.Tensor] = {}
    for speaker_id in speakers:
        logger.info(f"Computing centroid for {speaker_id} ...")
        all_samples = list_l2arctic_samples_with_transcripts(l2arctic_zip, speaker_id)
        train_samples, _ = _split_samples(all_samples, _N_EVAL, None, seed=0)

        if extractor == "wavlm":
            c = compute_speaker_centroid(
                train_samples, device=device, model_name=wavlm_model, cache_dir=cache_dir
            )
        else:
            c = compute_speaker_centroid_wav2vec2(
                train_samples, device=device, profile_layer=profile_layer, cache_dir=cache_dir
            )
        centroids[speaker_id] = c.cpu()
    return centroids


def _pairwise_cosine(centroids: dict[str, torch.Tensor]) -> tuple[list[str], np.ndarray]:
    """Return (ordered speaker list, NxN cosine similarity matrix)."""
    speakers = list(centroids.keys())
    n = len(speakers)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i, s_i in enumerate(speakers):
        for j, s_j in enumerate(speakers):
            sim = F.cosine_similarity(centroids[s_i].unsqueeze(0), centroids[s_j].unsqueeze(0)).item()
            matrix[i, j] = sim
    return speakers, matrix


def _save_csv(speakers: list[str], matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker_a", "speaker_b", "cosine_similarity"])
        for i, j in combinations(range(len(speakers)), 2):
            writer.writerow([speakers[i], speakers[j], round(float(matrix[i, j]), 6)])
    logger.info(f"Saved pairwise CSV → {path}")


def _plot(speakers: list[str], matrix: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(speakers)

    # Off-diagonal values only (exclude self-similarity = 1.0)
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]
    mean_sim = float(off_diag.mean())
    std_sim = float(off_diag.std())
    min_sim = float(off_diag.min())
    max_sim = float(off_diag.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title}\ninter-speaker cosine similarity  (mean={mean_sim:.4f}, std={std_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f})")

    # Heatmap
    ax = axes[0]
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(speakers, rotation=90, fontsize=7)
    ax.set_yticklabels(speakers, fontsize=7)
    ax.set_title("Pairwise cosine similarity matrix")
    fig.colorbar(im, ax=ax)

    # Histogram of off-diagonal values
    ax2 = axes[1]
    ax2.hist(off_diag, bins=30, edgecolor="black", alpha=0.8)
    ax2.axvline(mean_sim, color="red", linestyle="--", label=f"mean={mean_sim:.4f}")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Count (speaker pairs)")
    ax2.set_title("Distribution of inter-speaker similarities")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot → {path}")


def run_profile_discriminability(
    l2arctic_zip: str,
    output_dir: str | Path,
    speakers: list[str] | None = None,
    extractor: str = "wav2vec2",
    profile_layer: int = -1,
    cache_dir: str | None = None,
    wavlm_model: str = "microsoft/wavlm-base-plus",
) -> None:
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speakers = speakers or _ALL_SPEAKERS
    layer_tag = f"layer{profile_layer}" if profile_layer >= 0 else "layerlast"
    stem = f"profile_discriminability_{extractor}_{layer_tag}"

    logger.info(f"Computing centroids for {len(speakers)} speakers | extractor={extractor} layer={profile_layer}")
    centroids = _compute_centroids(speakers, l2arctic_zip, device, extractor, profile_layer, cache_dir, wavlm_model)

    ordered_speakers, matrix = _pairwise_cosine(centroids)

    mask = ~np.eye(len(ordered_speakers), dtype=bool)
    off_diag = matrix[mask]
    logger.info(
        f"Inter-speaker cosine similarity — "
        f"mean={off_diag.mean():.4f}  std={off_diag.std():.4f}  "
        f"min={off_diag.min():.4f}  max={off_diag.max():.4f}"
    )

    _save_csv(ordered_speakers, matrix, output_dir / f"{stem}.csv")
    _plot(
        ordered_speakers, matrix,
        output_dir / f"{stem}.png",
        title=f"Profile discriminability — {extractor}, layer={profile_layer}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inter-speaker profile discriminability analysis.")
    parser.add_argument("--l2arctic-zip", required=True, help="Path to l2arctic_release_v5.0.zip")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and plot outputs")
    parser.add_argument("--speakers", nargs="+", default=None, help="Speaker IDs to include (default: all 24)")
    parser.add_argument("--extractor", default="wav2vec2", choices=["wav2vec2", "wavlm"], help="Profile extractor model")
    parser.add_argument("--profile-layer", type=int, default=-1, help="Encoder layer to extract profile from (default -1 = last)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model cache directory")
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus", help="WavLM model name (used with --extractor wavlm)")

    args = parser.parse_args()
    run_profile_discriminability(
        l2arctic_zip=args.l2arctic_zip,
        output_dir=args.output_dir,
        speakers=args.speakers,
        extractor=args.extractor,
        profile_layer=args.profile_layer,
        cache_dir=args.cache_dir,
        wavlm_model=args.wavlm_model,
    )
