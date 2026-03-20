from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger

from src.asr_adaptation.pipeline.lora_train import run_lora_train

# N-values and seeds used across the full sweep
N_VALUES = [1, 5, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2]


def run_data_size_single(
    speaker_id: str,
    n_train: int,
    seed: int,
    l2arctic_zip: str,
    output_dir: str | Path,
    cache_dir: str | None = None,
) -> Path:
    """
    Train and evaluate one (speaker, n_train, seed) combination.

    Reuses run_lora_train() and saves a single-row CSV so that parallel
    SLURM jobs never write to the same file at the same time.

    Output file: {output_dir}/{speaker_id}_{n_train:04d}_seed{seed}.csv
    Columns: speaker_id, n_train, seed, wer_baseline, wer_adapted, wer_delta

    Args:
        speaker_id: L2-ARCTIC speaker ID, e.g. "ABA".
        n_train: Number of utterances to use for training.
        seed: Random seed controlling which n_train utterances are selected.
        l2arctic_zip: Path to l2arctic_release_v5.0.zip.
        output_dir: Directory for individual result files.
        cache_dir: HuggingFace model cache directory.

    Returns:
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data size sweep | speaker={speaker_id} n_train={n_train} seed={seed}")

    rows = run_lora_train(
        speaker_id=speaker_id,
        l2arctic_zip=l2arctic_zip,
        output_dir=str(output_dir / "_lora_tmp"),  # throwaway adapter location
        cache_dir=cache_dir,
        n_train=n_train,
        seed=seed,
    )

    avg_baseline = sum(r.wer_baseline for r in rows) / len(rows)
    avg_adapted = sum(r.wer_adapted for r in rows) / len(rows)

    out_path = output_dir / f"{speaker_id}_{n_train:04d}_seed{seed}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["speaker_id", "n_train", "seed",
                           "wer_baseline", "wer_adapted", "wer_delta"]
        )
        writer.writeheader()
        writer.writerow(
            dict(
                speaker_id=speaker_id,
                n_train=n_train,
                seed=seed,
                wer_baseline=round(avg_baseline, 4),
                wer_adapted=round(avg_adapted, 4),
                wer_delta=round(avg_adapted - avg_baseline, 4),
            )
        )

    logger.info(f"Saved → {out_path}")
    return out_path


def merge_results(output_dir: str | Path) -> Path:
    """
    Merge all individual result CSVs in output_dir into per-speaker summary files.

    Reads every *_seed*.csv file and groups rows by speaker_id.
    Output: {output_dir}/{speaker_id}_wer_vs_n.csv — one row per (n_train, seed).

    Args:
        output_dir: Directory containing the individual result CSVs.

    Returns:
        Path to the output directory (all merged CSVs written there).
    """
    output_dir = Path(output_dir)
    individual_files = sorted(output_dir.glob("*_seed*.csv"))

    if not individual_files:
        raise FileNotFoundError(f"No result CSVs found in {output_dir}")

    # Group rows by speaker
    by_speaker: dict[str, list[dict[str, str]]] = {}
    for path in individual_files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                speaker = row["speaker_id"]
                by_speaker.setdefault(speaker, []).append(row)

    fieldnames = ["speaker_id", "n_train", "seed",
                  "wer_baseline", "wer_adapted", "wer_delta"]

    for speaker_id, rows in by_speaker.items():
        rows_sorted = sorted(rows, key=lambda r: (int(r["n_train"]), int(r["seed"])))
        merged_path = output_dir / f"{speaker_id}_wer_vs_n.csv"
        with open(merged_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
        logger.info(f"Merged {len(rows_sorted)} rows → {merged_path}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data size sweep for RQ1.3 — one (speaker, n_train, seed) per invocation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run: one sweep job ---
    run_p = subparsers.add_parser("run", help="Run one (speaker, n_train, seed) combination")
    run_p.add_argument("--speaker",      required=True,              help="L2-ARCTIC speaker ID, e.g. ABA")
    run_p.add_argument("--n-train",      required=True, type=int,    help="Number of training utterances")
    run_p.add_argument("--seed",         required=True, type=int,    help="Random seed")
    run_p.add_argument("--l2arctic-zip", required=True,              help="Path to l2arctic_release_v5.0.zip")
    run_p.add_argument("--output-dir",   required=True,              help="Directory for individual result CSVs")
    run_p.add_argument("--cache-dir",    default=None,               help="HuggingFace model cache directory")

    # --- merge: combine individual CSVs ---
    merge_p = subparsers.add_parser("merge", help="Merge individual CSVs into per-speaker summary")
    merge_p.add_argument("--output-dir", required=True, help="Directory containing individual result CSVs")

    args = parser.parse_args()

    if args.command == "run":
        run_data_size_single(
            speaker_id=args.speaker,
            n_train=args.n_train,
            seed=args.seed,
            l2arctic_zip=args.l2arctic_zip,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
    else:
        merge_results(args.output_dir)
