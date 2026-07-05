"""
Compare a prompted full-call run against the full-call baseline.

Joins the two ``*_all.csv`` files on ``call_id`` (each has one row per call), computes
per-call and corpus-level (word-weighted) WER and entity-EER deltas, writes a comparison
CSV, and logs a summary. Negative deltas mean the prompt *helped*.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.compare_fullcall \\
        --baseline data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline/baseline_all.csv \\
        --prompted data/processed/lexical_stylistic_prompting/earnings21_fullcall_metadata_only/prompted_all.csv \\
        --output   data/processed/lexical_stylistic_prompting/earnings21_fullcall_metadata_only/comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def _corpus_wer(df: pd.DataFrame, wer_col: str) -> float:
    return float((df[wer_col] * df["reference_words"]).sum() / df["reference_words"].sum())


def _corpus_eer(df: pd.DataFrame, err_col: str, tok_col: str) -> float:
    total_tokens = df[tok_col].sum()
    return float(df[err_col].sum() / total_tokens) if total_tokens > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="baseline *_all.csv")
    parser.add_argument("--prompted", required=True, help="prompted *_all.csv")
    parser.add_argument("--output", required=True, help="comparison CSV output path")
    args = parser.parse_args()

    base = pd.read_csv(args.baseline).add_prefix("base_").rename(columns={"base_call_id": "call_id"})
    prom = pd.read_csv(args.prompted).add_prefix("prom_").rename(columns={"prom_call_id": "call_id"})

    merged = base.merge(prom, on="call_id", how="inner")
    if merged.empty:
        logger.error("No overlapping call_id between baseline and prompted — nothing to compare")
        return
    logger.info(f"Joined {len(merged)} calls (baseline={len(base)}, prompted={len(prom)})")

    merged["wer_delta"] = merged["prom_wer"] - merged["base_wer"]
    merged["entity_eer_delta"] = merged["prom_entity_eer"] - merged["base_entity_eer"]

    out_cols = [
        "call_id",
        "base_wer", "prom_wer", "wer_delta",
        "base_entity_eer", "prom_entity_eer", "entity_eer_delta",
        "base_n_entity_tokens", "prom_n_entity_tokens",
        "base_reference_words",
    ]
    result = merged[out_cols].sort_values("entity_eer_delta")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(f"Saved comparison → {out_path}")

    # Corpus-level summary (word/token weighted).
    base_wer = _corpus_wer(merged.rename(columns={"base_wer": "wer", "base_reference_words": "reference_words"}), "wer")
    prom_wer = _corpus_wer(merged.rename(columns={"prom_wer": "wer", "prom_reference_words": "reference_words"}), "wer")
    base_eer = _corpus_eer(merged, "base_n_entity_errors", "base_n_entity_tokens")
    prom_eer = _corpus_eer(merged, "prom_n_entity_errors", "prom_n_entity_tokens")

    improved = int((merged["entity_eer_delta"] < 0).sum())
    logger.info("── Corpus summary (metadata_only vs baseline) ──")
    logger.info(f"WER        : {base_wer:.4f} → {prom_wer:.4f}  (Δ {prom_wer - base_wer:+.4f})")
    logger.info(f"Entity EER : {base_eer:.4f} → {prom_eer:.4f}  (Δ {prom_eer - base_eer:+.4f})")
    logger.info(f"Calls with lower entity-EER: {improved}/{len(merged)}")


if __name__ == "__main__":
    main()
