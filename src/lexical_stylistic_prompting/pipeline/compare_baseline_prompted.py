"""
Compare prompted vs baseline evaluation results for Earnings21.

Joins on segment_id — baseline is filtered to the same evaluation-window
segments that appear in the prompted output, so numbers are directly comparable.

Outputs a per-call comparison CSV and prints an overall summary.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.compare_baseline_prompted \\
        --baseline  data/processed/lexical_stylistic_prompting/earnings21_baseline/earnings21_baseline_all.csv \\
        --prompted-dir data/processed/lexical_stylistic_prompting/earnings21_prompted_metadata_only \\
        --output    data/processed/lexical_stylistic_prompting/comparison/comparison_metadata_only_n20.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from loguru import logger


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict]) -> tuple[int, int]:
    """Return (total_entity_errors, total_entity_tokens) for a set of rows."""
    errors = sum(int(r["n_entity_errors"]) for r in rows)
    tokens = sum(int(r["n_entity_tokens"]) for r in rows)
    return errors, tokens


def compare(
    baseline_path: Path,
    prompted_dir: Path,
    output_path: Path,
) -> None:
    baseline_rows = _load_csv(baseline_path)
    baseline_by_segment: dict[str, dict] = {r["segment_id"]: r for r in baseline_rows}

    prompted_files = sorted(Path(prompted_dir).glob("earnings21_prompted_*.csv"))
    if not prompted_files:
        logger.error(f"No prompted CSV files found in {prompted_dir}")
        return

    comparison_rows = []
    all_baseline_eval: list[dict] = []
    all_prompted_eval: list[dict] = []

    for pfile in prompted_files:
        prompted_rows = _load_csv(pfile)
        if not prompted_rows:
            continue

        call_id = prompted_rows[0]["call_id"]
        segment_ids = {r["segment_id"] for r in prompted_rows}
        baseline_eval = [r for r in baseline_rows if r["segment_id"] in segment_ids]

        if not baseline_eval:
            logger.warning(f"{call_id}: no matching baseline segments found")
            continue

        b_errors, b_tokens = _aggregate(baseline_eval)
        p_errors, p_tokens = _aggregate(prompted_rows)

        b_eer = b_errors / b_tokens if b_tokens > 0 else 0.0
        p_eer = p_errors / p_tokens if p_tokens > 0 else 0.0
        delta_eer = p_eer - b_eer

        b_wer = sum(float(r["wer"]) for r in baseline_eval) / len(baseline_eval)
        p_wer = sum(float(r["wer"]) for r in prompted_rows) / len(prompted_rows)

        comparison_rows.append({
            "call_id": call_id,
            "n_eval_segments": len(prompted_rows),
            "baseline_entity_errors": b_errors,
            "baseline_entity_tokens": b_tokens,
            "baseline_eer": round(b_eer, 4),
            "prompted_entity_errors": p_errors,
            "prompted_entity_tokens": p_tokens,
            "prompted_eer": round(p_eer, 4),
            "delta_eer": round(delta_eer, 4),
            "baseline_wer": round(b_wer, 4),
            "prompted_wer": round(p_wer, 4),
            "delta_wer": round(p_wer - b_wer, 4),
        })

        all_baseline_eval.extend(baseline_eval)
        all_prompted_eval.extend(prompted_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(comparison_rows[0].keys()) if comparison_rows else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    logger.info(f"Saved comparison → {output_path}")

    # Overall summary
    b_errors, b_tokens = _aggregate(all_baseline_eval)
    p_errors, p_tokens = _aggregate(all_prompted_eval)
    b_eer_overall = b_errors / b_tokens if b_tokens > 0 else 0.0
    p_eer_overall = p_errors / p_tokens if p_tokens > 0 else 0.0
    b_wer_overall = sum(float(r["wer"]) for r in all_baseline_eval) / len(all_baseline_eval)
    p_wer_overall = sum(float(r["wer"]) for r in all_prompted_eval) / len(all_prompted_eval)

    print("\n=== Overall comparison ===")
    print(f"Calls evaluated : {len(comparison_rows)}")
    print(f"Segments        : {len(all_prompted_eval)}")
    print(f"Baseline  EER   : {b_eer_overall:.4f} ({b_errors}/{b_tokens})")
    print(f"Prompted  EER   : {p_eer_overall:.4f} ({p_errors}/{p_tokens})")
    print(f"Delta     EER   : {p_eer_overall - b_eer_overall:+.4f}")
    print(f"Baseline  WER   : {b_wer_overall:.4f}")
    print(f"Prompted  WER   : {p_wer_overall:.4f}")
    print(f"Delta     WER   : {p_wer_overall - b_wer_overall:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompted vs baseline Earnings21 results")
    parser.add_argument("--baseline", required=True,
                        help="Path to earnings21_baseline_all.csv")
    parser.add_argument("--prompted-dir", required=True,
                        help="Directory containing per-call prompted CSVs")
    parser.add_argument("--output", required=True,
                        help="Path for the per-call comparison CSV output")
    args = parser.parse_args()

    compare(Path(args.baseline), Path(args.prompted_dir), Path(args.output))


if __name__ == "__main__":
    main()
