"""Compare a whole approach against baseline: per-call table + averaged deltas.

Merges the approach's per-call CSVs (if needed), joins to baseline on the common call set,
prints a per-call WER/EER table, and reports both micro (corpus) and macro (per-call-mean)
average deltas — once over all calls and once excluding pathological baseline calls
(WER > 1), which otherwise distort the macro average.

Usage:
    uv run -m src.lexical_stylistic_prompting.comparison.compare_approach \\
        --approach transcript_plus_knowledge
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.lexical_stylistic_prompting.comparison import loaders, metrics

PATHOLOGICAL_WER = 1.0  # a baseline WER above this means a broken reference, not a real call


def _log_table(table: pd.DataFrame) -> None:
    logger.info("Per-call (sorted by entity-EER delta):")
    formatted = table.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    for line in formatted.splitlines():
        logger.info("  " + line)


def _check_references(merged: pd.DataFrame) -> None:
    n = metrics.reference_mismatches(merged)
    if n:
        logger.warning(f"{n} calls have differing baseline/prompted references — NOT apples-to-apples")
    else:
        logger.info("Reference identity check: all references match ✅")


def _pathological(merged: pd.DataFrame) -> list[str]:
    return merged.loc[merged["base_wer"] > PATHOLOGICAL_WER, "call_id"].tolist()


def analyze_approach(approach: str, base_dir: Path, force_merge: bool) -> None:
    merged_path = loaders.merge_per_call(loaders.approach_dir(base_dir, approach), force=force_merge)
    prompted = loaders.load_run(merged_path)
    baseline = loaders.load_baseline(base_dir)
    merged = loaders.join_to_baseline(baseline, prompted)

    _check_references(merged)
    _log_table(metrics.per_call_table(merged))

    metrics.summarize(merged, f"{approach} — ALL calls")
    bad = _pathological(merged)
    if bad:
        logger.info(f"Pathological baseline calls (WER>{PATHOLOGICAL_WER}): {bad}")
        metrics.summarize(merged[~merged["call_id"].isin(bad)], f"{approach} — excluding pathological")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True, help="strategy name, e.g. transcript_only")
    parser.add_argument("--base-dir", type=Path, default=loaders.DEFAULT_BASE_DIR)
    parser.add_argument("--force-merge", action="store_true", help="re-merge even if prompted_all.csv is current")
    args = parser.parse_args()
    analyze_approach(args.approach, args.base_dir, args.force_merge)


if __name__ == "__main__":
    main()
