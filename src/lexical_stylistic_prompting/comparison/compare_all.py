"""Cross-approach comparison: every approach vs baseline on one common call set.

Merges each approach, restricts all of them to the calls common to baseline and every
approach (strict apples-to-apples), and prints one summary row per approach — WER and
entity-EER, baseline vs prompted, micro and macro deltas, plus paired Wilcoxon p-values.
Reported over all calls and again excluding pathological baseline calls (WER > 1).

Usage:
    uv run -m src.lexical_stylistic_prompting.comparison.compare_all
    uv run -m src.lexical_stylistic_prompting.comparison.compare_all --approach metadata_only --approach transcript_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.lexical_stylistic_prompting.comparison import loaders, metrics
from src.lexical_stylistic_prompting.comparison.compare_approach import PATHOLOGICAL_WER

DEFAULT_APPROACHES = [
    "metadata_only",
    "transcript_only",
    "transcript_plus_knowledge",
    "transcript_metadata_knowledge",
]
SUMMARY_CSV = "cross_approach_summary.csv"


def load_approaches(base_dir: Path, approaches: list[str]) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}
    for approach in approaches:
        merged_path = loaders.merge_per_call(loaders.approach_dir(base_dir, approach))
        runs[approach] = loaders.load_run(merged_path)
    return runs


def common_call_ids(baseline: pd.DataFrame, runs: dict[str, pd.DataFrame]) -> set[str]:
    common = set(baseline["call_id"])
    for df in runs.values():
        common &= set(df["call_id"])
    return common


def _restrict(df: pd.DataFrame, call_ids: set[str]) -> pd.DataFrame:
    return df[df["call_id"].isin(call_ids)]


def build_table(baseline: pd.DataFrame, runs: dict[str, pd.DataFrame], call_ids: set[str]) -> pd.DataFrame:
    base = _restrict(baseline, call_ids)
    rows = [
        metrics.summary_row(loaders.join_to_baseline(base, _restrict(prompted, call_ids)), approach)
        for approach, prompted in runs.items()
    ]
    return pd.DataFrame(rows)


def _log_table(table: pd.DataFrame, label: str) -> None:
    logger.info(f"══ Cross-approach summary — {label} ══")
    for line in table.to_string(index=False).splitlines():
        logger.info("  " + line)


def analyze_all(base_dir: Path, approaches: list[str], out_dir: Path) -> None:
    baseline = loaders.load_baseline(base_dir)
    runs = load_approaches(base_dir, approaches)

    common = common_call_ids(baseline, runs)
    logger.info(f"Common call set across baseline + {len(runs)} approaches: {len(common)} calls")

    table_all = build_table(baseline, runs, common)
    _log_table(table_all, f"all {len(common)} calls")

    pathological = set(baseline.loc[baseline["wer"] > PATHOLOGICAL_WER, "call_id"])
    clean = common - pathological
    if pathological:
        logger.info(f"Excluding pathological baseline calls (WER>{PATHOLOGICAL_WER}): {sorted(pathological)}")
        _log_table(build_table(baseline, runs, clean), f"{len(clean)} calls (pathological excluded)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / SUMMARY_CSV
    table_all.to_csv(out_path, index=False)
    logger.success(f"Wrote cross-approach summary -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", action="append", default=None,
                        help="strategy name (repeatable); defaults to all four approaches")
    parser.add_argument("--base-dir", type=Path, default=loaders.DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="summary directory (default <base-dir>/comparison)")
    args = parser.parse_args()
    approaches = args.approach or DEFAULT_APPROACHES
    out_dir = args.out_dir or (args.base_dir / "comparison")
    analyze_all(args.base_dir, approaches, out_dir)


if __name__ == "__main__":
    main()
