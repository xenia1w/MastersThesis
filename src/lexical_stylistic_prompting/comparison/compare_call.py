"""Compare one call's prompted transcript against baseline, with a word-level diff.

Writes a per-call report (WER/EER deltas, entity-tagged fixed/degraded tokens, insertions)
to a ``.txt`` under ``<base-dir>/comparison/`` and logs a short summary to the terminal.

Usage:
    uv run -m src.lexical_stylistic_prompting.comparison.compare_call \\
        --approach transcript_plus_knowledge --call-id 4360674
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger

from src.asr_adaptation.metrics.wer import _normalize
from src.lexical_stylistic_prompting.comparison import loaders, worddiff

DEFAULT_DATA_DIR = Path("data/raw/earnings21")
COMPARISON_SUBDIR = "comparison"


def _row(df: pd.DataFrame, call_id: str, source: str) -> pd.Series:
    hit = df[df["call_id"] == call_id]
    if hit.empty:
        raise SystemExit(f"call {call_id!r} not found in {source}")
    return hit.iloc[0]


def _headline_lines(call_id: str, approach: str, base: pd.Series, prom: pd.Series) -> list[str]:
    return [
        f"call {call_id}   approach: {approach}",
        f"WER : {base.wer:.4f} -> {prom.wer:.4f}  (delta {prom.wer - base.wer:+.4f})",
        f"EER : {base.entity_eer:.4f} -> {prom.entity_eer:.4f}  (delta {prom.entity_eer - base.entity_eer:+.4f})",
    ]


def _word_diff_lines(ref_words: list[str], base_hyp: str, prom_hyp: str,
                     ref_norm: str, mask: list[bool]) -> tuple[list[str], dict[str, int]]:
    base_err = worddiff.ref_error_map(ref_norm, base_hyp)
    prom_err = worddiff.ref_error_map(ref_norm, prom_hyp)
    fixed = list(base_err.keys() - prom_err.keys())
    degraded = list(prom_err.keys() - base_err.keys())
    counts = {
        "fixed": len(fixed),
        "fixed_entities": sum(worddiff.is_entity(i, mask) for i in fixed),
        "degraded": len(degraded),
        "degraded_entities": sum(worddiff.is_entity(i, mask) for i in degraded),
    }
    lines = [f"FIXED: {counts['fixed']} tokens ({counts['fixed_entities']} on entities)"]
    lines += worddiff.format_changes(fixed, ref_words, base_err, mask)
    lines += ["", f"DEGRADED: {counts['degraded']} tokens ({counts['degraded_entities']} on entities)"]
    lines += worddiff.format_changes(degraded, ref_words, prom_err, mask)
    return lines, counts


def _insertion_lines(ref_norm: str, base_hyp: str, prom_hyp: str) -> list[str]:
    introduced = Counter(worddiff.inserted_tokens(ref_norm, prom_hyp)) - Counter(
        worddiff.inserted_tokens(ref_norm, base_hyp))
    top = [w for w, _ in introduced.most_common(15)]
    return [f"INSERTIONS introduced by prompt: {sum(introduced.values())} (top: {top})"]


def _write_report(lines: list[str], out_dir: Path, approach: str, call_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{approach}_{call_id}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_call(call_id: str, approach: str, base_dir: Path, data_dir: Path,
                 n_profile: int, out_dir: Path) -> None:
    baseline = loaders.load_baseline(base_dir)
    prom_file = loaders.approach_dir(base_dir, approach) / f"prompted_{call_id}.csv"
    prompted = loaders.load_run(prom_file)
    base = _row(baseline, call_id, "baseline")
    prom = _row(prompted, call_id, str(prom_file))

    ref_norm = _normalize(str(base.reference))
    base_hyp, prom_hyp = _normalize(str(base.hypothesis)), _normalize(str(prom.hypothesis))
    mask = worddiff.entity_mask_for_call(call_id, data_dir, n_profile)

    headline = _headline_lines(call_id, approach, base, prom)
    diff_lines, counts = _word_diff_lines(ref_norm.split(), base_hyp, prom_hyp, ref_norm, mask)
    insertions = _insertion_lines(ref_norm, base_hyp, prom_hyp)
    path = _write_report(headline + [""] + diff_lines + [""] + insertions, out_dir, approach, call_id)

    for line in headline:
        logger.info(line)
    logger.info(f"  fixed {counts['fixed']} ({counts['fixed_entities']} entity) | "
                f"degraded {counts['degraded']} ({counts['degraded_entities']} entity)")
    logger.success(f"  report -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--approach", required=True, help="strategy name, e.g. transcript_only")
    parser.add_argument("--base-dir", type=Path, default=loaders.DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-profile", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="report directory (default <base-dir>/comparison)")
    args = parser.parse_args()
    out_dir = args.out_dir or (args.base_dir / COMPARISON_SUBDIR)
    analyze_call(args.call_id, args.approach, args.base_dir, args.data_dir, args.n_profile, out_dir)


if __name__ == "__main__":
    main()
