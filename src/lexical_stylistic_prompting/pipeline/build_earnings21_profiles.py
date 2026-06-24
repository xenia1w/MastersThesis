"""
Build LLM-generated ASR context profiles for all Earnings21 calls.

Reads call metadata from earnings21-file-metadata.csv, calls the KISSKI LLM
for each call, and saves a profile JSON to --profiles-dir.

Run this locally (requires internet / KISSKI access). Push the resulting
JSON files to the cluster before running the prompted evaluation.

Usage:
    # metadata_only (no baseline transcriptions needed)
    uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \\
        --data-dir data/raw/earnings21 \\
        --strategy metadata_only \\
        --n-profile 20 \\
        --skip-existing

    # transcript_only (requires completed baseline CSVs)
    uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \\
        --data-dir data/raw/earnings21 \\
        --strategy transcript_only \\
        --baseline-dir data/processed/lexical_stylistic_prompting/earnings21_baseline \\
        --n-profile 20 \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger

from src.lexical_stylistic_prompting.models.constants import DEFAULT_MODEL, PROFILES_DIR
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    _get_client,
    build_profile,
    save_profile,
)


def _load_metadata(data_dir: Path) -> list[dict]:
    meta_path = data_dir / "earnings21-file-metadata.csv"
    with open(meta_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_baseline_segments(baseline_dir: Path, call_id: str, n: int) -> list[str]:
    """Load the first n hypothesis strings from the baseline CSV for call_id."""
    csv_path = baseline_dir / f"earnings21_baseline_{call_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found for {call_id}: {csv_path}\n"
            "Run the baseline evaluation first."
        )
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["segment_id"])
    return [r["hypothesis"] for r in rows[:n] if r["hypothesis"].strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM profiles for all Earnings21 calls"
    )
    parser.add_argument("--data-dir", required=True, help="Path to earnings21 data directory")
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument(
        "--strategy", required=True,
        choices=[s.value for s in ProfileStrategy],
        help="Profile strategy to use",
    )
    parser.add_argument(
        "--baseline-dir", default=None,
        help="Directory with baseline CSVs (required for transcript_only strategy)",
    )
    parser.add_argument("--n-profile", type=int, default=20)
    parser.add_argument("--kisski-model", default=DEFAULT_MODEL)
    parser.add_argument("--call-id", default=None, help="Process a single call only")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    strategy = ProfileStrategy(args.strategy)

    if strategy == ProfileStrategy.TRANSCRIPT_ONLY and not args.baseline_dir:
        parser.error("--baseline-dir is required for the transcript_only strategy")

    profiles_dir = Path(args.profiles_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    rows = _load_metadata(Path(args.data_dir))

    if args.call_id:
        rows = [r for r in rows if r["file_id"].strip() == args.call_id]
        if not rows:
            logger.error(f"Call {args.call_id!r} not found in metadata CSV")
            return

    client = _get_client()
    built, skipped = 0, 0

    for row in rows:
        call_id = row["file_id"].strip()
        out_path = profiles_dir / strategy.value / f"{call_id}_{args.n_profile}.json"

        if args.skip_existing and out_path.exists():
            logger.info(f"{call_id}: skipping (cached)")
            skipped += 1
            continue

        company_name = row["company_name"].strip()
        sector = row["sector"].strip()
        financial_quarter = row["financial_quarter"].strip()

        segments: list[str] | None = None
        if strategy in (ProfileStrategy.TRANSCRIPT_ONLY, ProfileStrategy.TRANSCRIPT_PLUS_KNOWLEDGE):
            assert baseline_dir is not None
            try:
                segments = _load_baseline_segments(baseline_dir, call_id, args.n_profile)
            except FileNotFoundError as e:
                logger.error(str(e))
                continue
            logger.info(
                f"{call_id}: building profile ({strategy.value}, "
                f"{len(segments)} transcript segments) ..."
            )
        else:
            logger.info(
                f"{call_id}: building profile ({company_name}, {sector}, Q{financial_quarter}) ..."
            )

        profile = build_profile(
            speaker_id=call_id,
            strategy=strategy,
            n_profile=args.n_profile,
            company_name=company_name,
            sector=sector,
            financial_quarter=financial_quarter,
            segments=segments,
            model=args.kisski_model,
            client=client,
        )
        save_profile(profile, profiles_dir)
        logger.info(f"{call_id}: prompt preview: {profile.prompt[:100]} ...")
        built += 1

    logger.info(f"Done — built {built}, skipped {skipped} / {len(rows)} calls")


if __name__ == "__main__":
    main()
