"""
Build LLM-generated ASR context profiles for all Earnings21 calls (metadata_only).

Reads call metadata from earnings21-file-metadata.csv, calls the KISSKI LLM for each
call, and saves a profile JSON to ``<profiles-dir>/<strategy>/<call_id>_<n_profile>.json``.

Run this locally (requires internet / KISSKI access). Push the resulting JSON files to
the cluster before running the prompted evaluation.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \\
        --data-dir data/raw/earnings21 \\
        --strategy metadata_only \\
        --n-profile 20 \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger

from src.lexical_stylistic_prompting.models.constants import DEFAULT_LLM_MODEL, PROFILES_DIR
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM profiles for all Earnings21 calls"
    )
    parser.add_argument("--data-dir", required=True, help="Path to earnings21 data directory")
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument("--strategy", default=ProfileStrategy.METADATA_ONLY.value,
                        choices=[s.value for s in ProfileStrategy],
                        help="Profile strategy to use")
    parser.add_argument("--n-profile", type=int, default=20,
                        help="Profile window size (segments skipped from eval, default 20)")
    parser.add_argument("--kisski-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--call-id", default=None, help="Process a single call only")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip calls that already have a cached profile")
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    strategy = ProfileStrategy(args.strategy)
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

        logger.info(f"{call_id}: building profile ({company_name}, {sector}, Q{financial_quarter}) ...")
        profile = build_profile(
            speaker_id=call_id,
            strategy=strategy,
            n_profile=args.n_profile,
            company_name=company_name,
            sector=sector,
            financial_quarter=financial_quarter,
            model=args.kisski_model,
            client=client,
        )
        save_profile(profile, profiles_dir)
        logger.info(f"{call_id}: prompt ({len(profile.prompt.split(','))} terms): {profile.prompt[:120]} ...")
        built += 1

    logger.info(f"Done — built {built}, skipped {skipped} / {len(rows)} calls")


if __name__ == "__main__":
    main()
