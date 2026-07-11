"""
Build LLM-generated ASR context profiles for all Earnings21 calls.

Reads call metadata from earnings21-file-metadata.csv, calls the KISSKI LLM for each
call, and saves a profile JSON to ``<profiles-dir>/<strategy>/<call_id>_<n_profile>.json``.

For the transcript_only / transcript_plus_knowledge strategies the LLM also reads the noisy
Whisper transcript of the call's first n_profile turns; run earnings21_profile_window.py first
to produce those under --transcripts-dir.

Run this locally (requires internet / KISSKI access). Push the resulting JSON files to
the cluster before running the prompted evaluation.

Usage:
    # metadata_only (no transcript needed)
    uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \\
        --data-dir data/raw/earnings21 --strategy metadata_only --n-profile 20 --skip-existing

    # transcript_only / transcript_plus_knowledge (needs profile_transcripts/ first)
    uv run -m src.lexical_stylistic_prompting.pipeline.build_earnings21_profiles \\
        --data-dir data/raw/earnings21 --strategy transcript_only --n-profile 20 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from loguru import logger

from src.lexical_stylistic_prompting.models.constants import (
    DEFAULT_LLM_MODEL,
    PROFILE_TRANSCRIPTS_DIR,
    PROFILES_DIR,
)
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    PromptFormat,
    _get_client,
    build_profile,
    profile_subdir,
    save_profile,
)

_TRANSCRIPT_STRATEGIES = {
    ProfileStrategy.TRANSCRIPT_ONLY,
    ProfileStrategy.TRANSCRIPT_PLUS_KNOWLEDGE,
    ProfileStrategy.TRANSCRIPT_METADATA_KNOWLEDGE,
}


def _load_metadata(data_dir: Path) -> list[dict]:
    meta_path = data_dir / "earnings21-file-metadata.csv"
    with open(meta_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_transcript(transcripts_dir: Path, call_id: str, n_profile: int) -> tuple[str, int]:
    path = transcripts_dir / f"{call_id}_{n_profile}.json"
    data = json.loads(path.read_text())
    return data["transcript"], int(data.get("n_segments", n_profile))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM profiles for all Earnings21 calls"
    )
    parser.add_argument("--data-dir", required=True, help="Path to earnings21 data directory")
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument("--strategy", default=ProfileStrategy.METADATA_ONLY.value,
                        choices=[s.value for s in ProfileStrategy],
                        help="Profile strategy to use")
    parser.add_argument("--prompt-format", default=PromptFormat.LIST.value,
                        choices=[f.value for f in PromptFormat],
                        help="initial_prompt surface form: keyword list or natural prose")
    parser.add_argument("--n-profile", type=int, default=20,
                        help="Profile window size (segments skipped from eval, default 20)")
    parser.add_argument("--transcripts-dir", default=str(PROFILE_TRANSCRIPTS_DIR),
                        help="Dir with profile-window transcripts (transcript_* strategies)")
    parser.add_argument("--kisski-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--call-id", default=None, help="Process a single call only")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip calls that already have a cached profile")
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    transcripts_dir = Path(args.transcripts_dir)
    strategy = ProfileStrategy(args.strategy)
    prompt_format = PromptFormat(args.prompt_format)
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
        out_path = profiles_dir / profile_subdir(strategy, prompt_format) / f"{call_id}_{args.n_profile}.json"

        if args.skip_existing and out_path.exists():
            logger.info(f"{call_id}: skipping (cached)")
            skipped += 1
            continue

        company_name = row["company_name"].strip()
        sector = row["sector"].strip()
        financial_quarter = row["financial_quarter"].strip()

        transcript, n_segments = "", 0
        if strategy in _TRANSCRIPT_STRATEGIES:
            try:
                transcript, n_segments = _load_transcript(transcripts_dir, call_id, args.n_profile)
            except FileNotFoundError:
                logger.error(f"{call_id}: no profile-window transcript in {transcripts_dir} — "
                             "run earnings21_profile_window.py first. Skipping.")
                continue
            logger.info(f"{call_id}: building profile ({strategy.value}, {n_segments} turns, "
                        f"{len(transcript.split())} transcript words) ...")
        else:
            logger.info(f"{call_id}: building profile ({company_name}, {sector}, Q{financial_quarter}) ...")

        profile = build_profile(
            speaker_id=call_id,
            strategy=strategy,
            n_profile=args.n_profile,
            company_name=company_name,
            sector=sector,
            financial_quarter=financial_quarter,
            transcript=transcript,
            n_segments=n_segments,
            prompt_format=prompt_format,
            model=args.kisski_model,
            client=client,
        )
        save_profile(profile, profiles_dir)
        logger.info(f"{call_id}: {prompt_format.value} prompt: {profile.prompt[:120]} ...")
        built += 1

    logger.info(f"Done — built {built}, skipped {skipped} / {len(rows)} calls")


if __name__ == "__main__":
    main()
