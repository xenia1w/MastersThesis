"""
Smoke test 1/3 — KISSKI connection + profile building.

Builds a metadata_only profile for Monro Inc (call 4320211) and saves it
to data/processed/lexical_stylistic_prompting/profiles/ for use in step 3.

Run: uv run scripts/smoke_test_prompted_1_kisski.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    build_profile,
    save_profile,
)

CALL_ID = "4320211"
COMPANY = "Monro Inc"
SECTOR = "Consumer Goods"
QUARTER = "3"
N_PROFILE = 20

logger.info(f"Building metadata_only profile for {COMPANY} ({CALL_ID}) via KISSKI ...")
profile = build_profile(
    speaker_id=CALL_ID,
    strategy=ProfileStrategy.METADATA_ONLY,
    n_profile=N_PROFILE,
    company_name=COMPANY,
    sector=SECTOR,
    financial_quarter=QUARTER,
)

logger.info(f"Generated prompt ({len(profile.prompt)} chars):")
logger.info(profile.prompt)

path = save_profile(profile)
logger.info(f"Profile saved to: {path}")
logger.info("Smoke test 1 PASSED — run smoke_test_prompted_2_whisper_baseline.py next")
