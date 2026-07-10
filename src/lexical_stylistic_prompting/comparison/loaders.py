"""Loading, merging and joining of full-call result CSVs."""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from loguru import logger

DEFAULT_BASE_DIR = Path("data/processed/lexical_stylistic_prompting")
BASELINE_CSV = "earnings21_fullcall_baseline/baseline_all.csv"


def approach_dir(base_dir: Path, approach: str) -> Path:
    return base_dir / f"earnings21_fullcall_{approach}"


def per_call_files(directory: Path) -> list[Path]:
    """Numeric per-call CSVs only (ignores any merged ``*_all.csv``)."""
    return [Path(p) for p in sorted(glob.glob(str(directory / "prompted_[0-9]*.csv")))]


def merge_per_call(directory: Path, force: bool = False) -> Path:
    """Concatenate per-call CSVs into ``prompted_all.csv``; return its path.

    Re-merges when ``force`` is set or when the merged file is missing/older than the
    newest per-call CSV, so a stale merge is never silently reused.
    """
    merged = directory / "prompted_all.csv"
    files = per_call_files(directory)
    if not files:
        raise FileNotFoundError(f"no per-call CSVs in {directory}")
    if not force and merged.exists():
        newest = max(f.stat().st_mtime for f in files)
        if merged.stat().st_mtime >= newest:
            logger.info(f"Using existing merge: {merged} ({len(files)} per-call files present)")
            return merged
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame.to_csv(merged, index=False)
    logger.info(f"Merged {len(files)} per-call files → {merged}")
    return merged


def load_run(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["call_id"] = df["call_id"].astype(str)
    return df


def load_baseline(base_dir: Path) -> pd.DataFrame:
    return load_run(base_dir / BASELINE_CSV)


def join_to_baseline(baseline: pd.DataFrame, prompted: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on ``call_id`` with ``base_``/``prom_`` prefixes (common calls only)."""
    b = baseline.add_prefix("base_").rename(columns={"base_call_id": "call_id"})
    p = prompted.add_prefix("prom_").rename(columns={"prom_call_id": "call_id"})
    merged = b.merge(p, on="call_id", how="inner")
    logger.info(f"Joined {len(merged)} common calls (baseline={len(baseline)}, prompted={len(prompted)})")
    return merged
