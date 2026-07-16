"""
Post-hoc LLM correction of the v2 baseline ASR hypotheses (RQ2 extension).

Prompting Whisper via ``initial_prompt`` hurt WER (see the v2 metadata_only result). This
module tries the opposite lever: leave decoding alone and instead have an LLM fix recognition
errors in the *raw* baseline transcript of the [5:00, 15:00] evaluation window, then score the
corrected text against the same hand-annotated golden references.

Runs locally (KISSKI LLM, like profile building — no cluster). Reads ``baseline_all.csv`` and
writes ``<out-dir>/prompted_all.csv`` (plus per-call ``posthoc_<id>.csv``) in the exact shape
``earnings21_window_score.py`` expects, so scoring needs no changes::

    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_window_score \\
        --annotations manual_annotation.csv --approach posthoc_blind

Design (see arXiv 2409.09554 / 2307.04172):
- Conservative / verbatim correction — the reference is verbatim .nlp text, and we only have a
  single (1-best) hypothesis, the setup most prone to over-correction. The prompt forbids
  rewriting; an edit-ratio guard reverts to the raw hypothesis if the LLM changed too much.
- Sentence-boundary chunking (~200 words) keeps each LLM call short so verbatim discipline
  holds and output length can't drift/truncate over a long span.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_posthoc_correct \\
        --baseline data/processed/lexical_stylistic_prompting/v2/earnings21_window_baseline/baseline_all.csv \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.models.constants import DEFAULT_LLM_MODEL
from src.lexical_stylistic_prompting.models.prompts import (
    POSTHOC_CORRECT_SYSTEM,
    POSTHOC_CORRECT_USER,
)
from src.lexical_stylistic_prompting.models.speaker_profile import _get_client, _llm_call

DEFAULT_BASELINE = Path(
    "data/processed/lexical_stylistic_prompting/v2/earnings21_window_baseline/baseline_all.csv"
)
DEFAULT_OUT_DIR = Path(
    "data/processed/lexical_stylistic_prompting/v2/earnings21_window_posthoc_blind"
)
# Sentence-grouped chunk size sent to the LLM. Larger chunks = fewer requests (KISSKI caps at
# ~10/min, 200/hour, 400/day); a ~1,400-word window at 450 words is ~3 requests instead of ~7.
CHUNK_TARGET_WORDS = 450
DEFAULT_MAX_EDIT_RATIO = 0.30  # revert to raw if the correction changed more than this fraction


class PosthocRow(BaseModel):
    call_id: str
    model: str
    raw_words: int
    corrected_words: int
    edit_ratio: float       # word-level change of the LLM output vs the raw hypothesis
    guard_applied: bool     # True → over-correction guard tripped, hypothesis reverted to raw
    hypothesis: str         # final (possibly reverted) text — what score.py reads
    raw_hypothesis: str


# ── chunking ──────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split on whitespace following sentence-terminal punctuation, keeping the punctuation."""
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in pieces if p.strip()]


def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """Group whole sentences into chunks of ~target_words (never splitting a sentence)."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > target_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── correction ────────────────────────────────────────────────────────────────

def _correct_chunk(client: OpenAI, model: str, chunk: str) -> str:
    words = len(chunk.split())
    # allow headroom over the input length so a correct-length output never truncates
    max_tokens = max(256, int(words * 2) + 64)
    raw = _llm_call(
        client,
        model,
        POSTHOC_CORRECT_SYSTEM,
        POSTHOC_CORRECT_USER.format(chunk=chunk),
        max_tokens=max_tokens,
    )
    return raw.strip().strip("\"'").strip()


def correct_hypothesis(client: OpenAI, model: str, hypothesis: str,
                       chunk_words: int = CHUNK_TARGET_WORDS, label: str = "") -> str:
    """Correct a full window hypothesis chunk-by-chunk and reassemble it."""
    chunks = chunk_text(hypothesis, chunk_words)
    corrected: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"    {label} chunk {i}/{len(chunks)} ({len(chunk.split())} words) → KISSKI ...")
        corrected.append(_correct_chunk(client, model, chunk))
    return " ".join(part for part in corrected if part).strip()


def _edit_ratio(raw: str, corrected: str) -> float:
    """Word-level fraction of the raw hypothesis changed by the correction (normalized)."""
    if not raw.strip():
        return 0.0
    return compute_wer([raw], [corrected])


def process_call(client: OpenAI, model: str, call_id: str, raw: str,
                 max_edit_ratio: float, chunk_words: int = CHUNK_TARGET_WORDS) -> PosthocRow:
    corrected = correct_hypothesis(client, model, raw, chunk_words, label=call_id)
    ratio = _edit_ratio(raw, corrected)
    guard = ratio > max_edit_ratio or not corrected.strip()
    if guard:
        logger.warning(f"{call_id}: edit_ratio={ratio:.3f} > {max_edit_ratio} — reverting to raw")
    final = raw if guard else corrected
    return PosthocRow(
        call_id=call_id,
        model=model,
        raw_words=len(raw.split()),
        corrected_words=len(corrected.split()),
        edit_ratio=round(ratio, 4),
        guard_applied=guard,
        hypothesis=final,
        raw_hypothesis=raw,
    )


# ── io ────────────────────────────────────────────────────────────────────────

def _write_row(out_dir: Path, row: PosthocRow) -> None:
    out = out_dir / f"posthoc_{row.call_id}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PosthocRow.model_fields))
        writer.writeheader()
        writer.writerow(row.model_dump())


def _merge_all(out_dir: Path) -> Path:
    files = sorted(glob.glob(str(out_dir / "posthoc_[0-9]*.csv")))
    merged = out_dir / "prompted_all.csv"
    pd.concat([pd.read_csv(f) for f in files]).to_csv(merged, index=False)
    logger.info(f"Merged {len(files)} per-call files → {merged}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc LLM correction of v2 baseline hypotheses")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="baseline_all.csv with call_id + hypothesis columns")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--max-edit-ratio", type=float, default=DEFAULT_MAX_EDIT_RATIO,
                        help="revert to the raw hypothesis if the correction changed more than this")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_TARGET_WORDS,
                        help="sentence-grouped chunk size sent to the LLM (larger = fewer requests)")
    parser.add_argument("--call-id", default=None, help="Process a single call only")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip calls that already have a posthoc_<id>.csv")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.baseline)
    df["call_id"] = df["call_id"].astype(str)
    if args.call_id:
        df = df[df["call_id"] == args.call_id]
        if df.empty:
            logger.error(f"Call {args.call_id!r} not found in {args.baseline}")
            return

    client = _get_client()
    corrected, skipped, reverted = 0, 0, 0
    total = len(df)

    for idx, (_, r) in enumerate(df.iterrows(), 1):
        call_id = str(r["call_id"])
        out_path = args.out_dir / f"posthoc_{call_id}.csv"
        if args.skip_existing and out_path.exists():
            logger.info(f"[{idx}/{total}] {call_id}: skipping (cached)")
            skipped += 1
            continue
        raw = str(r["hypothesis"]) if pd.notna(r["hypothesis"]) else ""
        n_chunks = len(chunk_text(raw, args.chunk_words))
        logger.info(f"[{idx}/{total}] {call_id}: correcting {len(raw.split())} words in {n_chunks} chunks ...")
        row = process_call(client, args.model, call_id, raw, args.max_edit_ratio, args.chunk_words)
        _write_row(args.out_dir, row)
        corrected += 1
        reverted += int(row.guard_applied)
        logger.success(f"[{idx}/{total}] {call_id}: done — edit_ratio={row.edit_ratio} "
                       f"guard={row.guard_applied} ({row.raw_words}→{row.corrected_words} words)")

    logger.success(f"Finished — corrected {corrected} (reverted {reverted}), skipped {skipped} / {total}")
    _merge_all(args.out_dir)


if __name__ == "__main__":
    main()
