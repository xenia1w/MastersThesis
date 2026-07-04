"""
Earnings21 full-call baseline evaluation (no prompt).

For each call:
  1. Splits at the profile boundary: segments[:n_profile] are the profile window,
     segments[n_profile:] form the evaluation reference.
  2. Loads audio from segments[n_profile-1].end_ts to end of file — this avoids
     the broken per-segment RTTM-NLP matching by treating the evaluation portion
     as one continuous audio slice.
  3. Runs Whisper on the full evaluation audio (no prompt).
  4. Computes WER and entity EER against the concatenated NLP reference text.

The first n_profile segments are almost always correctly aligned (misalignment
appears later in Q&A sections), so the split timestamp is reliable.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_baseline_eval \
        --data-dir  data/raw/earnings21 \
        --output    data/processed/lexical_stylistic_prompting/earnings21_fullcall_baseline/baseline_all.csv \
        --n-profile 20

    # Single call:
        --call-id 4392809
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from jiwer import process_words
from loguru import logger
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import _normalize, compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    Earnings21Call,
    load_earnings21,
)

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "openai/whisper-medium"


class FullCallBaselineRow(BaseModel):
    call_id: str
    n_eval_segments: int
    reference_words: int
    hypothesis_words: int
    wer: float
    n_entity_tokens: int
    n_entity_errors: int
    entity_eer: float
    reference: str
    hypothesis: str


def _load_eval_audio(call: Earnings21Call, split_ts: float, end_ts: float | None = None) -> dict:
    duration = (end_ts - split_ts) if end_ts is not None else None
    audio, _ = librosa.load(
        str(call.audio_path),
        sr=SAMPLE_RATE,
        offset=split_ts,
        duration=duration,
        mono=True,
    )
    return {"raw": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE}


def _compute_entity_errors(
    reference: str,
    hypothesis: str,
    entity_mask: list[bool],
) -> tuple[int, int]:
    ref_norm = _normalize(reference)
    hyp_norm = _normalize(hypothesis)
    ref_words = ref_norm.split()

    n_entity_tokens = sum(
        entity_mask[i] for i in range(min(len(entity_mask), len(ref_words)))
    )
    if n_entity_tokens == 0:
        return 0, 0

    try:
        out = process_words(ref_norm, hyp_norm)
    except Exception:
        return 0, n_entity_tokens

    n_entity_errors = 0
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                idx = chunk.ref_start_idx + i
                if idx < len(entity_mask) and entity_mask[idx]:
                    n_entity_errors += 1

    return n_entity_errors, n_entity_tokens


def evaluate_call(
    call: Earnings21Call,
    n_profile: int,
    asr_pipe,
    max_eval_segments: int | None = None,
) -> FullCallBaselineRow | None:
    if len(call.segments) <= n_profile:
        logger.warning(f"{call.call_id}: only {len(call.segments)} segments, need > {n_profile} — skipping")
        return None

    split_ts = call.segments[n_profile - 1].end_ts
    eval_segments = call.segments[n_profile:]
    if max_eval_segments is not None:
        eval_segments = eval_segments[:max_eval_segments]

    # Build reference and entity mask from NLP text (correct ordering, no RTTM needed)
    reference = " ".join(seg.text for seg in eval_segments)
    entity_mask = [flag for seg in eval_segments for flag in seg.entity_mask]

    logger.info(
        f"{call.call_id}: split at {split_ts:.1f}s, "
        f"{len(eval_segments)} eval segments, "
        f"{len(reference.split())} reference words"
    )

    end_ts = eval_segments[-1].end_ts if max_eval_segments is not None else None
    audio = _load_eval_audio(call, split_ts, end_ts)
    result = asr_pipe(audio)
    hypothesis = result["text"].strip() if isinstance(result, dict) else result[0]["text"].strip()

    wer = compute_wer([reference], [hypothesis])
    n_entity_errors, n_entity_tokens = _compute_entity_errors(reference, hypothesis, entity_mask)
    entity_eer = n_entity_errors / n_entity_tokens if n_entity_tokens > 0 else 0.0

    logger.info(
        f"{call.call_id}: WER={wer:.4f}, "
        f"entity EER={entity_eer:.4f} ({n_entity_tokens} entity tokens)"
    )

    return FullCallBaselineRow(
        call_id=call.call_id,
        n_eval_segments=len(eval_segments),
        reference_words=len(reference.split()),
        hypothesis_words=len(hypothesis.split()),
        wer=round(wer, 4),
        n_entity_tokens=n_entity_tokens,
        n_entity_errors=n_entity_errors,
        entity_eer=round(entity_eer, 4),
        reference=reference,
        hypothesis=hypothesis,
    )


def _save_csv(rows: list[FullCallBaselineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FullCallBaselineRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    logger.info(f"Saved {len(rows)} rows → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--n-profile",  type=int, default=20)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir",  default=None)
    parser.add_argument("--call-id",    default=None, help="Evaluate a single call only")
    parser.add_argument("--max-eval-segments", type=int, default=None,
                        help="Cap evaluation to first N segments after profile (local smoke test)")
    args = parser.parse_args()

    if args.cache_dir:
        import os
        os.environ.setdefault("HF_HOME", args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Device: {device} | Model: {args.model}")

    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )

    calls = load_earnings21(Path(args.data_dir), min_tokens=5)
    if args.call_id:
        calls = [c for c in calls if c.call_id == args.call_id]
        if not calls:
            logger.error(f"Call {args.call_id!r} not found")
            return

    rows = []
    for call in calls:
        row = evaluate_call(call, args.n_profile, asr_pipe, args.max_eval_segments)
        if row is not None:
            rows.append(row)

    _save_csv(rows, Path(args.output))

    if rows:
        overall_wer = sum(r.wer * r.reference_words for r in rows) / sum(r.reference_words for r in rows)
        total_entity_errors = sum(r.n_entity_errors for r in rows)
        total_entity_tokens = sum(r.n_entity_tokens for r in rows)
        overall_eer = total_entity_errors / total_entity_tokens if total_entity_tokens > 0 else 0.0
        logger.info(f"Overall WER={overall_wer:.4f}, entity EER={overall_eer:.4f} ({len(rows)} calls)")


if __name__ == "__main__":
    main()
