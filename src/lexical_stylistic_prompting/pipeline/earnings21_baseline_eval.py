from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from jiwer import process_words
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import _normalize, compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    Earnings21Segment,
    load_audio_segment,
    load_earnings21,
)

DEFAULT_MODEL = "openai/whisper-medium"


class Earnings21BaselineRow(BaseModel):
    call_id: str
    segment_id: str
    speaker: int
    reference: str
    hypothesis: str
    wer: float
    n_entity_tokens: int
    n_entity_errors: int
    entity_wer: float


def _compute_entity_wer(
    reference: str,
    hypothesis: str,
    entity_mask: list[bool],
) -> tuple[int, int]:
    """Return (n_entity_errors, n_entity_tokens) using word-level alignment."""
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


def run_segment_baseline(
    segment: Earnings21Segment,
    asr_pipe,
    audio_cache: dict,
) -> Earnings21BaselineRow | None:
    audio_key = (segment.call_id, segment.start_ts, segment.end_ts)
    if audio_key not in audio_cache:
        return None

    audio = audio_cache[audio_key]
    result = asr_pipe(audio)
    hypothesis = result["text"].strip()
    wer = compute_wer([segment.text], [hypothesis])
    n_entity_errors, n_entity_tokens = _compute_entity_wer(
        segment.text, hypothesis, segment.entity_mask
    )
    entity_wer = n_entity_errors / n_entity_tokens if n_entity_tokens > 0 else 0.0

    return Earnings21BaselineRow(
        call_id=segment.call_id,
        segment_id=segment.segment_id,
        speaker=segment.speaker,
        reference=segment.text,
        hypothesis=hypothesis,
        wer=wer,
        n_entity_tokens=n_entity_tokens,
        n_entity_errors=n_entity_errors,
        entity_wer=round(entity_wer, 4),
    )


def _save_csv(rows: list[Earnings21BaselineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Earnings21BaselineRow.model_fields))
        writer.writeheader()
        for row in rows:
            d = row.model_dump()
            d["wer"] = round(d["wer"], 4)
            writer.writerow(d)
    logger.info(f"Saved {len(rows)} rows → {path}")


def main(args: argparse.Namespace) -> None:
    if args.cache_dir:
        import os
        os.environ.setdefault("HF_HOME", args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")
    logger.info(f"Loading {args.model} ...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        chunk_length_s=30,
    )

    data_dir = Path(args.data_dir)
    logger.info(f"Loading Earnings21 from {data_dir} ...")
    calls = load_earnings21(data_dir, min_tokens=args.min_tokens)

    if args.call_id:
        calls = [c for c in calls if c.call_id == args.call_id]
        if not calls:
            logger.error(f"Call {args.call_id!r} not found")
            return

    output_dir = Path(args.output_dir)
    all_rows: list[Earnings21BaselineRow] = []

    for call in calls:
        logger.info(f"Processing call {call.call_id} ({len(call.segments)} segments) ...")

        # Pre-load all audio slices for this call at once
        audio_cache: dict = {}
        for seg in tqdm(call.segments, desc=f"Loading audio {call.call_id}", unit="seg"):
            key = (seg.call_id, seg.start_ts, seg.end_ts)
            try:
                audio_cache[key] = load_audio_segment(
                    call.audio_path, seg.start_ts, seg.end_ts
                )
            except Exception as e:
                logger.warning(f"Failed to load audio for {seg.segment_id}: {e}")

        rows: list[Earnings21BaselineRow] = []
        for seg in tqdm(call.segments, desc=f"Transcribing {call.call_id}", unit="seg"):
            row = run_segment_baseline(seg, asr_pipe, audio_cache)
            if row is not None:
                rows.append(row)

        all_rows.extend(rows)

        if rows:
            mean_wer = sum(r.wer for r in rows) / len(rows)
            entity_rows = [r for r in rows if r.n_entity_tokens > 0]
            mean_entity_wer = (
                sum(r.entity_wer for r in entity_rows) / len(entity_rows)
                if entity_rows else 0.0
            )
            logger.info(
                f"{call.call_id}: WER={mean_wer:.4f}, "
                f"entity WER={mean_entity_wer:.4f} ({len(entity_rows)} segs with entities)"
            )

        _save_csv(rows, output_dir / f"earnings21_baseline_{call.call_id}.csv")

    if all_rows:
        overall_wer = sum(r.wer for r in all_rows) / len(all_rows)
        entity_rows = [r for r in all_rows if r.n_entity_tokens > 0]
        overall_entity_wer = (
            sum(r.entity_wer for r in entity_rows) / len(entity_rows)
            if entity_rows else 0.0
        )
        logger.info(
            f"Overall WER={overall_wer:.4f}, "
            f"entity WER={overall_entity_wer:.4f} "
            f"({len(all_rows)} segments, {len(calls)} calls)"
        )


parser = argparse.ArgumentParser(description="Earnings21 Whisper baseline evaluation")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--data-dir", required=True, help="Path to earnings21 data directory")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--cache-dir", default=None)
parser.add_argument("--call-id", default=None, help="Process a single call only")
parser.add_argument("--min-tokens", type=int, default=5)

if __name__ == "__main__":
    main(parser.parse_args())
