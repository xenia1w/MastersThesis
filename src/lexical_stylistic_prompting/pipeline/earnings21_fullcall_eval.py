"""
Earnings21 full-call evaluation with openai-whisper (baseline OR metadata_only).

One unified evaluator so the baseline and the prompted run differ *only* by the injected
``initial_prompt``. For each call:

  1. Split at the profile boundary: ``segments[:n_profile]`` are the (unused-here) profile
     window; ``segments[n_profile:]`` form the evaluation reference.
  2. Load one continuous audio slice from ``segments[n_profile-1].end_ts`` to end of file —
     this sidesteps per-segment RTTM/NLP alignment by treating the eval portion as one slice.
  3. Transcribe with Whisper Medium.
       - strategy=baseline      → no prompt.
       - strategy=metadata_only → initial_prompt = the LLM keyword list, carried to every
         30 s window via carry_initial_prompt=True so the whole call is biased.
  4. Compute WER and entity EER against the concatenated NLP reference.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval \\
        --data-dir     data/raw/earnings21 \\
        --output       data/processed/lexical_stylistic_prompting/earnings21_fullcall_metadata_only/prompted_4392809.csv \\
        --strategy     metadata_only \\
        --n-profile    20 \\
        --call-id      4392809

    # Local smoke test (cap eval to a >30 s slice):
        --call-id 4392809 --max-eval-segments 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper
from jiwer import process_words
from loguru import logger
from pydantic import BaseModel

from src.asr_adaptation.metrics.wer import _normalize, compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    Earnings21Call,
    load_earnings21,
)
from src.lexical_stylistic_prompting.models.constants import PROFILES_DIR
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    load_profile,
)

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "medium"


class FullCallRow(BaseModel):
    call_id: str
    strategy: str
    prompt: str
    n_eval_segments: int
    reference_words: int
    hypothesis_words: int
    wer: float
    n_entity_tokens: int
    n_entity_errors: int
    entity_eer: float
    reference: str
    hypothesis: str


def _load_eval_audio(call: Earnings21Call, split_ts: float, end_ts: float | None) -> np.ndarray:
    duration = (end_ts - split_ts) if end_ts is not None else None
    audio, _ = librosa.load(
        str(call.audio_path),
        sr=SAMPLE_RATE,
        offset=split_ts,
        duration=duration,
        mono=True,
    )
    return audio.astype(np.float32)


def _compute_entity_errors(
    reference: str,
    hypothesis: str,
    entity_mask: list[bool],
) -> tuple[int, int]:
    """Count substitution/deletion errors that land on entity-marked reference tokens."""
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


def _transcribe(
    model: whisper.Whisper,
    audio: np.ndarray,
    initial_prompt: str | None,
    fp16: bool,
) -> str:
    kwargs: dict = {
        "language": "en",
        "condition_on_previous_text": True,
        "fp16": fp16,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
        kwargs["carry_initial_prompt"] = True  # bias every 30 s window, not just the first
    result = model.transcribe(audio, **kwargs)
    return str(result["text"]).strip()


def evaluate_call(
    call: Earnings21Call,
    n_profile: int,
    strategy_label: str,
    model: whisper.Whisper,
    fp16: bool,
    prompt_text: str | None,
    max_eval_segments: int | None = None,
    max_eval_seconds: float | None = None,
) -> FullCallRow | None:
    if len(call.segments) <= n_profile:
        logger.warning(
            f"{call.call_id}: only {len(call.segments)} segments, need > {n_profile} — skipping"
        )
        return None

    split_ts = call.segments[n_profile - 1].end_ts
    eval_segments = call.segments[n_profile:]
    if max_eval_segments is not None:
        eval_segments = eval_segments[:max_eval_segments]
    if max_eval_seconds is not None:
        # Keep whole segments that fit within the time budget (always keep at least one).
        capped = [s for s in eval_segments if s.end_ts - split_ts <= max_eval_seconds]
        eval_segments = capped or eval_segments[:1]

    reference = " ".join(seg.text for seg in eval_segments)
    entity_mask = [flag for seg in eval_segments for flag in seg.entity_mask]

    bounded = max_eval_segments is not None or max_eval_seconds is not None
    end_ts = eval_segments[-1].end_ts if bounded else None
    slice_secs = (end_ts - split_ts) if end_ts is not None else (call.segments[-1].end_ts - split_ts)
    logger.info(
        f"{call.call_id}: split at {split_ts:.1f}s, {len(eval_segments)} eval segments, "
        f"~{slice_secs:.0f}s audio, {len(reference.split())} reference words, strategy={strategy_label}"
    )

    audio = _load_eval_audio(call, split_ts, end_ts)
    hypothesis = _transcribe(model, audio, prompt_text, fp16)

    # Defensive leak check — openai-whisper strips the prompt, but warn if it ever appears.
    if prompt_text:
        first_term = _normalize(prompt_text.split(",")[0])
        if first_term and _normalize(hypothesis).startswith(first_term):
            logger.warning(f"{call.call_id}: hypothesis may start with the prompt — inspect output")

    wer = compute_wer([reference], [hypothesis])
    n_entity_errors, n_entity_tokens = _compute_entity_errors(reference, hypothesis, entity_mask)
    entity_eer = n_entity_errors / n_entity_tokens if n_entity_tokens > 0 else 0.0

    logger.info(
        f"{call.call_id}: WER={wer:.4f}, entity EER={entity_eer:.4f} "
        f"({n_entity_errors}/{n_entity_tokens} entity tokens)"
    )

    return FullCallRow(
        call_id=call.call_id,
        strategy=strategy_label,
        prompt=prompt_text or "",
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


def _save_csv(rows: list[FullCallRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FullCallRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    logger.info(f"Saved {len(rows)} rows → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", default=ProfileStrategy.METADATA_ONLY.value,
                        choices=["baseline", *[s.value for s in ProfileStrategy]])
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument("--n-profile", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name, e.g. 'medium'")
    parser.add_argument("--download-root", default=None, help="openai-whisper weights cache dir")
    parser.add_argument("--call-id", default=None, help="Evaluate a single call only")
    parser.add_argument("--max-eval-segments", type=int, default=None,
                        help="Cap evaluation to first N segments after the profile (smoke test)")
    parser.add_argument("--max-eval-seconds", type=float, default=None,
                        help="Cap evaluation to ~N seconds of audio after the profile (smoke test)")
    args = parser.parse_args()

    is_baseline = args.strategy == "baseline"
    strategy = None if is_baseline else ProfileStrategy(args.strategy)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    fp16 = device == "cuda"
    logger.info(f"Device: {device} | Model: {args.model} | Strategy: {args.strategy}")

    model = whisper.load_model(args.model, device=device, download_root=args.download_root)

    profiles_dir = Path(args.profiles_dir)
    calls = load_earnings21(Path(args.data_dir), min_tokens=5)
    if args.call_id:
        calls = [c for c in calls if c.call_id == args.call_id]
        if not calls:
            logger.error(f"Call {args.call_id!r} not found")
            return

    rows: list[FullCallRow] = []
    for call in calls:
        prompt_text: str | None = None
        if strategy is not None:
            try:
                profile = load_profile(call.call_id, args.n_profile, strategy, profiles_dir)
            except FileNotFoundError:
                logger.warning(f"{call.call_id}: no {args.strategy} profile found, skipping")
                continue
            prompt_text = profile.prompt

        row = evaluate_call(
            call=call,
            n_profile=args.n_profile,
            strategy_label=args.strategy,
            model=model,
            fp16=fp16,
            prompt_text=prompt_text,
            max_eval_segments=args.max_eval_segments,
            max_eval_seconds=args.max_eval_seconds,
        )
        if row is not None:
            rows.append(row)

    _save_csv(rows, Path(args.output))

    if rows:
        overall_wer = sum(r.wer * r.reference_words for r in rows) / sum(r.reference_words for r in rows)
        total_entity_errors = sum(r.n_entity_errors for r in rows)
        total_entity_tokens = sum(r.n_entity_tokens for r in rows)
        overall_eer = total_entity_errors / total_entity_tokens if total_entity_tokens > 0 else 0.0
        logger.info(
            f"Overall WER={overall_wer:.4f}, entity EER={overall_eer:.4f} ({len(rows)} calls)"
        )


if __name__ == "__main__":
    main()
