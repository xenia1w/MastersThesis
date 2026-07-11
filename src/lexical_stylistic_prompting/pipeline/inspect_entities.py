"""
Diagnostic: show which entity tokens are counted as errors for a single call/slice.

Reproduces the entity-EER alignment from earnings21_fullcall_eval, but instead of a single
number it prints, for every entity-marked reference token, its entity_type and whether it was
a substitution/deletion error under the baseline vs the prompted hypothesis. Also flags any
mask/normalization misalignment (a possible source of spurious "errors").

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.inspect_entities \\
        --data-dir data/raw/earnings21 --call-id 4320211 --model medium --max-eval-seconds 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import whisper
from jiwer import process_words
from loguru import logger

from src.asr_adaptation.metrics.wer import _normalize
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    VOCABULARY_ENTITY_TYPES,
    load_earnings21,
)
from src.lexical_stylistic_prompting.models.constants import PROFILES_DIR
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    load_profile,
)
from src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval import (
    _load_eval_audio,
    _transcribe,
)


def _error_ref_indices(reference: str, hypothesis: str) -> set[int]:
    """Reference word indices that are a substitution or deletion (i.e. wrong/missing)."""
    out = process_words(_normalize(reference), _normalize(hypothesis))
    bad: set[int] = set()
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                bad.add(chunk.ref_start_idx + i)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument("--n-profile", type=int, default=20)
    parser.add_argument("--max-eval-seconds", type=float, default=120.0)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--download-root", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    fp16 = device == "cuda"

    calls = [c for c in load_earnings21(Path(args.data_dir), min_tokens=5) if c.call_id == args.call_id]
    if not calls:
        logger.error(f"Call {args.call_id!r} not found")
        return
    call = calls[0]

    # Same eval-window selection as evaluate_call (seconds cap).
    split_ts = call.segments[args.n_profile - 1].end_ts
    eval_segments = call.segments[args.n_profile:]
    capped = [s for s in eval_segments if s.end_ts - split_ts <= args.max_eval_seconds]
    eval_segments = capped or eval_segments[:1]

    # Token-level view: text + entity_type, in the same order as the reference words.
    tokens = [(t.text, t.entity_type) for seg in eval_segments for t in seg.tokens]
    reference = " ".join(t.text for seg in eval_segments for t in seg.tokens)
    ref_words = _normalize(reference).split()

    # Alignment sanity check between the entity mask and normalized reference words.
    if len(ref_words) != len(tokens):
        logger.warning(
            f"MISALIGNMENT: {len(tokens)} NLP tokens but {len(ref_words)} normalized ref words. "
            "Entity indices may be offset — this alone can create spurious entity errors."
        )

    prompt = load_profile(
        call.call_id, args.n_profile, ProfileStrategy.METADATA_ONLY, Path(args.profiles_dir)
    ).prompt

    model = whisper.load_model(args.model, device=device, download_root=args.download_root)
    audio = _load_eval_audio(call, split_ts, eval_segments[-1].end_ts)
    logger.info("Transcribing baseline ...")
    base_hyp = _transcribe(model, audio, None, fp16)
    logger.info("Transcribing prompted ...")
    prom_hyp = _transcribe(model, audio, prompt, fp16)

    base_bad = _error_ref_indices(reference, base_hyp)
    prom_bad = _error_ref_indices(reference, prom_hyp)

    entity_idx = [i for i, (_, et) in enumerate(tokens) if et in VOCABULARY_ENTITY_TYPES]
    logger.info(f"{len(entity_idx)} entity tokens in this slice")
    logger.info(f"{'idx':>4}  {'token':<20} {'type':<12} {'baseline':<9} {'prompted':<9}")
    for i in entity_idx:
        text, et = tokens[i]
        b = "ERR" if i in base_bad else "ok"
        p = "ERR" if i in prom_bad else "ok"
        mark = "  <-- changed" if (i in base_bad) != (i in prom_bad) else ""
        logger.info(f"{i:>4}  {text:<20} {str(et):<12} {b:<9} {p:<9}{mark}")

    logger.info(f"Baseline entity errors: {sum(i in base_bad for i in entity_idx)}/{len(entity_idx)}")
    logger.info(f"Prompted entity errors: {sum(i in prom_bad for i in entity_idx)}/{len(entity_idx)}")
    logger.info(f"baseline hyp: {base_hyp}")
    logger.info(f"prompted hyp: {prom_hyp}")


if __name__ == "__main__":
    main()
