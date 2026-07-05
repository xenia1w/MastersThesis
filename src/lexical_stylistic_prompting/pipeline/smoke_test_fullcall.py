"""
Local smoke test for the full-call metadata_only pipeline — run this BEFORE a SLURM job.

On a single call it runs the unified evaluator twice (baseline, then metadata_only) over a
short, time-bounded slice (default ~5 min, so a laptop can handle it) that still spans
several 30 s Whisper windows, and checks the three failure modes the earlier attempt might
have hit:

  1. CRASH      — both strategies must run to completion.
  2. LEAKING    — the injected keyword list must NOT appear verbatim in the transcription.
  3. NO EFFECT  — the prompt must actually change decoding beyond the first 30 s window
                  (verifies carry_initial_prompt biases the whole call, not just the opening).

It also prints WER + entity-EER for both and the delta, so you can eyeball whether the
prompt helps before committing GPU hours.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.smoke_test_fullcall \\
        --data-dir data/raw/earnings21 --call-id 4392809 --max-eval-seconds 300

    # Faster mechanism check with a small model (WER numbers won't match medium):
        --model base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import whisper
from loguru import logger

from src.asr_adaptation.metrics.wer import _normalize
from src.lexical_stylistic_prompting.data.earnings21_utils import load_earnings21
from src.lexical_stylistic_prompting.models.constants import PROFILES_DIR
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    load_profile,
)
from src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval import evaluate_call

# ~ first-window word budget: 30 s of speech is roughly 60-90 words. Compare tails past this.
FIRST_WINDOW_WORDS = 90


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument("--n-profile", type=int, default=20)
    parser.add_argument("--max-eval-seconds", type=float, default=300.0,
                        help="Cap the audio slice to ~N seconds so a laptop can run it (default 5 min)")
    parser.add_argument("--max-eval-segments", type=int, default=None,
                        help="Alternative segment-count cap (overrides seconds if set)")
    parser.add_argument("--model", default="medium", help="Whisper model (use 'base' for speed)")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--prompt", default=None,
                        help="Override the injected prompt (else the metadata_only profile is loaded)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    fp16 = device == "cuda"
    logger.info(f"Device: {device} | Model: {args.model}")

    calls = load_earnings21(Path(args.data_dir), min_tokens=5)
    calls = [c for c in calls if c.call_id == args.call_id]
    if not calls:
        logger.error(f"Call {args.call_id!r} not found")
        return 1
    call = calls[0]

    prompt_text = args.prompt
    if prompt_text is None:
        try:
            prompt_text = load_profile(
                call.call_id, args.n_profile, ProfileStrategy.METADATA_ONLY, Path(args.profiles_dir)
            ).prompt
        except FileNotFoundError:
            logger.error(
                f"No metadata_only profile for {call.call_id}. Build profiles first, or pass --prompt."
            )
            return 1
    logger.info(f"Prompt ({len(prompt_text.split(','))} terms): {prompt_text}")

    model = whisper.load_model(args.model, device=device, download_root=args.download_root)

    # A segment cap, if given, takes precedence; otherwise bound by seconds.
    secs = None if args.max_eval_segments is not None else args.max_eval_seconds
    logger.info("── Running BASELINE (no prompt) ──")
    base = evaluate_call(call, args.n_profile, "baseline", model, fp16, None,
                         args.max_eval_segments, secs)
    logger.info("── Running METADATA_ONLY (prompted) ──")
    prom = evaluate_call(call, args.n_profile, "metadata_only", model, fp16, prompt_text,
                         args.max_eval_segments, secs)

    # 1. CRASH
    if base is None or prom is None:
        logger.error("FAIL: a strategy returned no row (too few segments?).")
        return 1

    # 2. LEAKING
    hyp_norm = _normalize(prom.hypothesis)
    leaked = [t for t in (x.strip() for x in prompt_text.split(",")) if t and _normalize(t) and
              hyp_norm.startswith(_normalize(t))]
    leak_ok = not leaked
    logger.info(f"[{'PASS' if leak_ok else 'FAIL'}] Leak check: "
                f"{'no prompt terms at start of hypothesis' if leak_ok else f'leaked: {leaked}'}")

    # 3. NO EFFECT beyond the first window
    base_tail = _normalize(base.hypothesis).split()[FIRST_WINDOW_WORDS:]
    prom_tail = _normalize(prom.hypothesis).split()[FIRST_WINDOW_WORDS:]
    biased_whole_call = base_tail != prom_tail
    logger.info(f"[{'PASS' if biased_whole_call else 'WARN'}] Whole-call biasing: "
                f"{'prompt changes decoding past the first ~30 s' if biased_whole_call else 'no change past first window — check carry_initial_prompt / slice length'}")

    # Metrics summary
    logger.info("── Metrics (this slice only) ──")
    logger.info(f"WER        : baseline {base.wer:.4f} → prompted {prom.wer:.4f}  (Δ {prom.wer - base.wer:+.4f})")
    logger.info(f"Entity EER : baseline {base.entity_eer:.4f} → prompted {prom.entity_eer:.4f}  "
                f"(Δ {prom.entity_eer - base.entity_eer:+.4f}, {base.n_entity_tokens} entity tokens)")
    logger.info(f"baseline  hyp: {base.hypothesis[:200]}")
    logger.info(f"prompted  hyp: {prom.hypothesis[:200]}")

    if not leak_ok:
        logger.error("Smoke test FAILED (prompt leaking). Do not submit the SLURM job yet.")
        return 1
    logger.info("Smoke test passed the hard checks (no crash, no leak). Review metrics above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
