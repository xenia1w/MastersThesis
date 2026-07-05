"""
Transcribe each call's profile window (first n_profile turns) with Whisper, no prompt.

This is the data-dependent first stage of the transcript_only / transcript_plus_knowledge
approaches: the noisy transcript of segments[:n_profile] (audio [0, split_ts]) is what the LLM
later reads to build the keyword list. Output is one JSON per call:
    { call_id, n_profile, n_segments, transcript }

Run it wherever Whisper is available (locally in the background, or as a cluster array). The
resulting JSONs are consumed by build_earnings21_profiles.py --strategy transcript_only.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_profile_window \\
        --data-dir data/raw/earnings21 --n-profile 20 --model medium

    # Single call, capped window (smoke test):
        --call-id 4320211 --max-window-seconds 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import whisper
from loguru import logger

from src.lexical_stylistic_prompting.data.earnings21_utils import load_earnings21
from src.lexical_stylistic_prompting.models.constants import PROFILE_TRANSCRIPTS_DIR
from src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval import (
    DEFAULT_MODEL,
    _load_eval_audio,
    _transcribe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=str(PROFILE_TRANSCRIPTS_DIR))
    parser.add_argument("--n-profile", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--call-id", default=None, help="Transcribe a single call only")
    parser.add_argument("--max-window-seconds", type=float, default=None,
                        help="Cap the transcribed window to ~N seconds (smoke test)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    fp16 = device == "cuda"
    logger.info(f"Device: {device} | Model: {args.model}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calls = load_earnings21(Path(args.data_dir), min_tokens=5)
    if args.call_id:
        calls = [c for c in calls if c.call_id == args.call_id]
        if not calls:
            logger.error(f"Call {args.call_id!r} not found")
            return

    model = whisper.load_model(args.model, device=device, download_root=args.download_root)

    for call in calls:
        out_path = out_dir / f"{call.call_id}_{args.n_profile}.json"
        if args.skip_existing and out_path.exists():
            logger.info(f"{call.call_id}: skipping (cached)")
            continue
        if len(call.segments) <= args.n_profile:
            logger.warning(f"{call.call_id}: only {len(call.segments)} segments — skipping")
            continue

        window_segments = call.segments[:args.n_profile]
        split_ts = window_segments[-1].end_ts
        end_ts = min(split_ts, args.max_window_seconds) if args.max_window_seconds else split_ts

        logger.info(f"{call.call_id}: transcribing profile window [0, {end_ts:.1f}s] "
                    f"({len(window_segments)} turns)")
        audio = _load_eval_audio(call, 0.0, end_ts)
        transcript = _transcribe(model, audio, None, fp16)

        out_path.write_text(json.dumps({
            "call_id": call.call_id,
            "n_profile": args.n_profile,
            "n_segments": len(window_segments),
            "transcript": transcript,
        }, indent=2))
        logger.info(f"{call.call_id}: saved → {out_path} ({len(transcript.split())} words)")


if __name__ == "__main__":
    main()
