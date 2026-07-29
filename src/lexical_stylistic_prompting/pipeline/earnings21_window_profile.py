"""
v2 profile-window transcription for the lexical/stylistic prompting experiment.

Transcribe each call's fixed profile window ``[0, window-seconds]`` (default 300 s = 5 min)
with openai-whisper, no prompt. The resulting noisy transcript is what the LLM later reads
to build the transcript-based keyword profiles (transcript_only / transcript_plus_knowledge /
transcript_metadata_knowledge).

Loader-independent: this slices the audio directly by wall-clock time from the media file, so
it does NOT depend on the segment builder (the source of the v1 mis-timing bug). Output is one
JSON per call, keyed ``<call_id>_<profile-tag>.json`` so build_earnings21_profiles.py can
consume it unchanged:
    { call_id, window_seconds, transcript }

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_window_profile \\
        --data-dir data/raw/earnings21 \\
        --output-dir data/processed/lexical_stylistic_prompting/v2/profile_transcripts \\
        --window-seconds 300 --model medium
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import whisper
from loguru import logger

from src.lexical_stylistic_prompting.pipeline.earnings21_window_eval import (
    DEFAULT_MODEL,
    _load_window,
    _transcribe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--call-id", default=None, help="Transcribe a single call only")
    parser.add_argument("--window-seconds", type=float, default=300.0,
                        help="Profile window length in seconds (v2 default 300 = 5 min)")
    parser.add_argument("--profile-tag", default="300",
                        help="Filename tag; must match build_earnings21_profiles --n-profile")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    fp16 = device == "cuda"
    logger.info(f"Device: {device} | Model: {args.model} | window [0, {args.window_seconds:.0f}]s")

    media_dir = Path(args.data_dir) / "media"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.call_id:
        call_ids = [args.call_id]
    else:
        call_ids = [p.stem for p in sorted(media_dir.glob("*.mp3"))]

    model = whisper.load_model(args.model, device=device, download_root=args.download_root)

    for call_id in call_ids:
        out_path = out_dir / f"{call_id}_{args.profile_tag}.json"
        if args.skip_existing and out_path.exists():
            logger.info(f"{call_id}: skipping (cached)")
            continue

        audio = _load_window(media_dir / f"{call_id}.mp3", 0.0, args.window_seconds)
        transcript = _transcribe(model, audio, None, fp16)

        out_path.write_text(json.dumps({
            "call_id": call_id,
            "window_seconds": args.window_seconds,
            "transcript": transcript,
        }, indent=2))
        logger.info(f"{call_id}: saved → {out_path} ({len(transcript.split())} words)")


if __name__ == "__main__":
    main()
