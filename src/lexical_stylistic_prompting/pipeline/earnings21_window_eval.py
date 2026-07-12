"""
v2 fixed-window transcription for the lexical/stylistic prompting experiment.

Transcribe a fixed wall-clock audio window ``[eval-start, eval-end]`` of an Earnings21
call with openai-whisper, optionally injecting a speaker profile as ``initial_prompt``.

This deliberately does NOT compute WER/entity-EER. Scoring is deferred and done locally,
because the ground-truth reference for the window is cut from the hand-annotated 5:00 /
15:00 boundaries (which are not needed to run the transcription). The job only saves the
hypothesis, so baseline and prompted runs differ solely by the injected prompt.

v2 windows: profile window = [0, 300] s (used to build profiles), eval window = [300, 900] s.

Usage (baseline, no prompt):
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_window_eval \\
        --data-dir   data/raw/earnings21 \\
        --call-id    4392809 \\
        --eval-start 300 --eval-end 900 \\
        --strategy   baseline \\
        --output     data/processed/lexical_stylistic_prompting/v2/earnings21_window_baseline/baseline_4392809.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import torch
import whisper
from loguru import logger
from pydantic import BaseModel

from src.lexical_stylistic_prompting.models.constants import PROFILES_DIR
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    PromptFormat,
    load_profile,
)

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "medium"


class WindowRow(BaseModel):
    call_id: str
    strategy: str
    prompt: str
    eval_start: float
    eval_end: float
    hypothesis_words: int
    hypothesis: str


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
        kwargs["carry_initial_prompt"] = True  # same prompting mechanism as the v1 eval
    result = model.transcribe(audio, **kwargs)
    return str(result["text"]).strip()


def _load_window(audio_path: Path, start: float, end: float) -> np.ndarray:
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        offset=start,
        duration=end - start,
        mono=True,
    )
    return audio.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--eval-start", type=float, required=True, help="window start (seconds)")
    parser.add_argument("--eval-end", type=float, required=True, help="window end (seconds)")
    parser.add_argument("--strategy", default="baseline",
                        choices=["baseline", *[s.value for s in ProfileStrategy]])
    parser.add_argument("--profiles-dir", default=str(PROFILES_DIR))
    parser.add_argument("--prompt-format", default=PromptFormat.LIST.value,
                        choices=[f.value for f in PromptFormat])
    parser.add_argument("--profile-tag", default="0300",
                        help="tag identifying the profile window in the profile filename")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    fp16 = device == "cuda"
    logger.info(f"Device: {device} | Model: {args.model} | Strategy: {args.strategy} | "
                f"Window: [{args.eval_start:.0f}, {args.eval_end:.0f}]s")

    model = whisper.load_model(args.model, device=device, download_root=args.download_root)

    prompt_text: str | None = None
    if args.strategy != "baseline":
        strategy = ProfileStrategy(args.strategy)
        profile = load_profile(
            args.call_id, args.profile_tag, strategy, Path(args.profiles_dir),
            PromptFormat(args.prompt_format),
        )
        prompt_text = profile.prompt

    audio_path = Path(args.data_dir) / "media" / f"{args.call_id}.mp3"
    audio = _load_window(audio_path, args.eval_start, args.eval_end)
    hypothesis = _transcribe(model, audio, prompt_text, fp16)

    row = WindowRow(
        call_id=args.call_id,
        strategy=args.strategy,
        prompt=prompt_text or "",
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        hypothesis_words=len(hypothesis.split()),
        hypothesis=hypothesis,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(WindowRow.model_fields))
        writer.writeheader()
        writer.writerow(row.model_dump())
    logger.info(f"{args.call_id}: {args.strategy} [{args.eval_start:.0f},{args.eval_end:.0f}] "
                f"-> {len(hypothesis.split())} words -> {out}")


if __name__ == "__main__":
    main()
