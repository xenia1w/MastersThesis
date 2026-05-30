from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.data.tedlium_utils import (
    SpeakerSplit,
    build_splits,
)

DEFAULT_MODEL = "openai/whisper-medium"


class TedliumBaselineRow(BaseModel):
    speaker_id: str
    segment_id: str
    reference: str
    hypothesis: str
    wer: float


def run_speaker_baseline(
    split: SpeakerSplit,
    asr_pipe,
    seg_to_example: dict,
) -> list[TedliumBaselineRow]:
    rows: list[TedliumBaselineRow] = []
    for seg in split.test_segments:
        example = seg_to_example.get(seg.segment_id)
        if example is None:
            logger.warning(f"Segment {seg.segment_id} not found in dataset, skipping")
            continue
        audio = example["audio"]  # dict: {"array": np.ndarray, "sampling_rate": int}
        result = asr_pipe(audio)
        hypothesis = result["text"].strip()
        wer = compute_wer([seg.text], [hypothesis])
        rows.append(
            TedliumBaselineRow(
                speaker_id=split.speaker_id,
                segment_id=seg.segment_id,
                reference=seg.text,
                hypothesis=hypothesis,
                wer=wer,
            )
        )
    return rows


def _save_csv(rows: list[TedliumBaselineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TedliumBaselineRow.model_fields))
        writer.writeheader()
        for row in rows:
            d = row.model_dump()
            d["wer"] = round(d["wer"], 4)
            writer.writerow(d)
    logger.info(f"Saved {len(rows)} rows → {path}")


def main(args: argparse.Namespace) -> None:
    if args.cache_dir:
        # HF_HOME must be set before any HuggingFace import resolves its cache path
        os.environ.setdefault("HF_HOME", args.cache_dir)

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Device: {'cuda' if device == 0 else 'cpu'}")

    logger.info(f"Loading {args.model} ...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        chunk_length_s=30,
    )

    logger.info("Building speaker splits ...")
    dataset_splits = build_splits(
        n_profile=args.n_profile,
        min_segments=args.min_segments,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
    )

    logger.info("Loading TED-LIUM audio index ...")
    hf_split = f"train[:{args.max_examples}]" if args.max_examples is not None else "train"
    dataset = load_dataset(
        "distil-whisper/tedlium", "release3",
        split=hf_split,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    seg_to_example = {ex["id"]: ex for ex in dataset}

    splits_to_process = dataset_splits.splits
    if args.speaker:
        splits_to_process = [s for s in splits_to_process if s.speaker_id == args.speaker]
        if not splits_to_process:
            logger.error(f"Speaker {args.speaker!r} not found in splits (check --min-segments)")
            return

    output_dir = Path(args.output_dir)
    all_rows: list[TedliumBaselineRow] = []

    for split in splits_to_process:
        logger.info(f"Transcribing speaker {split.speaker_id} ({split.n_test} test segments) ...")
        rows = run_speaker_baseline(split, asr_pipe, seg_to_example)
        all_rows.extend(rows)

        if rows:
            mean_wer = sum(r.wer for r in rows) / len(rows)
            print(f"{split.speaker_id}: mean WER = {mean_wer:.4f} ({len(rows)} segments)")

        if args.speaker:
            csv_name = f"tedlium_baseline_{split.speaker_id}.csv"
            _save_csv(rows, output_dir / csv_name)

    if not args.speaker:
        _save_csv(all_rows, output_dir / "tedlium_baseline.csv")
        if all_rows:
            overall_mean = sum(r.wer for r in all_rows) / len(all_rows)
            print(f"\nOverall mean WER = {overall_mean:.4f} ({len(all_rows)} segments, {len(splits_to_process)} speakers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline WER evaluation on TED-LIUM 3 using Whisper (no prompting)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name on HuggingFace Hub")
    parser.add_argument("--speaker", default=None, help="Evaluate a single speaker only (SLURM-friendly)")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV(s)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model/dataset cache directory")
    parser.add_argument("--n-profile", type=int, default=10, help="Profile segments per speaker (must match manifest)")
    parser.add_argument("--min-segments", type=int, default=20, help="Minimum segments to include a speaker")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit dataset rows loaded (smoke test)")
    main(parser.parse_args())
