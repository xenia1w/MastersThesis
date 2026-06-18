from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.data.common_types import SpeakerSplit
from src.lexical_stylistic_prompting.data.voxpopuli_utils import (
    VOXPOPULI_CONFIG,
    VOXPOPULI_REPO,
    build_voxpopuli_splits,
)

DEFAULT_MODEL = "openai/whisper-medium"


class VoxPopuliBaselineRow(BaseModel):
    speaker_id: str
    segment_id: str
    reference: str
    hypothesis: str
    wer: float


def run_speaker_baseline(
    split: SpeakerSplit,
    asr_pipe,
    seg_to_example: dict,
    max_test_segments: int | None = None,
) -> list[VoxPopuliBaselineRow]:
    segments = split.test_segments
    if max_test_segments is not None:
        segments = segments[:max_test_segments]
    rows: list[VoxPopuliBaselineRow] = []
    for seg in tqdm(segments, desc=split.speaker_id, unit="seg"):
        example = seg_to_example.get(seg.segment_id)
        if example is None:
            logger.warning(f"Segment {seg.segment_id} not found, skipping")
            continue
        result = asr_pipe(example["audio"])
        hypothesis = result["text"].strip()
        wer = compute_wer([seg.text], [hypothesis])
        rows.append(
            VoxPopuliBaselineRow(
                speaker_id=split.speaker_id,
                segment_id=seg.segment_id,
                reference=seg.text,
                hypothesis=hypothesis,
                wer=wer,
            )
        )
    return rows


def _save_csv(rows: list[VoxPopuliBaselineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(VoxPopuliBaselineRow.model_fields))
        writer.writeheader()
        for row in rows:
            d = row.model_dump()
            d["wer"] = round(d["wer"], 4)
            writer.writerow(d)
    logger.info(f"Saved {len(rows)} rows → {path}")


def _load_dataset(cache_dir: str | None, splits: list[str] | None = None) -> object:
    hf_splits = splits if splits is not None else ["train", "validation", "test"]
    parts = []
    for split in hf_splits:
        logger.info(f"Loading VoxPopuli {split} ...")
        parts.append(
            load_dataset(
                VOXPOPULI_REPO,
                VOXPOPULI_CONFIG,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        )
    return concatenate_datasets(parts)


def main(args: argparse.Namespace) -> None:
    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading {args.model} ...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        chunk_length_s=30,
    )

    hf_splits = args.splits.split(",") if args.splits else None

    logger.info("Building VoxPopuli speaker splits ...")
    dataset_splits = build_voxpopuli_splits(
        skip_intro=args.skip_intro,
        n_profile=args.n_profile,
        n_test=args.n_test,
        min_segments=args.min_segments,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
        splits=hf_splits,
    )
    logger.info(f"Total eligible speakers: {dataset_splits.n_speakers}")

    splits_to_process = dataset_splits.splits
    if args.n_speakers:
        splits_to_process = splits_to_process[: args.n_speakers]
        logger.info(f"Spot-check mode: processing first {args.n_speakers} speakers")

    if args.speaker:
        splits_to_process = [s for s in splits_to_process if s.speaker_id == args.speaker]
        if not splits_to_process:
            logger.error(f"Speaker {args.speaker!r} not found")
            return

    n_max_test = args.max_test_segments
    needed_ids: set[str] = {
        seg.segment_id
        for split in splits_to_process
        for seg in (split.test_segments[:n_max_test] if n_max_test else split.test_segments)
    }
    logger.info(f"Need {len(needed_ids)} segments across {len(splits_to_process)} speakers")

    dataset = _load_dataset(args.cache_dir, splits=hf_splits)
    seg_to_example: dict = {}
    for ex in tqdm(dataset, desc="Building audio index", unit="seg"):
        if ex["audio_id"] in needed_ids:
            seg_to_example[ex["audio_id"]] = ex
            if len(seg_to_example) == len(needed_ids):
                break

    logger.info(f"Indexed {len(seg_to_example)} / {len(needed_ids)} segments")

    output_dir = Path(args.output_dir)
    all_rows: list[VoxPopuliBaselineRow] = []

    for split in splits_to_process:
        n_test = min(len(split.test_segments), n_max_test or len(split.test_segments))
        logger.info(f"Transcribing {split.speaker_id} ({n_test} segments) ...")
        rows = run_speaker_baseline(split, asr_pipe, seg_to_example, n_max_test)
        all_rows.extend(rows)

        if rows:
            mean_wer = sum(r.wer for r in rows) / len(rows)
            logger.info(f"{split.speaker_id}: mean WER = {mean_wer:.4f} ({len(rows)} segments)")

        csv_name = f"voxpopuli_baseline_{split.speaker_id}.csv"
        _save_csv(rows, output_dir / csv_name)

    if all_rows:
        overall_mean = sum(r.wer for r in all_rows) / len(all_rows)
        logger.info(f"Overall mean WER = {overall_mean:.4f} ({len(all_rows)} segments, {len(splits_to_process)} speakers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline WER evaluation on VoxPopuli English using Whisper (no prompting)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--speaker", default=None, help="Evaluate a single speaker only")
    parser.add_argument("--n-speakers", type=int, default=None, help="Spot-check: evaluate only the first N eligible speakers")
    parser.add_argument("--skip-intro",   type=int, default=5)
    parser.add_argument("--n-profile",    type=int, default=20)
    parser.add_argument("--n-test",       type=int, default=40)
    parser.add_argument("--min-segments", type=int, default=65)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-test-segments", type=int, default=None)
    parser.add_argument("--splits", default=None, help="Comma-separated HF splits to load, e.g. 'train' or 'train,validation,test' (default: all three)")
    main(parser.parse_args())
