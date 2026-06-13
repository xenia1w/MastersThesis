from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.data.tedlium_utils import (
    FILTERED_DATASET_PATH,
    SpeakerSplit,
    build_splits,
)

DEFAULT_MODEL = "openai/whisper-small"


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
    max_test_segments: int | None = None,
) -> list[TedliumBaselineRow]:
    segments = split.test_segments
    if max_test_segments is not None:
        segments = segments[:max_test_segments]
    rows: list[TedliumBaselineRow] = []
    for seg in tqdm(segments, desc=f"{split.speaker_id}", unit="seg"):
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

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading {args.model} ...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        chunk_length_s=30,
    )

    logger.info("Building speaker splits ...")
    dataset_splits = build_splits(
        skip_intro=args.skip_intro,
        n_profile=args.n_profile,
        n_test=args.n_test,
        min_segments=args.min_segments,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
    )

    # ── Filter speakers before touching audio ────────────────────────────────
    splits_to_process = dataset_splits.splits

    if args.speaker:
        splits_to_process = [s for s in splits_to_process if s.speaker_id == args.speaker]
        if not splits_to_process:
            logger.error(f"Speaker {args.speaker!r} not found in splits (check --min-segments)")
            return

    if args.speakers_file:
        selected = {line.strip() for line in Path(args.speakers_file).read_text().splitlines() if line.strip()}
        splits_to_process = [s for s in splits_to_process if s.speaker_id in selected]
        logger.info(f"Filtered to {len(splits_to_process)} speakers from {args.speakers_file}")
        if not splits_to_process:
            logger.error("No matching speakers found — check that speakers_selected.txt uses full talk IDs (e.g. AlGore_2006)")
            return

    # ── Build audio index — only for segments we actually need ───────────────
    n_max_test = args.max_test_segments
    needed_ids: set[str] = {
        seg.segment_id
        for split in splits_to_process
        for seg in (split.test_segments[:n_max_test] if n_max_test else split.test_segments)
    }
    logger.info(f"Building audio index for {len(needed_ids)} segments across {len(splits_to_process)} speakers ...")

    dataset_path = Path(args.dataset_path) if args.dataset_path else FILTERED_DATASET_PATH
    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path} ...")
        dataset = load_from_disk(str(dataset_path))
        assert isinstance(dataset, Dataset)
        if args.max_examples is not None:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    else:
        hf_split = f"train[:{args.max_examples}]" if args.max_examples is not None else "train"
        dataset = load_dataset(
            "distil-whisper/tedlium", "release3",
            split=hf_split,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    seg_to_example: dict = {}
    for ex in tqdm(dataset, desc="Building audio index", unit="seg"):
        if ex["id"] in needed_ids:
            seg_to_example[ex["id"]] = ex
            if len(seg_to_example) == len(needed_ids):
                break  # found everything we need — stop scanning early

    logger.info(f"Indexed {len(seg_to_example)} / {len(needed_ids)} segments")

    # ── Transcribe ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    all_rows: list[TedliumBaselineRow] = []

    for split in splits_to_process:
        n_test = min(len(split.test_segments), n_max_test or len(split.test_segments))
        logger.info(f"Transcribing speaker {split.speaker_id} ({n_test} test segments) ...")
        rows = run_speaker_baseline(split, asr_pipe, seg_to_example, n_max_test)
        all_rows.extend(rows)

        if rows:
            mean_wer = sum(r.wer for r in rows) / len(rows)
            logger.info(f"{split.speaker_id}: mean WER = {mean_wer:.4f} ({len(rows)} segments)")

        csv_name = f"tedlium_baseline_{split.speaker_id}.csv"
        _save_csv(rows, output_dir / csv_name)

    if all_rows:
        overall_mean = sum(r.wer for r in all_rows) / len(all_rows)
        logger.info(f"Overall mean WER = {overall_mean:.4f} ({len(all_rows)} segments, {len(splits_to_process)} speakers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline WER evaluation on TED-LIUM 3 using Whisper (no prompting)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name on HuggingFace Hub")
    parser.add_argument("--speaker", default=None, help="Evaluate a single speaker only (SLURM-friendly)")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV(s)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model/dataset cache directory")
    parser.add_argument("--skip-intro",   type=int, default=5,  help="Segments to skip at the start of each talk (intro)")
    parser.add_argument("--n-profile",    type=int, default=20, help="Fixed number of profile segments (from middle of talk)")
    parser.add_argument("--n-test",       type=int, default=40, help="Fixed number of test segments (anchored to end of talk)")
    parser.add_argument("--min-segments", type=int, default=65, help="Minimum segments to include a speaker (must be >= skip-intro + n-profile + n-test)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit dataset rows loaded (smoke test)")
    parser.add_argument("--speakers-file",  default=None, help="Path to file with selected talk IDs, one per line")
    parser.add_argument("--dataset-path",   default=None, help="Path to a pre-filtered HF dataset on disk (overrides default)")
    parser.add_argument("--max-test-segments", type=int, default=None, help="Cap test segments per speaker (smoke test)")
    main(parser.parse_args())

