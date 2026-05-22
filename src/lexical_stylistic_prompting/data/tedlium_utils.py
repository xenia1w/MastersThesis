from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel, computed_field


class TedliumSegmentMeta(BaseModel):
    segment_id: str
    speaker_id: str
    talk_id: str
    text: str


class SpeakerData(BaseModel):
    speaker_id: str
    segments: list[TedliumSegmentMeta]

    @computed_field
    @property
    def n_segments(self) -> int:
        return len(self.segments)


class TedliumDataset(BaseModel):
    speakers: list[SpeakerData]

    @computed_field
    @property
    def n_speakers(self) -> int:
        return len(self.speakers)


class SpeakerSplit(BaseModel):
    speaker_id: str
    profile_segments: list[TedliumSegmentMeta]
    test_segments: list[TedliumSegmentMeta]

    @computed_field
    @property
    def n_profile(self) -> int:
        return len(self.profile_segments)

    @computed_field
    @property
    def n_test(self) -> int:
        return len(self.test_segments)


class DatasetSplits(BaseModel):
    splits: list[SpeakerSplit]
    n_profile: int
    min_segments: int

    @computed_field
    @property
    def n_speakers(self) -> int:
        return len(self.splits)


class ManifestRow(BaseModel):
    speaker_id: str
    total_segments: int
    n_profile: int
    n_test: int
    profile_segment_ids: str
    test_segment_ids: str


def load_tedlium_speakers(
    min_segments: int = 20,
    cache_dir: str | None = None,
    max_examples: int | None = None,
) -> TedliumDataset:
    split = f"train[:{max_examples}]" if max_examples is not None else "train"
    logger.info(f"Loading TED-LIUM 3 (release3, split={split!r}) ...")
    dataset = load_dataset("distil-whisper/tedlium", "release3", split=split, cache_dir=cache_dir, trust_remote_code=True)

    by_speaker: dict[str, list[TedliumSegmentMeta]] = {}
    for example in dataset:
        seg = TedliumSegmentMeta(
            segment_id=example["id"],
            speaker_id=example["speaker_id"],
            talk_id=Path(example["file"]).stem,
            text=example["text"],
        )
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    speakers: list[SpeakerData] = []
    n_dropped = 0
    for speaker_id, segs in by_speaker.items():
        segs_sorted = sorted(segs, key=lambda s: s.segment_id)
        if len(segs_sorted) < min_segments:
            n_dropped += 1
            continue
        speakers.append(SpeakerData(speaker_id=speaker_id, segments=segs_sorted))

    logger.info(
        f"Kept {len(speakers)} speakers with ≥ {min_segments} segments "
        f"(dropped {n_dropped})"
    )
    return TedliumDataset(speakers=speakers)


def build_splits(
    min_segments: int = 20,
    n_profile: int = 10,
    cache_dir: str | None = None,
    max_examples: int | None = None,
) -> DatasetSplits:
    tedlium = load_tedlium_speakers(min_segments=min_segments, cache_dir=cache_dir, max_examples=max_examples)

    splits: list[SpeakerSplit] = []
    for speaker in tedlium.speakers:
        splits.append(
            SpeakerSplit(
                speaker_id=speaker.speaker_id,
                profile_segments=speaker.segments[:n_profile],
                test_segments=speaker.segments[n_profile:],
            )
        )

    logger.info(
        f"Built splits for {len(splits)} speakers "
        f"(n_profile={n_profile}, min_segments={min_segments})"
    )
    return DatasetSplits(splits=splits, n_profile=n_profile, min_segments=min_segments)


def _save_manifest(dataset_splits: DatasetSplits, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = [
        ManifestRow(
            speaker_id=s.speaker_id,
            total_segments=s.n_profile + s.n_test,
            n_profile=s.n_profile,
            n_test=s.n_test,
            profile_segment_ids=",".join(seg.segment_id for seg in s.profile_segments),
            test_segment_ids=",".join(seg.segment_id for seg in s.test_segments),
        )
        for s in dataset_splits.splits
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ManifestRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    logger.info(f"Saved manifest with {len(rows)} speakers → {path}")


def main(args: argparse.Namespace) -> None:
    dataset_splits = build_splits(
        min_segments=args.min_segments,
        n_profile=args.n_profile,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
    )
    _save_manifest(dataset_splits, Path(args.output_dir) / "tedlium_manifest.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TED-LIUM 3 speaker splits for lexical/stylistic prompting"
    )
    parser.add_argument("--n-profile", type=int, default=10, help="Segments per speaker in profile set")
    parser.add_argument("--min-segments", type=int, default=20, help="Minimum segments to include a speaker")
    parser.add_argument("--output-dir", required=True, help="Directory for tedlium_manifest.csv")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace dataset cache directory")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit rows loaded (smoke test)")
    main(parser.parse_args())
