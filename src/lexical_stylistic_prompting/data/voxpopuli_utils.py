from __future__ import annotations

from datasets import concatenate_datasets, load_dataset
from loguru import logger
from tqdm import tqdm

from src.lexical_stylistic_prompting.data.common_types import (
    DatasetSplits,
    SegmentMeta,
    SpeakerData,
    SpeakerDataset,
    SpeakerSplit,
)

VOXPOPULI_REPO = "facebook/voxpopuli"
VOXPOPULI_CONFIG = "en"


def load_voxpopuli_speakers(
    min_segments: int = 65,
    cache_dir: str | None = None,
    max_examples: int | None = None,
    splits: list[str] | None = None,
) -> SpeakerDataset:
    if splits is None:
        splits = ["train", "validation", "test"]

    parts = []
    for split in splits:
        split_str = f"{split}[:{max_examples}]" if max_examples and split == splits[0] else split
        logger.info(f"Loading VoxPopuli {VOXPOPULI_CONFIG}/{split} ...")
        parts.append(
            load_dataset(
                VOXPOPULI_REPO,
                VOXPOPULI_CONFIG,
                split=split_str,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        )

    dataset = concatenate_datasets(parts)

    by_speaker: dict[str, list[SegmentMeta]] = {}
    for example in tqdm(dataset, desc="Indexing speakers", unit="seg"):
        if not example.get("is_gold_transcript", True):
            continue
        text = example["normalized_text"].strip()
        if not text:
            continue
        seg = SegmentMeta(
            segment_id=example["audio_id"],
            speaker_id=example["speaker_id"],
            talk_id=example["speaker_id"],
            text=text,
        )
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    speakers: list[SpeakerData] = []
    n_dropped = 0
    for speaker_id, segs in sorted(by_speaker.items()):
        segs_sorted = sorted(segs, key=lambda s: s.segment_id)
        if len(segs_sorted) < min_segments:
            n_dropped += 1
            continue
        speakers.append(SpeakerData(speaker_id=speaker_id, segments=segs_sorted))

    logger.info(
        f"Kept {len(speakers)} speakers with >= {min_segments} segments "
        f"(dropped {n_dropped})"
    )
    return SpeakerDataset(speakers=speakers)


def build_voxpopuli_splits(
    skip_intro: int = 5,
    n_profile: int = 20,
    n_test: int = 40,
    min_segments: int = 65,
    cache_dir: str | None = None,
    max_examples: int | None = None,
    splits: list[str] | None = None,
) -> DatasetSplits:
    assert skip_intro + n_profile + n_test <= min_segments, (
        f"min_segments ({min_segments}) must be >= skip_intro + n_profile + n_test "
        f"({skip_intro} + {n_profile} + {n_test} = {skip_intro + n_profile + n_test})"
    )
    voxpopuli = load_voxpopuli_speakers(
        min_segments=min_segments,
        cache_dir=cache_dir,
        max_examples=max_examples,
        splits=splits,
    )

    result: list[SpeakerSplit] = []
    for speaker in voxpopuli.speakers:
        n = len(speaker.segments)
        profile_end = skip_intro + n_profile
        test_start = n - n_test

        if profile_end > test_start:
            logger.warning(
                f"Skipping {speaker.speaker_id}: windows overlap "
                f"({n} segs, profile_end={profile_end}, test_start={test_start})"
            )
            continue

        result.append(
            SpeakerSplit(
                speaker_id=speaker.speaker_id,
                profile_segments=speaker.segments[skip_intro:profile_end],
                test_segments=speaker.segments[test_start:],
            )
        )

    logger.info(f"Built splits for {len(result)} speakers")
    return DatasetSplits(
        splits=result,
        skip_intro=skip_intro,
        n_profile=n_profile,
        n_test=n_test,
        min_segments=min_segments,
    )
