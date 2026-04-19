from __future__ import annotations

from typing import List, Tuple

import librosa
import torch

from src.acoustic_feature_extraction.models.acoustic import SAASample, SAASegmentSample


def _merge_intervals(
    intervals: List[Tuple[int, int]],
    max_gap_samples: int,
) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    merged: List[Tuple[int, int]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start - current_end <= max_gap_samples:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _split_long_interval(
    start: int,
    end: int,
    max_samples: int,
    min_samples: int,
) -> List[Tuple[int, int]]:
    if end <= start:
        return []
    if max_samples <= 0:
        return [(start, end)]

    segments: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + max_samples, end)
        # Avoid creating a tiny tail segment by folding it into the current one.
        if end - stop < min_samples and stop < end:
            stop = end
        segments.append((cursor, stop))
        cursor = stop
    return segments


def segment_saa_recording(
    sample: SAASample,
    waveform: torch.Tensor,
    sampling_rate: int,
    min_duration_sec: float = 2.0,
    max_duration_sec: float = 8.0,
    merge_gap_sec: float = 0.25,
    top_db: int = 30,
) -> List[SAASegmentSample]:
    if waveform.numel() == 0:
        return []

    min_samples = max(1, int(min_duration_sec * sampling_rate))
    max_samples = max(min_samples, int(max_duration_sec * sampling_rate))
    max_gap_samples = max(0, int(merge_gap_sec * sampling_rate))

    intervals = librosa.effects.split(
        y=waveform.numpy(),
        top_db=top_db,
        frame_length=2048,
        hop_length=512,
    )
    raw_intervals = [(int(start), int(end)) for start, end in intervals.tolist()]
    merged = _merge_intervals(raw_intervals, max_gap_samples=max_gap_samples)

    segment_bounds: List[Tuple[int, int]] = []
    for start, end in merged:
        duration = end - start
        if duration < min_samples:
            continue
        segment_bounds.extend(
            _split_long_interval(
                start=start,
                end=end,
                max_samples=max_samples,
                min_samples=min_samples,
            )
        )

    out: List[SAASegmentSample] = []
    for idx, (start, end) in enumerate(segment_bounds):
        duration_seconds = float(end - start) / float(sampling_rate)
        segment_id = f"{sample.filename}_seg_{idx:04d}_{start}_{end}"
        out.append(
            SAASegmentSample(
                speaker_id=sample.speaker_id,
                filename=sample.filename,
                segment_id=segment_id,
                start_sample=start,
                end_sample=end,
                sampling_rate=sampling_rate,
                duration_seconds=duration_seconds,
                native_language=sample.native_language,
                sex=sample.sex,
                country=sample.country,
                age=sample.age,
                age_onset=sample.age_onset,
                birthplace=sample.birthplace,
            )
        )
    return out
