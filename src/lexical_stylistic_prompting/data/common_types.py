from __future__ import annotations

from pydantic import BaseModel, computed_field


class SegmentMeta(BaseModel):
    segment_id: str
    speaker_id: str
    talk_id: str
    text: str


class SpeakerData(BaseModel):
    speaker_id: str
    segments: list[SegmentMeta]

    @computed_field
    @property
    def n_segments(self) -> int:
        return len(self.segments)


class SpeakerDataset(BaseModel):
    speakers: list[SpeakerData]

    @computed_field
    @property
    def n_speakers(self) -> int:
        return len(self.speakers)


class SpeakerSplit(BaseModel):
    speaker_id: str
    profile_segments: list[SegmentMeta]
    test_segments: list[SegmentMeta]

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
    skip_intro: int
    n_profile: int
    n_test: int
    min_segments: int

    @computed_field
    @property
    def n_speakers(self) -> int:
        return len(self.splits)
