from __future__ import annotations

from typing import Literal, Optional

import pydantic
import torch
from pydantic import BaseModel

if pydantic.VERSION.startswith("2."):
    from pydantic import ConfigDict

    class TensorModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

else:

    class TensorModel(BaseModel):
        class Config:
            arbitrary_types_allowed = True


class L2ArcticSample(TensorModel):

    speaker_id: str
    wav_name: str


class SAASample(TensorModel):

    speaker_id: str
    filename: str
    native_language: str
    sex: Optional[str] = None
    country: Optional[str] = None
    age: Optional[int] = None
    age_onset: Optional[int] = None
    birthplace: Optional[str] = None


class SAASegmentSample(TensorModel):

    speaker_id: str
    filename: str
    segment_id: str
    start_sample: int
    end_sample: int
    sampling_rate: int
    duration_seconds: float
    native_language: str
    sex: Optional[str] = None
    country: Optional[str] = None
    age: Optional[int] = None
    age_onset: Optional[int] = None
    birthplace: Optional[str] = None


class SAAMetadata(TensorModel):

    native_language: str
    sex: Optional[str] = None
    country: Optional[str] = None
    age: Optional[int] = None
    age_onset: Optional[int] = None
    birthplace: Optional[str] = None


class ProsodyEmbedding(TensorModel):

    dataset: Literal["l2arctic", "saa"]
    speaker_id: str
    file_name: str
    sampling_rate: int
    frames: torch.Tensor
    utt_emb: torch.Tensor
    utt_emb_meanstd: torch.Tensor
    saa_metadata: Optional[SAAMetadata] = None
