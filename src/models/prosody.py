from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict


class L2ArcticSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    wav_name: str


class SAASample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    filename: str
    native_language: str
    sex: Optional[str] = None
    country: Optional[str] = None
    age: Optional[int] = None
    age_onset: Optional[int] = None
    birthplace: Optional[str] = None


class SAAMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    native_language: str
    sex: Optional[str] = None
    country: Optional[str] = None
    age: Optional[int] = None
    age_onset: Optional[int] = None
    birthplace: Optional[str] = None


class ProsodyEmbedding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Literal["l2arctic", "saa"]
    speaker_id: str
    file_name: str
    sampling_rate: int
    frames: torch.Tensor
    utt_emb: torch.Tensor
    utt_emb_meanstd: torch.Tensor
    saa_metadata: Optional[SAAMetadata] = None
