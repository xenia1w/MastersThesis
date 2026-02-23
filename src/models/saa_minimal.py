from __future__ import annotations

from typing import List, Optional

import torch
from pydantic import BaseModel, ConfigDict


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


class SAAEmbedding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    filename: str
    native_language: str
    sex: Optional[str] = None
    country: Optional[str] = None
    age: Optional[int] = None
    age_onset: Optional[int] = None
    birthplace: Optional[str] = None
    sampling_rate: int
    frames: torch.Tensor
    utt_emb: torch.Tensor
    utt_emb_meanstd: torch.Tensor


class LanguageGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    native_language: str
    embeddings: List[SAAEmbedding]


class LanguageCentroid(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    native_language: str
    centroid: torch.Tensor
