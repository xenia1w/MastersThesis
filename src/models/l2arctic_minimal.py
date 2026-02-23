from __future__ import annotations

from typing import List

import torch
from pydantic import BaseModel, ConfigDict


class L2ArcticSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    wav_name: str


class L2ArcticEmbedding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    wav_name: str
    sampling_rate: int
    frames: torch.Tensor
    utt_emb: torch.Tensor
    utt_emb_meanstd: torch.Tensor


class SpeakerGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    embeddings: List[L2ArcticEmbedding]


class SpeakerCentroid(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    centroid: torch.Tensor
