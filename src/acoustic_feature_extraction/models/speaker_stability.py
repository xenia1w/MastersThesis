from __future__ import annotations

from typing import Dict, List, Literal

import pydantic
import torch
from pydantic import BaseModel, Field

if pydantic.VERSION.startswith("2."):
    from pydantic import ConfigDict

    class TensorModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

else:

    class TensorModel(BaseModel):
        class Config:
            arbitrary_types_allowed = True

from src.acoustic_feature_extraction.models.wavlm_encoder import WavLMBaseEncoder, WavLMEncoder

DatasetName = Literal["l2arctic", "saa"]
EmbeddingType = Literal["mean_std", "xvector"]


class KFloatMap(BaseModel):
    """Mapping from utterance count k (integer) to a float metric value."""
    values: Dict[int, float] = Field(default_factory=dict)


class EmbeddingsByK(TensorModel):
    """Per-k centroid embeddings plus the full-corpus centroid."""
    by_k: Dict[int, torch.Tensor]
    full: torch.Tensor


class RepresentationConfig(BaseModel):
    name: str
    model_name: str
    embedding_type: EmbeddingType


class RepresentationRuntime(TensorModel):

    config: RepresentationConfig
    xvector_encoder: WavLMEncoder | None = None
    base_encoder: WavLMBaseEncoder | None = None


class RepresentationEmbeddings(TensorModel):

    name: str
    embeddings: List[torch.Tensor]


class SpeakerStabilityPayload(TensorModel):

    dataset: DatasetName
    speaker_id: str
    representation: str
    model_name: str
    ordering: str
    random_seed: int
    ks: List[int]
    ordered_utterance_ids: List[str]
    num_utterances: int
    total_seconds: float
    total_seconds_by_k: KFloatMap
    embeddings_by_k: EmbeddingsByK
    cosine_to_full: KFloatMap
    cosine_consecutive: KFloatMap
    stability_k: int | None
    stability_seconds: float | None
    stability_threshold: float
    stability_epsilon: float
    stability_consecutive: int
    stability_consecutive_k: int | None
    stability_consecutive_seconds: float | None
    stability_consecutive_threshold: float


class SpeakerStabilitySummary(BaseModel):
    speaker_id: str
    representation: str
    model_name: str
    ks: List[int]
    cosine_to_full: KFloatMap
    cosine_consecutive: KFloatMap
    total_seconds_by_k: KFloatMap
    stability_k: int | None
    stability_seconds: float | None
    stability_threshold: float
    stability_epsilon: float
    stability_consecutive: int
    stability_consecutive_k: int | None
    stability_consecutive_seconds: float | None
    stability_consecutive_threshold: float


class SpeakerStabilityCsvRow(BaseModel):
    speaker_id: str
    dataset: DatasetName
    representation: str
    model_name: str
    ordering: str
    random_seed: int
    k: int
    cosine_to_full: float
    cosine_consecutive: float | None
    cumulative_seconds: float
    num_utterances: int
    total_seconds: float
    stability_k: int | None
    stability_seconds: float | None
    stability_threshold: float
    stability_epsilon: float
    stability_consecutive: int
    stability_consecutive_k: int | None
    stability_consecutive_seconds: float | None
    stability_consecutive_threshold: float


class SAASegmentationConfig(BaseModel):
    min_sec: float = 2.0
    max_sec: float = 8.0
    merge_gap_sec: float = 0.25
    top_db: int = 30


class SpeakerStabilityConfig(BaseModel):
    dataset: DatasetName = Field(..., description="Dataset to run: l2arctic or saa.")
    outer_zip: str | None = Field(default=None, description="Path to dataset zip.")
    save_root: str | None = Field(default=None, description="Output directory.")
    ordering: Literal["chronological", "random"] = Field(
        default="chronological", description="Utterance ordering strategy."
    )
    random_seed: int = Field(default=1337, description="Random seed for ordering.")
    ks: List[int] = Field(default_factory=lambda: [1, 2, 3, 5, 10])
    representations: List[RepresentationConfig] = Field(default_factory=list)
    max_speakers: int | None = None
    max_utterances_per_speaker: int | None = None
    use_cached_utterances: bool = True
    utterance_cache_root: str | None = None
    stability_threshold: float = 0.95
    stability_epsilon: float = 0.002
    stability_consecutive: int = 2
    stability_consecutive_threshold: float = 0.999
    saa_segmentation: SAASegmentationConfig | None = None
