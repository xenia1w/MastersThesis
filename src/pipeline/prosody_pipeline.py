from __future__ import annotations

from typing import Iterable, List, Literal

from pydantic import BaseModel, Field

from src.data.saa_utils import load_saa_samples
from src.models.prosody import L2ArcticSample, ProsodyEmbedding, SAASample
from src.pipeline.l2arctic_minimal import default_samples, run_l2arctic_minimal
from src.pipeline.saa_minimal import run_saa_minimal

DatasetName = Literal["l2arctic", "saa"]
SampleInput = Iterable[L2ArcticSample] | Iterable[SAASample] | None
EmbeddingOutput = List[ProsodyEmbedding]


class ProsodyPipelineConfig(BaseModel):
    dataset: DatasetName = Field(
        ..., description="Dataset to run: l2arctic or saa."
    )
    outer_zip: str | None = Field(
        default=None, description="Path to the dataset zip archive."
    )
    model_name: str = Field(
        default="microsoft/wavlm-base-plus-sv",
        description="Model identifier for feature extraction.",
    )
    save_root: str | None = Field(
        default=None, description="Output directory for saved embeddings."
    )
    max_items: int | None = Field(
        default=None, description="Optional limit for number of samples."
    )
    include_missing: bool = Field(
        default=False, description="Include missing SAA recordings when loading."
    )
    validate_files: bool = Field(
        default=True, description="Validate SAA recordings exist in the archive."
    )


def _resolve_outer_zip(config: ProsodyPipelineConfig) -> str:
    if config.outer_zip is not None:
        return config.outer_zip
    if config.dataset == "l2arctic":
        return "data/raw/l2arctic_release_v5.0.zip"
    return "data/raw/archive.zip"


def _resolve_save_root(config: ProsodyPipelineConfig) -> str:
    if config.save_root is not None:
        return config.save_root
    if config.dataset == "l2arctic":
        return "data/processed/l2arctic_minimal_embeddings"
    return "data/processed/saa_minimal_embeddings"


def run_prosody_pipeline(
    config: ProsodyPipelineConfig,
    samples: SampleInput = None,
) -> EmbeddingOutput:
    outer_zip = _resolve_outer_zip(config)
    save_root = _resolve_save_root(config)

    if config.dataset == "l2arctic":
        sample_list = list(samples) if samples is not None else default_samples()
        if config.max_items is not None:
            sample_list = sample_list[: config.max_items]
        return run_l2arctic_minimal(
            outer_zip=outer_zip,
            samples=sample_list,
            model_name=config.model_name,
            save_root=save_root,
        )

    sample_list = (
        list(samples)
        if samples is not None
        else load_saa_samples(
            outer_zip,
            include_missing=config.include_missing,
            validate_files=config.validate_files,
        )
    )
    if config.max_items is not None:
        sample_list = sample_list[: config.max_items]

    return run_saa_minimal(
        outer_zip=outer_zip,
        samples=sample_list,
        model_name=config.model_name,
        save_root=save_root,
    )
