from __future__ import annotations

from pathlib import Path

from src.prosodic_feature_extraction.pipeline.extract_perturbation_embeddings import (
    PerturbationRow,
    _embedding_output_path,
    _resolve_manifest_csv,
    ExtractEmbeddingsConfig,
)


def test_resolve_manifest_csv_default() -> None:
    config = ExtractEmbeddingsConfig(dataset="l2arctic")
    manifest = _resolve_manifest_csv(config)
    assert str(manifest).endswith("data/processed/perturbations/l2arctic/manifest.csv")


def test_embedding_output_path(tmp_path: Path) -> None:
    row = PerturbationRow(
        dataset="saa",
        speaker_id="123",
        utterance_id="demo",
        variant="rate_p10",
        source="perturbed",
        output_path="dummy.wav",
        sampling_rate=16000,
        num_samples=32000,
        duration_seconds=2.0,
        params_json="{}",
    )
    out_path = _embedding_output_path(tmp_path, row)
    assert str(out_path).endswith("saa/123/demo__rate_p10.pt")

