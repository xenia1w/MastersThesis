from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.acoustic_feature_extraction.pipeline.analyze_perturbation_sensitivity import (
    EmbeddingManifestRow,
    build_aggregate_rows,
    build_detail_rows,
)


def _write_payload(path: Path, emb_mean: torch.Tensor, emb_meanstd: torch.Tensor, emb_xvec: torch.Tensor) -> None:
    payload = {
        "embedding_mean": emb_mean,
        "embedding_meanstd": emb_meanstd,
        "embedding_xvector": emb_xvec,
    }
    torch.save(payload, path)


def test_build_detail_and_aggregate_rows() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        orig_path = tmp / "orig.pt"
        pert_path = tmp / "pert.pt"
        _write_payload(
            orig_path,
            emb_mean=torch.tensor([1.0, 0.0]),
            emb_meanstd=torch.tensor([1.0, 0.0]),
            emb_xvec=torch.tensor([1.0, 0.0]),
        )
        _write_payload(
            pert_path,
            emb_mean=torch.tensor([0.0, 1.0]),
            emb_meanstd=torch.tensor([0.0, 1.0]),
            emb_xvec=torch.tensor([1.0, 0.0]),
        )

        rows = [
            EmbeddingManifestRow(
                dataset="l2arctic",
                speaker_id="ABA",
                utterance_id="arctic_a0001",
                variant="orig",
                source="original",
                audio_path="orig.wav",
                embedding_path=str(orig_path),
                sampling_rate=16000,
                num_samples=1000,
                duration_seconds=0.0625,
                mean_dim=2,
                meanstd_dim=2,
                xvector_dim=2,
                base_model_name="base",
                sv_model_name="sv",
            ),
            EmbeddingManifestRow(
                dataset="l2arctic",
                speaker_id="ABA",
                utterance_id="arctic_a0001",
                variant="rate_p10",
                source="perturbed",
                audio_path="pert.wav",
                embedding_path=str(pert_path),
                sampling_rate=16000,
                num_samples=1000,
                duration_seconds=0.0625,
                mean_dim=2,
                meanstd_dim=2,
                xvector_dim=2,
                base_model_name="base",
                sv_model_name="sv",
            ),
        ]

        detail = build_detail_rows(rows)
        assert len(detail) == 1
        assert abs(detail[0].cosine_base_mean - 0.0) < 1e-6
        assert abs(detail[0].cosine_sv_xvector - 1.0) < 1e-6

        aggregate = build_aggregate_rows(detail)
        assert len(aggregate) == 1
        assert aggregate[0].n == 1
        assert abs(aggregate[0].delta_xvector_minus_mean_avg - 1.0) < 1e-6

