from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Sequence, cast

import librosa
import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.acoustic_feature_extraction.features.utterance_embedding import mean_pool, mean_std_pool
from src.acoustic_feature_extraction.models.wav2vec2_encoder import Wav2Vec2ProfileExtractor
from src.acoustic_feature_extraction.models.wavlm_encoder import WavLMBaseEncoder, WavLMEncoder

DatasetName = Literal["l2arctic", "saa"]


class ExtractEmbeddingsConfig(BaseModel):
    dataset: DatasetName = Field(..., description="Dataset to run: l2arctic or saa.")
    perturbation_root: str = Field(
        default="data/processed/perturbations",
        description="Root directory that contains perturbation audio and manifest.",
    )
    manifest_csv: str | None = Field(
        default=None,
        description="Optional explicit path to perturbation manifest CSV.",
    )
    out_root: str = Field(
        default="data/processed/perturbation_embeddings",
        description="Root output directory for extracted embeddings.",
    )
    base_model_name: str = Field(
        default="microsoft/wavlm-base-plus",
        description="Base WavLM model for frame-level embeddings.",
    )
    sv_model_name: str = Field(
        default="microsoft/wavlm-base-plus-sv",
        description="WavLM x-vector model for speaker embeddings.",
    )
    target_sr: int = Field(default=16000, description="Target sampling rate.")
    max_items: int | None = Field(
        default=None,
        description="Optional limit for number of manifest rows to process.",
    )
    encoder_type: str = Field(
        default="wavlm",
        description="Encoder family for frame-level embeddings: 'wavlm' or 'wav2vec2'.",
    )


class PerturbationRow(BaseModel):
    dataset: DatasetName
    speaker_id: str
    utterance_id: str
    variant: str
    source: str
    output_path: str
    sampling_rate: int
    num_samples: int
    duration_seconds: float
    params_json: str


class EmbeddingManifestRow(BaseModel):
    dataset: DatasetName
    speaker_id: str
    utterance_id: str
    variant: str
    source: str
    audio_path: str
    embedding_path: str
    sampling_rate: int
    num_samples: int
    duration_seconds: float
    mean_dim: int
    meanstd_dim: int
    xvector_dim: int
    base_model_name: str
    sv_model_name: str


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_fields(model_cls: type[BaseModel]) -> List[str]:
    if hasattr(model_cls, "model_fields"):
        return list(model_cls.model_fields.keys())
    return list(model_cls.__fields__.keys())


def _resolve_manifest_csv(config: ExtractEmbeddingsConfig) -> Path:
    if config.manifest_csv is not None:
        return Path(config.manifest_csv)
    return Path(config.perturbation_root) / config.dataset / "manifest.csv"


def _load_manifest(csv_path: Path) -> List[PerturbationRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Perturbation manifest not found: {csv_path}")
    rows: List[PerturbationRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            dataset_raw = (raw.get("dataset") or "").strip()
            if dataset_raw not in {"l2arctic", "saa"}:
                raise ValueError(f"Invalid dataset value in manifest: {dataset_raw}")
            dataset = cast(DatasetName, dataset_raw)

            rows.append(
                PerturbationRow(
                    dataset=dataset,
                    speaker_id=(raw.get("speaker_id") or "").strip(),
                    utterance_id=(raw.get("utterance_id") or "").strip(),
                    variant=(raw.get("variant") or "").strip(),
                    source=(raw.get("source") or "").strip(),
                    output_path=(raw.get("output_path") or "").strip(),
                    sampling_rate=int(raw.get("sampling_rate") or "0"),
                    num_samples=int(raw.get("num_samples") or "0"),
                    duration_seconds=float(raw.get("duration_seconds") or "0.0"),
                    params_json=(raw.get("params_json") or "{}").strip(),
                )
            )
    return rows


def _load_audio(audio_path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if int(sr) != int(target_sr):
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
        sr = target_sr
    waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
    return waveform, int(sr)


def _embedding_output_path(
    out_root: Path,
    row: PerturbationRow,
) -> Path:
    speaker_dir = out_root / row.dataset / row.speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    return speaker_dir / f"{row.utterance_id}__{row.variant}.pt"


def _save_embeddings_payload(
    out_path: Path,
    row: PerturbationRow,
    waveform: torch.Tensor,
    sr: int,
    mean_embedding: torch.Tensor,
    meanstd_embedding: torch.Tensor,
    xvector_embedding: torch.Tensor,
    base_model_name: str,
    sv_model_name: str,
) -> None:
    payload = {
        "dataset": row.dataset,
        "speaker_id": row.speaker_id,
        "utterance_id": row.utterance_id,
        "variant": row.variant,
        "source": row.source,
        "audio_path": row.output_path,
        "sampling_rate": sr,
        "num_samples": int(waveform.shape[0]),
        "duration_seconds": float(waveform.shape[0]) / float(sr),
        "params": json.loads(row.params_json) if row.params_json else {},
        "base_model_name": base_model_name,
        "sv_model_name": sv_model_name,
        "embedding_mean": mean_embedding.cpu(),
        "embedding_meanstd": meanstd_embedding.cpu(),
        "embedding_xvector": xvector_embedding.cpu(),
    }
    torch.save(payload, out_path)


def _write_embedding_manifest(
    out_root: Path,
    dataset: DatasetName,
    rows: List[EmbeddingManifestRow],
) -> Path:
    out_path = out_root / dataset / "manifest_embeddings.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = _model_fields(EmbeddingManifestRow)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_model_dump(row))
    return out_path


def run_extract_perturbation_embeddings(config: ExtractEmbeddingsConfig) -> Path:
    manifest_csv = _resolve_manifest_csv(config)
    perturbation_rows = _load_manifest(manifest_csv)
    if config.max_items is not None:
        perturbation_rows = perturbation_rows[: config.max_items]
    if not perturbation_rows:
        raise RuntimeError("No perturbation rows found to process.")

    out_root = Path(config.out_root)
    use_wav2vec2 = config.encoder_type == "wav2vec2"
    if use_wav2vec2:
        wav2vec2_encoder = Wav2Vec2ProfileExtractor(model_name=config.base_model_name)
        base_encoder = None
        sv_encoder = None
    else:
        wav2vec2_encoder = None
        base_encoder = WavLMBaseEncoder(model_name=config.base_model_name)
        sv_encoder = WavLMEncoder(model_name=config.sv_model_name)
    manifest_out_rows: List[EmbeddingManifestRow] = []

    for row in tqdm(perturbation_rows, desc="Extracting perturbation embeddings", unit="wav"):
        audio_path = Path(row.output_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = _load_audio(audio_path=audio_path, target_sr=config.target_sr)
        if use_wav2vec2:
            assert wav2vec2_encoder is not None
            frames = wav2vec2_encoder.encode_frames(waveform, sr)
            emb_mean = mean_pool(frames)
            emb_meanstd = mean_std_pool(frames)
            emb_xvector = torch.empty(0)
        else:
            assert base_encoder is not None
            assert sv_encoder is not None
            frames = base_encoder.encode_frames(waveform, sr)
            emb_mean = mean_pool(frames)
            emb_meanstd = mean_std_pool(frames)
            emb_xvector = sv_encoder.encode_utterance(waveform, sr)

        out_path = _embedding_output_path(out_root=out_root, row=row)
        _save_embeddings_payload(
            out_path=out_path,
            row=row,
            waveform=waveform,
            sr=sr,
            mean_embedding=emb_mean,
            meanstd_embedding=emb_meanstd,
            xvector_embedding=emb_xvector,
            base_model_name=config.base_model_name,
            sv_model_name=config.sv_model_name,
        )

        manifest_out_rows.append(
            EmbeddingManifestRow(
                dataset=row.dataset,
                speaker_id=row.speaker_id,
                utterance_id=row.utterance_id,
                variant=row.variant,
                source=row.source,
                audio_path=row.output_path,
                embedding_path=str(out_path),
                sampling_rate=sr,
                num_samples=int(waveform.shape[0]),
                duration_seconds=float(waveform.shape[0]) / float(sr),
                mean_dim=int(emb_mean.shape[0]),
                meanstd_dim=int(emb_meanstd.shape[0]),
                xvector_dim=int(emb_xvector.shape[0]),
                base_model_name=config.base_model_name,
                sv_model_name=config.sv_model_name,
            )
        )

    return _write_embedding_manifest(
        out_root=out_root,
        dataset=config.dataset,
        rows=manifest_out_rows,
    )


def parse_args(argv: Sequence[str]) -> ExtractEmbeddingsConfig:
    parser = argparse.ArgumentParser(
        description="Extract WavLM embeddings for original and perturbed audio."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["l2arctic", "saa"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--perturbation-root",
        default="data/processed/perturbations",
        help="Root containing perturbation outputs and manifest.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Optional explicit manifest CSV path.",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/perturbation_embeddings",
        help="Root output directory for embedding payloads.",
    )
    parser.add_argument(
        "--base-model-name",
        default="microsoft/wavlm-base-plus",
        help="WavLM base model used for frame-level pooling embeddings.",
    )
    parser.add_argument(
        "--sv-model-name",
        default="microsoft/wavlm-base-plus-sv",
        help="WavLM speaker verification model used for x-vector embeddings.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sampling rate for feature extraction.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of files to process.",
    )
    parser.add_argument(
        "--encoder-type",
        default="wavlm",
        choices=["wavlm", "wav2vec2"],
        help="Encoder family for frame-level embeddings (default: wavlm).",
    )

    parsed = parser.parse_args(list(argv))
    return ExtractEmbeddingsConfig(
        dataset=parsed.dataset,
        perturbation_root=parsed.perturbation_root,
        manifest_csv=parsed.manifest_csv,
        out_root=parsed.out_root,
        base_model_name=parsed.base_model_name,
        sv_model_name=parsed.sv_model_name,
        target_sr=parsed.target_sr,
        max_items=parsed.max_items,
        encoder_type=parsed.encoder_type,
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    manifest_path = run_extract_perturbation_embeddings(config)
    print(f"Saved embedding manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
