from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, TypeVar, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

DatasetName = Literal["l2arctic", "saa"]


class AnalyzeSensitivityConfig(BaseModel):
    dataset: Literal["l2arctic", "saa", "all"] = Field(
        default="all",
        description="Dataset to analyze.",
    )
    embeddings_root: str = Field(
        default="data/processed/perturbation_embeddings",
        description="Root directory containing embedding payloads and manifests.",
    )
    out_root: str = Field(
        default="data/processed/perturbation_sensitivity",
        description="Output root for sensitivity CSV/JSON files.",
    )


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


class SensitivityDetailRow(BaseModel):
    dataset: DatasetName
    speaker_id: str
    utterance_id: str
    perturbation_type: str
    cosine_base_mean: float
    cosine_base_meanstd: float
    cosine_sv_xvector: float
    delta_xvector_minus_mean: float
    delta_xvector_minus_meanstd: float
    orig_embedding_path: str
    pert_embedding_path: str


class SensitivityAggregateRow(BaseModel):
    dataset: DatasetName
    perturbation_type: str
    n: int
    cosine_base_mean_avg: float
    cosine_base_meanstd_avg: float
    cosine_sv_xvector_avg: float
    delta_xvector_minus_mean_avg: float
    delta_xvector_minus_meanstd_avg: float


RowT = TypeVar("RowT", bound=BaseModel)


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_fields(model_cls: type[BaseModel]) -> List[str]:
    if hasattr(model_cls, "model_fields"):
        return list(model_cls.model_fields.keys())
    return list(model_cls.__fields__.keys())


def _load_manifest(csv_path: Path) -> List[EmbeddingManifestRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Embedding manifest not found: {csv_path}")

    rows: List[EmbeddingManifestRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            dataset_raw = (raw.get("dataset") or "").strip()
            if dataset_raw not in {"l2arctic", "saa"}:
                raise ValueError(f"Invalid dataset value in manifest: {dataset_raw}")
            dataset = cast(DatasetName, dataset_raw)
            rows.append(
                EmbeddingManifestRow(
                    dataset=dataset,
                    speaker_id=(raw.get("speaker_id") or "").strip(),
                    utterance_id=(raw.get("utterance_id") or "").strip(),
                    variant=(raw.get("variant") or "").strip(),
                    source=(raw.get("source") or "").strip(),
                    audio_path=(raw.get("audio_path") or "").strip(),
                    embedding_path=(raw.get("embedding_path") or "").strip(),
                    sampling_rate=int(raw.get("sampling_rate") or "0"),
                    num_samples=int(raw.get("num_samples") or "0"),
                    duration_seconds=float(raw.get("duration_seconds") or "0.0"),
                    mean_dim=int(raw.get("mean_dim") or "0"),
                    meanstd_dim=int(raw.get("meanstd_dim") or "0"),
                    xvector_dim=int(raw.get("xvector_dim") or "0"),
                    base_model_name=(raw.get("base_model_name") or "").strip(),
                    sv_model_name=(raw.get("sv_model_name") or "").strip(),
                )
            )
    return rows


def _load_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Embedding payload not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type at {path}: {type(payload)}")
    return payload


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0).item())


def _group_by_utterance(
    rows: Iterable[EmbeddingManifestRow],
) -> Dict[tuple[DatasetName, str, str], List[EmbeddingManifestRow]]:
    grouped: Dict[tuple[DatasetName, str, str], List[EmbeddingManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.speaker_id, row.utterance_id)].append(row)
    return grouped


def build_detail_rows(rows: List[EmbeddingManifestRow]) -> List[SensitivityDetailRow]:
    grouped = _group_by_utterance(rows)
    detail_rows: List[SensitivityDetailRow] = []
    for (dataset, speaker_id, utterance_id), items in grouped.items():
        orig = next((item for item in items if item.variant == "orig"), None)
        if orig is None:
            continue

        orig_payload = _load_payload(Path(orig.embedding_path))
        orig_mean = cast(torch.Tensor, orig_payload["embedding_mean"])
        orig_meanstd = cast(torch.Tensor, orig_payload["embedding_meanstd"])
        orig_xvector = cast(torch.Tensor, orig_payload["embedding_xvector"])

        for item in items:
            if item.variant == "orig":
                continue
            pert_payload = _load_payload(Path(item.embedding_path))
            pert_mean = cast(torch.Tensor, pert_payload["embedding_mean"])
            pert_meanstd = cast(torch.Tensor, pert_payload["embedding_meanstd"])
            pert_xvector = cast(torch.Tensor, pert_payload["embedding_xvector"])

            cos_mean = _cosine(orig_mean, pert_mean)
            cos_meanstd = _cosine(orig_meanstd, pert_meanstd)
            cos_xvector = _cosine(orig_xvector, pert_xvector)

            detail_rows.append(
                SensitivityDetailRow(
                    dataset=dataset,
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    perturbation_type=item.variant,
                    cosine_base_mean=cos_mean,
                    cosine_base_meanstd=cos_meanstd,
                    cosine_sv_xvector=cos_xvector,
                    delta_xvector_minus_mean=cos_xvector - cos_mean,
                    delta_xvector_minus_meanstd=cos_xvector - cos_meanstd,
                    orig_embedding_path=orig.embedding_path,
                    pert_embedding_path=item.embedding_path,
                )
            )
    return detail_rows


def build_aggregate_rows(rows: List[SensitivityDetailRow]) -> List[SensitivityAggregateRow]:
    grouped: Dict[tuple[DatasetName, str], List[SensitivityDetailRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.perturbation_type)].append(row)

    out: List[SensitivityAggregateRow] = []
    for (dataset, perturbation), group in sorted(grouped.items()):
        n = len(group)
        mean_avg = sum(row.cosine_base_mean for row in group) / float(n)
        meanstd_avg = sum(row.cosine_base_meanstd for row in group) / float(n)
        xvec_avg = sum(row.cosine_sv_xvector for row in group) / float(n)
        out.append(
            SensitivityAggregateRow(
                dataset=dataset,
                perturbation_type=perturbation,
                n=n,
                cosine_base_mean_avg=mean_avg,
                cosine_base_meanstd_avg=meanstd_avg,
                cosine_sv_xvector_avg=xvec_avg,
                delta_xvector_minus_mean_avg=xvec_avg - mean_avg,
                delta_xvector_minus_meanstd_avg=xvec_avg - meanstd_avg,
            )
        )
    return out


def _write_csv(path: Path, model_cls: type[RowT], rows: List[RowT]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _model_fields(model_cls)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_model_dump(row))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _analyze_dataset(
    dataset: DatasetName,
    embeddings_root: Path,
    out_root: Path,
) -> dict:
    manifest_path = embeddings_root / dataset / "manifest_embeddings.csv"
    manifest_rows = _load_manifest(manifest_path)
    detail_rows = build_detail_rows(manifest_rows)
    aggregate_rows = build_aggregate_rows(detail_rows)

    dataset_out = out_root / dataset
    _write_csv(dataset_out / "sensitivity_detail.csv", SensitivityDetailRow, detail_rows)
    _write_csv(
        dataset_out / "sensitivity_aggregate.csv",
        SensitivityAggregateRow,
        aggregate_rows,
    )
    summary_payload = {
        "dataset": dataset,
        "num_manifest_rows": len(manifest_rows),
        "num_comparisons": len(detail_rows),
        "aggregate": [_model_dump(row) for row in aggregate_rows],
    }
    _write_json(dataset_out / "sensitivity_summary.json", summary_payload)
    return {
        "dataset": dataset,
        "manifest_rows": len(manifest_rows),
        "comparisons": len(detail_rows),
        "out_dir": str(dataset_out),
    }


def run_analyze_sensitivity(config: AnalyzeSensitivityConfig) -> Path:
    embeddings_root = Path(config.embeddings_root)
    out_root = Path(config.out_root)
    datasets: List[DatasetName]
    if config.dataset == "all":
        datasets = ["l2arctic", "saa"]
    else:
        datasets = [config.dataset]

    run_reports = [
        _analyze_dataset(dataset, embeddings_root=embeddings_root, out_root=out_root)
        for dataset in datasets
    ]
    summary_path = out_root / "run_summary.json"
    _write_json(summary_path, {"datasets": run_reports})
    return summary_path


def parse_args(argv: Sequence[str]) -> AnalyzeSensitivityConfig:
    parser = argparse.ArgumentParser(
        description="Compute embedding sensitivity to perturbations via cosine similarity."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["l2arctic", "saa", "all"],
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--embeddings-root",
        default="data/processed/perturbation_embeddings",
        help="Root directory containing embedding payloads and manifests.",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/perturbation_sensitivity",
        help="Output root for sensitivity CSV/JSON files.",
    )
    parsed = parser.parse_args(list(argv))
    return AnalyzeSensitivityConfig(
        dataset=parsed.dataset,
        embeddings_root=parsed.embeddings_root,
        out_root=parsed.out_root,
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    summary_path = run_analyze_sensitivity(config)
    print(f"Saved sensitivity results: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
