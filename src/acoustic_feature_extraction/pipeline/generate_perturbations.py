from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.acoustic_feature_extraction.data.audio_utils import load_l2arctic_wav, load_saa_mp3
from src.acoustic_feature_extraction.data.l2arctic_utils import list_l2arctic_samples
from src.acoustic_feature_extraction.data.saa_utils import load_saa_samples
from src.acoustic_feature_extraction.models.acoustic import L2ArcticSample, SAASample

DatasetName = Literal["l2arctic", "saa"]


class PerturbationConfig(BaseModel):
    dataset: DatasetName = Field(..., description="Dataset to run: l2arctic or saa.")
    outer_zip: str | None = Field(default=None, description="Path to dataset zip archive.")
    out_root: str = Field(
        default="data/processed/perturbations",
        description="Root directory for generated perturbation audio.",
    )
    max_speakers: int = Field(
        default=3, description="Maximum number of speakers for subset generation."
    )
    max_utterances_per_speaker: int = Field(
        default=3, description="Maximum utterances per speaker for subset generation."
    )
    variants: List[str] = Field(
        default_factory=lambda: [
            "rate_p10",
            "rate_m10",
            "pitch_p2st",
            "pitch_m2st",
            "pause_ins",
        ]
    )
    pause_duration_sec: float = Field(
        default=0.2, description="Duration of inserted silence for pause perturbation."
    )


class SelectedUtterance(BaseModel):
    dataset: DatasetName
    speaker_id: str
    utterance_id: str
    l2_sample: L2ArcticSample | None = None
    saa_sample: SAASample | None = None


class ManifestRow(BaseModel):
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


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_fields(model_cls: type[BaseModel]) -> List[str]:
    if hasattr(model_cls, "model_fields"):
        return list(model_cls.model_fields.keys())
    return list(model_cls.__fields__.keys())


def _resolve_outer_zip(config: PerturbationConfig) -> str:
    if config.outer_zip is not None:
        return config.outer_zip
    if config.dataset == "l2arctic":
        return "data/raw/l2arctic_release_v5.0.zip"
    return "data/raw/archive.zip"


def _numeric_sort_key(name: str) -> tuple[str, int]:
    digits = ""
    for ch in reversed(name):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    if digits:
        return (name.rstrip(digits), int(digits))
    return (name, -1)


def _select_l2arctic_subset(
    outer_zip: str,
    max_speakers: int,
    max_utterances_per_speaker: int,
) -> List[SelectedUtterance]:
    by_speaker: Dict[str, List[L2ArcticSample]] = defaultdict(list)
    for sample in list_l2arctic_samples(outer_zip):
        by_speaker[sample.speaker_id].append(sample)

    out: List[SelectedUtterance] = []
    for speaker_id in sorted(by_speaker.keys())[:max_speakers]:
        ordered = sorted(
            by_speaker[speaker_id],
            key=lambda s: _numeric_sort_key(s.wav_name),
        )
        for sample in ordered[:max_utterances_per_speaker]:
            out.append(
                SelectedUtterance(
                    dataset="l2arctic",
                    speaker_id=speaker_id,
                    utterance_id=sample.wav_name.replace(".wav", ""),
                    l2_sample=sample,
                )
            )
    return out


def _select_saa_subset(
    outer_zip: str,
    max_speakers: int,
    max_utterances_per_speaker: int,
) -> List[SelectedUtterance]:
    by_speaker: Dict[str, List[SAASample]] = defaultdict(list)
    for sample in load_saa_samples(outer_zip):
        by_speaker[sample.speaker_id].append(sample)

    out: List[SelectedUtterance] = []
    for speaker_id in sorted(by_speaker.keys())[:max_speakers]:
        ordered = sorted(by_speaker[speaker_id], key=lambda s: _numeric_sort_key(s.filename))
        for sample in ordered[:max_utterances_per_speaker]:
            out.append(
                SelectedUtterance(
                    dataset="saa",
                    speaker_id=speaker_id,
                    utterance_id=sample.filename,
                    saa_sample=sample,
                )
            )
    return out


def select_subset(config: PerturbationConfig, outer_zip: str) -> List[SelectedUtterance]:
    if config.dataset == "l2arctic":
        return _select_l2arctic_subset(
            outer_zip=outer_zip,
            max_speakers=config.max_speakers,
            max_utterances_per_speaker=config.max_utterances_per_speaker,
        )
    return _select_saa_subset(
        outer_zip=outer_zip,
        max_speakers=config.max_speakers,
        max_utterances_per_speaker=config.max_utterances_per_speaker,
    )


def _load_waveform(outer_zip: str, sample: SelectedUtterance) -> tuple[torch.Tensor, int]:
    if sample.dataset == "l2arctic":
        if sample.l2_sample is None:
            raise ValueError("Missing L2-ARCTIC sample payload.")
        return load_l2arctic_wav(
            outer_zip,
            sample.l2_sample.speaker_id,
            sample.l2_sample.wav_name,
        )
    if sample.saa_sample is None:
        raise ValueError("Missing SAA sample payload.")
    return load_saa_mp3(outer_zip, sample.saa_sample.filename)


def _to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)


def _clip_audio(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def _variant_rate(audio: np.ndarray, rate: float) -> np.ndarray:
    return _clip_audio(librosa.effects.time_stretch(audio, rate=rate))


def _variant_pitch(audio: np.ndarray, sr: int, steps: float) -> np.ndarray:
    return _clip_audio(librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps))


def _variant_pause_insert(audio: np.ndarray, sr: int, pause_duration_sec: float) -> np.ndarray:
    pause_samples = max(1, int(sr * pause_duration_sec))
    silence = np.zeros(pause_samples, dtype=np.float32)
    n = audio.shape[0]
    if n < 2 * pause_samples:
        split = n // 2
        return _clip_audio(np.concatenate([audio[:split], silence, audio[split:]], axis=0))

    split_a = int(0.35 * n)
    split_b = int(0.70 * n)
    return _clip_audio(
        np.concatenate(
            [audio[:split_a], silence, audio[split_a:split_b], silence, audio[split_b:]],
            axis=0,
        )
    )


def _variant_map(
    pause_duration_sec: float,
) -> Dict[str, tuple[Callable[[np.ndarray, int], np.ndarray], dict]]:
    return {
        "rate_p10": (lambda y, sr: _variant_rate(y, 1.10), {"rate": 1.10}),
        "rate_m10": (lambda y, sr: _variant_rate(y, 0.90), {"rate": 0.90}),
        "pitch_p2st": (lambda y, sr: _variant_pitch(y, sr, 2.0), {"semitones": 2.0}),
        "pitch_m2st": (lambda y, sr: _variant_pitch(y, sr, -2.0), {"semitones": -2.0}),
        "pause_ins": (
            lambda y, sr: _variant_pause_insert(y, sr, pause_duration_sec),
            {"pause_duration_sec": pause_duration_sec},
        ),
    }


def _build_output_path(
    out_root: Path,
    dataset: DatasetName,
    speaker_id: str,
    utterance_id: str,
    variant: str,
) -> Path:
    speaker_dir = out_root / dataset / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    return speaker_dir / f"{utterance_id}__{variant}.wav"


def _manifest_row(
    dataset: DatasetName,
    speaker_id: str,
    utterance_id: str,
    variant: str,
    source: str,
    output_path: Path,
    sampling_rate: int,
    num_samples: int,
    params: dict,
) -> ManifestRow:
    return ManifestRow(
        dataset=dataset,
        speaker_id=speaker_id,
        utterance_id=utterance_id,
        variant=variant,
        source=source,
        output_path=str(output_path),
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration_seconds=float(num_samples) / float(sampling_rate),
        params_json=json.dumps(params, sort_keys=True),
    )


def _write_manifest(out_root: Path, dataset: DatasetName, rows: Iterable[ManifestRow]) -> Path:
    out_path = out_root / dataset / "manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = _model_fields(ManifestRow)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_model_dump(row))
    return out_path


def run_generate_perturbations(config: PerturbationConfig) -> Path:
    outer_zip = _resolve_outer_zip(config)
    out_root = Path(config.out_root)
    selected = select_subset(config, outer_zip=outer_zip)
    if not selected:
        raise RuntimeError("No utterances selected for perturbation generation.")

    available = _variant_map(config.pause_duration_sec)
    unknown = [name for name in config.variants if name not in available]
    if unknown:
        raise ValueError(f"Unknown perturbation variants: {unknown}")

    manifest_rows: List[ManifestRow] = []
    for sample in tqdm(selected, desc="Generating perturbations", unit="utt"):
        waveform, sr = _load_waveform(outer_zip=outer_zip, sample=sample)
        audio = _to_numpy(waveform)

        orig_path = _build_output_path(
            out_root=out_root,
            dataset=sample.dataset,
            speaker_id=sample.speaker_id,
            utterance_id=sample.utterance_id,
            variant="orig",
        )
        sf.write(orig_path, _clip_audio(audio), sr)
        manifest_rows.append(
            _manifest_row(
                dataset=sample.dataset,
                speaker_id=sample.speaker_id,
                utterance_id=sample.utterance_id,
                variant="orig",
                source="original",
                output_path=orig_path,
                sampling_rate=sr,
                num_samples=int(audio.shape[0]),
                params={},
            )
        )

        for variant in config.variants:
            fn, params = available[variant]
            perturbed = fn(audio, sr)
            out_path = _build_output_path(
                out_root=out_root,
                dataset=sample.dataset,
                speaker_id=sample.speaker_id,
                utterance_id=sample.utterance_id,
                variant=variant,
            )
            sf.write(out_path, perturbed, sr)
            manifest_rows.append(
                _manifest_row(
                    dataset=sample.dataset,
                    speaker_id=sample.speaker_id,
                    utterance_id=sample.utterance_id,
                    variant=variant,
                    source="perturbed",
                    output_path=out_path,
                    sampling_rate=sr,
                    num_samples=int(perturbed.shape[0]),
                    params=params,
                )
            )

    return _write_manifest(out_root=out_root, dataset=config.dataset, rows=manifest_rows)


def parse_args(argv: Sequence[str]) -> PerturbationConfig:
    parser = argparse.ArgumentParser(
        description="Generate controlled acoustic perturbations for a subset of utterances."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["l2arctic", "saa"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--outer-zip",
        default=None,
        help="Path to dataset zip archive (overrides default).",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/perturbations",
        help="Root output directory for generated audio.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=3,
        help="Max number of speakers to process.",
    )
    parser.add_argument(
        "--max-utterances-per-speaker",
        type=int,
        default=3,
        help="Max utterances per speaker.",
    )
    parser.add_argument(
        "--variants",
        default="rate_p10,rate_m10,pitch_p2st,pitch_m2st,pause_ins",
        help="Comma-separated variant names.",
    )
    parser.add_argument(
        "--pause-duration-sec",
        type=float,
        default=0.2,
        help="Silence duration for pause insertion.",
    )

    parsed = parser.parse_args(list(argv))
    variants = [name.strip() for name in parsed.variants.split(",") if name.strip()]
    return PerturbationConfig(
        dataset=parsed.dataset,
        outer_zip=parsed.outer_zip,
        out_root=parsed.out_root,
        max_speakers=parsed.max_speakers,
        max_utterances_per_speaker=parsed.max_utterances_per_speaker,
        variants=variants,
        pause_duration_sec=parsed.pause_duration_sec,
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    manifest_path = run_generate_perturbations(config)
    print(f"Saved perturbation manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
