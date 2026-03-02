from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple, TypeVar

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.data.audio_utils import load_l2arctic_wav, load_saa_mp3
from src.data.l2arctic_utils import list_l2arctic_samples
from src.data.saa_segmentation import segment_saa_recording
from src.data.saa_utils import load_saa_samples
from src.features.incremental_embeddings import (
    cosine_to_full,
    normalize_embedding,
    running_centroids,
    select_k,
    stability_point,
)
from src.features.utterance_embedding import mean_std_pool
from src.metrics.similarity import cosine
from src.models.prosody import L2ArcticSample, SAASample, SAASegmentSample
from src.models.wavlm_encoder import WavLMBaseEncoder, WavLMEncoder

DatasetName = Literal["l2arctic", "saa"]
EmbeddingType = Literal["mean_std", "xvector"]
SampleT = TypeVar("SampleT", L2ArcticSample, SAASample, SAASegmentSample)


class RepresentationConfig(BaseModel):
    name: str
    model_name: str
    embedding_type: EmbeddingType


class RepresentationRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RepresentationConfig
    xvector_encoder: WavLMEncoder | None = None
    base_encoder: WavLMBaseEncoder | None = None


class RepresentationEmbeddings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    embeddings: List[torch.Tensor]


class SpeakerStabilityPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    total_seconds_by_k: Dict[str, float]
    embeddings_by_k: Dict[int | str, torch.Tensor]
    cosine_to_full: Dict[str, float]
    stability_k: int | None
    stability_seconds: float | None
    stability_threshold: float
    stability_epsilon: float
    stability_consecutive: int


class SpeakerStabilitySummary(BaseModel):
    speaker_id: str
    representation: str
    model_name: str
    ks: List[int]
    cosine_to_full: Dict[str, float]
    total_seconds_by_k: Dict[str, float]
    stability_k: int | None
    stability_seconds: float | None
    stability_threshold: float
    stability_epsilon: float
    stability_consecutive: int


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
    saa_segmentation: SAASegmentationConfig | None = None


def _resolve_outer_zip(config: SpeakerStabilityConfig) -> str:
    if config.outer_zip is not None:
        return config.outer_zip
    if config.dataset == "l2arctic":
        return "data/raw/l2arctic_release_v5.0.zip"
    return "data/raw/archive.zip"


def _resolve_save_root(config: SpeakerStabilityConfig) -> str:
    if config.save_root is not None:
        return config.save_root
    return f"data/processed/{config.dataset}_speaker_stability"


def _resolve_cache_root(config: SpeakerStabilityConfig) -> str:
    if config.utterance_cache_root is not None:
        return config.utterance_cache_root
    if config.dataset == "l2arctic":
        return "data/processed/l2arctic_minimal_embeddings"
    return "data/processed/saa_minimal_embeddings"


def _default_representations() -> List[RepresentationConfig]:
    return [
        RepresentationConfig(
            name="wavlm_base_meanstd",
            model_name="microsoft/wavlm-base-plus",
            embedding_type="mean_std",
        ),
        RepresentationConfig(
            name="wavlm_sv_xvector",
            model_name="microsoft/wavlm-base-plus-sv",
            embedding_type="xvector",
        ),
    ]


def _resolve_saa_segmentation(
    config: SpeakerStabilityConfig,
) -> SAASegmentationConfig:
    if config.saa_segmentation is not None:
        return config.saa_segmentation
    return SAASegmentationConfig()


def _utterance_id(sample: L2ArcticSample | SAASample | SAASegmentSample) -> str:
    if isinstance(sample, L2ArcticSample):
        return sample.wav_name
    if isinstance(sample, SAASegmentSample):
        return sample.segment_id
    return sample.filename


def _numeric_sort_key(name: str) -> tuple:
    digits = ""
    for ch in reversed(name):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    if digits:
        return (name.rstrip(digits), int(digits))
    return (name, -1)


def _chronological_sort_key(
    sample: L2ArcticSample | SAASample | SAASegmentSample,
) -> Tuple[str, float, int]:
    if isinstance(sample, L2ArcticSample):
        prefix, number = _numeric_sort_key(sample.wav_name)
        return (str(prefix), float(number), 0)
    if isinstance(sample, SAASegmentSample):
        return (sample.filename, float(sample.start_sample), sample.end_sample)
    prefix, number = _numeric_sort_key(sample.filename)
    return (str(prefix), float(number), 0)


def order_utterances(
    samples: List[SampleT],
    ordering: str,
    seed: int,
) -> List[SampleT]:
    ordered = list(samples)
    if ordering == "random":
        rng = random.Random(seed)
        rng.shuffle(ordered)
        return ordered
    ordered.sort(key=_chronological_sort_key)
    return ordered


def _load_cached_embedding(
    cache_root: Path,
    sample: L2ArcticSample | SAASample | SAASegmentSample,
    representation: RepresentationConfig,
) -> torch.Tensor | None:
    if isinstance(sample, L2ArcticSample):
        stem = sample.wav_name.replace(".wav", "")
        cache_path = cache_root / sample.speaker_id / f"{stem}.pt"
    elif isinstance(sample, SAASegmentSample):
        cache_path = cache_root / sample.speaker_id / f"{sample.segment_id}.pt"
    else:
        cache_path = cache_root / sample.speaker_id / f"{sample.filename}.pt"

    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("model_name") != representation.model_name:
        return None

    if representation.embedding_type == "xvector":
        key = "utterance_embedding"
    else:
        key = "utterance_embedding_meanstd"

    if key not in payload:
        return None

    embedding = payload[key]
    if not isinstance(embedding, torch.Tensor):
        return None
    return embedding


def _load_saa_waveform_cached(
    outer_zip: str,
    filename: str,
    audio_cache: Dict[str, tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, int]:
    cached = audio_cache.get(filename)
    if cached is not None:
        return cached
    waveform, sr = load_saa_mp3(outer_zip, filename)
    audio_cache[filename] = (waveform, sr)
    return waveform, sr


def _slice_segment_audio(
    waveform: torch.Tensor,
    sample: SAASegmentSample,
) -> torch.Tensor:
    start = max(0, sample.start_sample)
    end = min(int(waveform.shape[0]), sample.end_sample)
    if end <= start:
        return waveform[:0]
    return waveform[start:end]


def _load_audio(
    outer_zip: str,
    sample: L2ArcticSample | SAASample | SAASegmentSample,
    audio_cache: Dict[str, tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, int]:
    if isinstance(sample, L2ArcticSample):
        return load_l2arctic_wav(outer_zip, sample.speaker_id, sample.wav_name)
    if isinstance(sample, SAASegmentSample):
        waveform, sr = _load_saa_waveform_cached(outer_zip, sample.filename, audio_cache)
        return _slice_segment_audio(waveform, sample), sr
    return _load_saa_waveform_cached(outer_zip, sample.filename, audio_cache)


def _prepare_saa_segmented_samples(
    config: SpeakerStabilityConfig,
    outer_zip: str,
) -> List[SAASegmentSample]:
    saa_segmentation = _resolve_saa_segmentation(config)
    samples = load_saa_samples(outer_zip)
    segmented: List[SAASegmentSample] = []
    audio_cache: Dict[str, tuple[torch.Tensor, int]] = {}
    for sample in samples:
        waveform, sr = _load_saa_waveform_cached(outer_zip, sample.filename, audio_cache)
        segmented.extend(
            segment_saa_recording(
                sample=sample,
                waveform=waveform,
                sampling_rate=sr,
                min_duration_sec=saa_segmentation.min_sec,
                max_duration_sec=saa_segmentation.max_sec,
                merge_gap_sec=saa_segmentation.merge_gap_sec,
                top_db=saa_segmentation.top_db,
            )
        )
    return segmented


def _prepare_samples(config: SpeakerStabilityConfig, outer_zip: str):
    if config.dataset == "l2arctic":
        samples = list_l2arctic_samples(outer_zip)
    else:
        samples = _prepare_saa_segmented_samples(config, outer_zip)

    by_speaker: Dict[str, List[L2ArcticSample | SAASample | SAASegmentSample]] = {}
    for sample in samples:
        by_speaker.setdefault(sample.speaker_id, []).append(sample)

    speakers = sorted(by_speaker.keys())
    if config.max_speakers is not None:
        speakers = speakers[: config.max_speakers]
    return by_speaker, speakers


def run_speaker_stability(config: SpeakerStabilityConfig) -> None:
    outer_zip = _resolve_outer_zip(config)
    save_root = Path(_resolve_save_root(config))
    save_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(_resolve_cache_root(config))

    representations = (
        config.representations if config.representations else _default_representations()
    )
    runtimes: List[RepresentationRuntime] = []
    for rep in representations:
        if rep.embedding_type == "xvector":
            runtimes.append(
                RepresentationRuntime(
                    config=rep,
                    xvector_encoder=WavLMEncoder(model_name=rep.model_name),
                )
            )
        else:
            runtimes.append(
                RepresentationRuntime(
                    config=rep,
                    base_encoder=WavLMBaseEncoder(model_name=rep.model_name),
                )
            )

    by_speaker, speakers = _prepare_samples(config, outer_zip)

    logger.info("Running speaker stability for {} speakers", len(speakers))

    for speaker_id in speakers:
        audio_cache: Dict[str, tuple[torch.Tensor, int]] = {}
        samples = order_utterances(
            by_speaker[speaker_id], config.ordering, config.random_seed
        )
        if config.max_utterances_per_speaker is not None:
            samples = samples[: config.max_utterances_per_speaker]

        ordered_ids = [_utterance_id(sample) for sample in samples]
        durations: List[float] = []

        per_rep_embeddings: List[RepresentationEmbeddings] = [
            RepresentationEmbeddings(name=rep.config.name, embeddings=[])
            for rep in runtimes
        ]

        for sample in samples:
            waveform, sr = _load_audio(outer_zip, sample, audio_cache)
            durations.append(float(len(waveform)) / float(sr))

            for rep in runtimes:
                embedding = None
                if config.use_cached_utterances:
                    embedding = _load_cached_embedding(cache_root, sample, rep.config)

                if embedding is None:
                    if rep.config.embedding_type == "xvector":
                        if rep.xvector_encoder is None:
                            raise RuntimeError(
                                f"Missing xvector encoder for {rep.config.name}"
                            )
                        embedding = rep.xvector_encoder.encode_utterance(waveform, sr)
                    else:
                        if rep.base_encoder is None:
                            raise RuntimeError(
                                f"Missing base encoder for {rep.config.name}"
                            )
                        frames = rep.base_encoder.encode_frames(waveform, sr)
                        embedding = mean_std_pool(frames)

                for entry in per_rep_embeddings:
                    if entry.name == rep.config.name:
                        entry.embeddings.append(normalize_embedding(embedding).cpu())
                        break

        cumulative_seconds = []
        total = 0.0
        for dur in durations:
            total += dur
            cumulative_seconds.append(total)

        for rep in runtimes:
            rep_embeddings = next(
                entry.embeddings
                for entry in per_rep_embeddings
                if entry.name == rep.config.name
            )
            centroids = running_centroids(rep_embeddings)
            if not centroids:
                continue
            full = centroids[-1]

            selected = select_k(centroids, config.ks)
            cosines_by_k = cosine_to_full(selected, full)
            cosines_all = [cosine(c, full) for c in centroids]

            stable_k = stability_point(
                cosines_all,
                threshold=config.stability_threshold,
                epsilon=config.stability_epsilon,
                consecutive=config.stability_consecutive,
            )
            stable_seconds = (
                cumulative_seconds[stable_k - 1] if stable_k is not None else None
            )

            total_seconds_by_k = {
                str(k): cumulative_seconds[k - 1]
                for k in selected.keys()
                if k - 1 < len(cumulative_seconds)
            }

            payload = SpeakerStabilityPayload(
                dataset=config.dataset,
                speaker_id=speaker_id,
                representation=rep.config.name,
                model_name=rep.config.model_name,
                ordering=config.ordering,
                random_seed=config.random_seed,
                ks=list(config.ks),
                ordered_utterance_ids=ordered_ids,
                num_utterances=len(samples),
                total_seconds=cumulative_seconds[-1] if cumulative_seconds else 0.0,
                total_seconds_by_k=total_seconds_by_k,
                embeddings_by_k=selected | {"full": full},
                cosine_to_full={str(k): v for k, v in cosines_by_k.items()},
                stability_k=stable_k,
                stability_seconds=stable_seconds,
                stability_threshold=config.stability_threshold,
                stability_epsilon=config.stability_epsilon,
                stability_consecutive=config.stability_consecutive,
            )

            speaker_dir = save_root / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            out_path = speaker_dir / f"{rep.config.name}.pt"
            torch.save(payload.model_dump(), out_path)

            summary = SpeakerStabilitySummary(
                speaker_id=speaker_id,
                representation=rep.config.name,
                model_name=rep.config.model_name,
                ks=list(config.ks),
                cosine_to_full={str(k): v for k, v in cosines_by_k.items()},
                total_seconds_by_k=total_seconds_by_k,
                stability_k=stable_k,
                stability_seconds=stable_seconds,
                stability_threshold=config.stability_threshold,
                stability_epsilon=config.stability_epsilon,
                stability_consecutive=config.stability_consecutive,
            )
            summary_path = speaker_dir / f"{rep.config.name}.json"
            summary_path.write_text(
                json.dumps(summary.model_dump(), indent=2), encoding="utf-8"
            )

        logger.info("Saved stability outputs for speaker {}", speaker_id)


def parse_args(argv: Sequence[str]) -> SpeakerStabilityConfig:
    parser = argparse.ArgumentParser(description="Run speaker stability pipeline.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["l2arctic", "saa"],
        help="Dataset pipeline to run.",
    )
    parser.add_argument(
        "--outer-zip",
        default=None,
        help="Path to dataset zip archive (overrides default).",
    )
    parser.add_argument(
        "--save-root",
        default=None,
        help="Directory for saving stability outputs.",
    )
    parser.add_argument(
        "--ordering",
        default="chronological",
        choices=["chronological", "random"],
        help="Utterance ordering strategy.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Random seed for ordering.",
    )
    parser.add_argument(
        "--ks",
        default="1,2,3,5,10",
        help="Comma-separated list of k values.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional limit for number of speakers.",
    )
    parser.add_argument(
        "--max-utterances-per-speaker",
        type=int,
        default=None,
        help="Optional limit for number of utterances per speaker.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reuse of cached utterance embeddings.",
    )
    parser.add_argument(
        "--utterance-cache-root",
        default=None,
        help="Root directory for cached utterance embeddings.",
    )
    # Stability criteria can be tuned later from CLI without changing code.
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.95,
        help="Cosine threshold for stability point.",
    )
    parser.add_argument(
        "--stability-epsilon",
        type=float,
        default=0.002,
        help="Max delta allowed for plateau detection.",
    )
    parser.add_argument(
        "--stability-consecutive",
        type=int,
        default=2,
        help="Number of consecutive plateau steps required.",
    )
    parser.add_argument(
        "--saa-segment-min-sec",
        type=float,
        default=None,
        help="Minimum voiced segment duration for SAA segmentation.",
    )
    parser.add_argument(
        "--saa-segment-max-sec",
        type=float,
        default=None,
        help="Maximum voiced segment duration for SAA segmentation.",
    )
    parser.add_argument(
        "--saa-merge-gap-sec",
        type=float,
        default=None,
        help="Merge adjacent voiced regions if silence gap is below this value.",
    )
    parser.add_argument(
        "--saa-segment-top-db",
        type=int,
        default=None,
        help="Silence threshold (dB) used by librosa.effects.split for SAA.",
    )

    parsed = parser.parse_args(list(argv))
    ks = [int(k.strip()) for k in parsed.ks.split(",") if k.strip()]
    saa_segmentation: SAASegmentationConfig | None = None
    if (
        parsed.saa_segment_min_sec is not None
        or parsed.saa_segment_max_sec is not None
        or parsed.saa_merge_gap_sec is not None
        or parsed.saa_segment_top_db is not None
    ):
        saa_segmentation = SAASegmentationConfig(
            min_sec=(
                parsed.saa_segment_min_sec
                if parsed.saa_segment_min_sec is not None
                else 2.0
            ),
            max_sec=(
                parsed.saa_segment_max_sec
                if parsed.saa_segment_max_sec is not None
                else 8.0
            ),
            merge_gap_sec=(
                parsed.saa_merge_gap_sec
                if parsed.saa_merge_gap_sec is not None
                else 0.25
            ),
            top_db=(
                parsed.saa_segment_top_db if parsed.saa_segment_top_db is not None else 30
            ),
        )

    config = SpeakerStabilityConfig(
        dataset=parsed.dataset,
        outer_zip=parsed.outer_zip,
        save_root=parsed.save_root,
        ordering=parsed.ordering,
        random_seed=parsed.random_seed,
        ks=ks,
        max_speakers=parsed.max_speakers,
        max_utterances_per_speaker=parsed.max_utterances_per_speaker,
        use_cached_utterances=not parsed.no_cache,
        utterance_cache_root=parsed.utterance_cache_root,
        stability_threshold=parsed.stability_threshold,
        stability_epsilon=parsed.stability_epsilon,
        stability_consecutive=parsed.stability_consecutive,
        saa_segmentation=saa_segmentation,
    )
    if not config.representations:
        config.representations = _default_representations()
    return config


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    run_speaker_stability(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
