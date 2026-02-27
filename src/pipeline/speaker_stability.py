from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

import torch
from loguru import logger
from pydantic import BaseModel, Field

from src.data.audio_utils import load_l2arctic_wav, load_saa_mp3
from src.data.l2arctic_utils import list_l2arctic_samples
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
from src.models.prosody import L2ArcticSample, SAASample
from src.models.wavlm_encoder import WavLMEncoder

DatasetName = Literal["l2arctic", "saa"]
EmbeddingType = Literal["mean_std", "xvector"]


class RepresentationConfig(BaseModel):
    name: str
    model_name: str
    embedding_type: EmbeddingType


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


def _utterance_id(sample: L2ArcticSample | SAASample) -> str:
    if isinstance(sample, L2ArcticSample):
        return sample.wav_name
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


def order_utterances(
    samples: List[L2ArcticSample] | List[SAASample],
    ordering: str,
    seed: int,
) -> List[L2ArcticSample] | List[SAASample]:
    if ordering == "random":
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return shuffled
    return sorted(samples, key=lambda s: _numeric_sort_key(_utterance_id(s)))


def _load_cached_embedding(
    cache_root: Path,
    sample: L2ArcticSample | SAASample,
    representation: RepresentationConfig,
) -> torch.Tensor | None:
    if isinstance(sample, L2ArcticSample):
        stem = sample.wav_name.replace(".wav", "")
        cache_path = cache_root / sample.speaker_id / f"{stem}.pt"
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


def _load_audio(
    outer_zip: str,
    sample: L2ArcticSample | SAASample,
) -> tuple[torch.Tensor, int]:
    if isinstance(sample, L2ArcticSample):
        return load_l2arctic_wav(outer_zip, sample.speaker_id, sample.wav_name)
    return load_saa_mp3(outer_zip, sample.filename)


def _prepare_samples(config: SpeakerStabilityConfig, outer_zip: str):
    if config.dataset == "l2arctic":
        samples = list_l2arctic_samples(outer_zip)
    else:
        samples = load_saa_samples(outer_zip)

    by_speaker: Dict[str, List[L2ArcticSample | SAASample]] = {}
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
    encoders = {
        rep.name: WavLMEncoder(model_name=rep.model_name) for rep in representations
    }

    by_speaker, speakers = _prepare_samples(config, outer_zip)

    logger.info("Running speaker stability for {} speakers", len(speakers))

    for speaker_id in speakers:
        samples = order_utterances(
            by_speaker[speaker_id], config.ordering, config.random_seed
        )
        if config.max_utterances_per_speaker is not None:
            samples = samples[: config.max_utterances_per_speaker]

        ordered_ids = [_utterance_id(sample) for sample in samples]
        durations: List[float] = []

        per_rep_embeddings: Dict[str, List[torch.Tensor]] = {
            rep.name: [] for rep in representations
        }

        for sample in samples:
            waveform, sr = _load_audio(outer_zip, sample)
            durations.append(float(len(waveform)) / float(sr))

            for rep in representations:
                embedding = None
                if config.use_cached_utterances:
                    embedding = _load_cached_embedding(cache_root, sample, rep)

                if embedding is None:
                    encoder = encoders[rep.name]
                    if rep.embedding_type == "xvector":
                        embedding = encoder.encode_utterance(waveform, sr)
                    else:
                        frames = encoder.encode_frames(waveform, sr)
                        embedding = mean_std_pool(frames)

                per_rep_embeddings[rep.name].append(
                    normalize_embedding(embedding).cpu()
                )

        cumulative_seconds = []
        total = 0.0
        for dur in durations:
            total += dur
            cumulative_seconds.append(total)

        for rep in representations:
            rep_embeddings = per_rep_embeddings[rep.name]
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

            payload = {
                "dataset": config.dataset,
                "speaker_id": speaker_id,
                "representation": rep.name,
                "model_name": rep.model_name,
                "ordering": config.ordering,
                "random_seed": config.random_seed,
                "ks": list(config.ks),
                "ordered_utterance_ids": ordered_ids,
                "num_utterances": len(samples),
                "total_seconds": cumulative_seconds[-1] if cumulative_seconds else 0.0,
                "total_seconds_by_k": total_seconds_by_k,
                "embeddings_by_k": selected | {"full": full},
                "cosine_to_full": {str(k): v for k, v in cosines_by_k.items()},
                "stability_k": stable_k,
                "stability_seconds": stable_seconds,
                "stability_threshold": config.stability_threshold,
                "stability_epsilon": config.stability_epsilon,
                "stability_consecutive": config.stability_consecutive,
            }

            speaker_dir = save_root / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            out_path = speaker_dir / f"{rep.name}.pt"
            torch.save(payload, out_path)

            summary = {
                "speaker_id": speaker_id,
                "representation": rep.name,
                "model_name": rep.model_name,
                "ks": list(config.ks),
                "cosine_to_full": {str(k): v for k, v in cosines_by_k.items()},
                "total_seconds_by_k": total_seconds_by_k,
                "stability_k": stable_k,
                "stability_seconds": stable_seconds,
                "stability_threshold": config.stability_threshold,
                "stability_epsilon": config.stability_epsilon,
                "stability_consecutive": config.stability_consecutive,
            }
            summary_path = speaker_dir / f"{rep.name}.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

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

    parsed = parser.parse_args(list(argv))
    ks = [int(k.strip()) for k in parsed.ks.split(",") if k.strip()]
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
