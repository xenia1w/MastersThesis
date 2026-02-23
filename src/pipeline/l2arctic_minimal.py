from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

from src.data.audio_utils import load_l2arctic_wav
from src.features.utterance_embedding import mean_std_pool
from src.metrics.similarity import (
    cosine,
    frame_level_similarity_naive,
    frame_level_similarity_topk,
)
from src.models.wavlm_encoder import WavLMEncoder

SampleDict = Dict[str, Union[torch.Tensor, str, int]]


def default_samples() -> List[Tuple[str, str]]:
    return [
        ("ABA", "arctic_a0001.wav"),
        ("ABA", "arctic_a0002.wav"),
        ("ABA", "arctic_a0003.wav"),
        ("ASI", "arctic_a0001.wav"),
        ("ASI", "arctic_a0002.wav"),
        ("ASI", "arctic_a0003.wav"),
        ("BWC", "arctic_a0001.wav"),
        ("BWC", "arctic_a0002.wav"),
        ("BWC", "arctic_a0003.wav"),
    ]


def prepare_sample_list(
    samples: Iterable[Tuple[str, str]] | None,
) -> List[Tuple[str, str]]:
    return list(samples) if samples is not None else default_samples()


def encode_sample(
    encoder: WavLMEncoder,
    outer_zip: str,
    speaker_id: str,
    wav_name: str,
) -> SampleDict:
    waveform, sr = load_l2arctic_wav(outer_zip, speaker_id, wav_name)
    frames = encoder.encode_frames(waveform, sr)
    utt_emb = encoder.encode_utterance(waveform, sr)
    utt_emb_meanstd = mean_std_pool(frames)

    return {
        "speaker_id": speaker_id,
        "wav_name": wav_name,
        "sampling_rate": sr,
        "frames": frames,
        "utt_emb": utt_emb,
        "utt_emb_meanstd": utt_emb_meanstd,
    }


def save_embedding(
    save_root_path: Path,
    encoder: WavLMEncoder,
    sample: SampleDict,
) -> Path:
    speaker_id = str(sample["speaker_id"])
    wav_name = str(sample["wav_name"])
    speaker_dir = save_root_path / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    out_path = speaker_dir / f"{wav_name.replace('.wav', '')}.pt"
    torch.save(
        {
            "speaker_id": speaker_id,
            "wav_name": wav_name,
            "sampling_rate": sample["sampling_rate"],
            "frame_representations": sample["frames"],
            "utterance_embedding": sample["utt_emb"],
            "utterance_embedding_meanstd": sample["utt_emb_meanstd"],
            "model_name": encoder.model_name,
        },
        out_path,
    )
    return out_path


def group_by_speaker(
    results: List[SampleDict],
) -> Dict[str, List[SampleDict]]:
    by_speaker: Dict[str, List[SampleDict]] = {}
    for r in results:
        speaker_id = str(r["speaker_id"])
        by_speaker.setdefault(speaker_id, []).append(r)
    return by_speaker


def compute_speaker_centroids(
    by_speaker: Dict[str, List[SampleDict]],
) -> Dict[str, torch.Tensor]:
    centroids: Dict[str, torch.Tensor] = {}
    for speaker_id, items in by_speaker.items():
        embs = torch.stack([it["utt_emb"] for it in items], dim=0)
        centroids[speaker_id] = F.normalize(embs.mean(dim=0), dim=0)
    return centroids


def print_speaker_centroid_similarities(centroids: Dict[str, torch.Tensor]) -> None:
    logger.info("Speaker centroid cosine similarities (xvector):")
    speakers = sorted(centroids.keys())
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            s1, s2 = speakers[i], speakers[j]
            sim = cosine(centroids[s1], centroids[s2])
            logger.info("  {} vs {}: {:.4f}", s1, s2, sim)
    logger.info("")


def print_within_speaker_similarities(
    by_speaker: Dict[str, List[SampleDict]],
    centroids: Dict[str, torch.Tensor],
) -> None:
    logger.info("Within-speaker avg cosine to centroid (xvector):")
    for speaker_id, items in by_speaker.items():
        sims = [cosine(it["utt_emb"], centroids[speaker_id]) for it in items]
        avg_sim = sum(sims) / len(sims)
        logger.info("  {}: {:.4f}", speaker_id, avg_sim)
    logger.info("")


def print_pairwise_similarities(
    results: List[SampleDict],
) -> None:
    logger.info("Pairwise comparisons (xvector + frame-level):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            utt_sim = cosine(a["utt_emb"], b["utt_emb"])
            utt_sim_meanstd = cosine(a["utt_emb_meanstd"], b["utt_emb_meanstd"])
            frame_sim_naive = frame_level_similarity_naive(a["frames"], b["frames"])
            frame_sim_topk = frame_level_similarity_topk(a["frames"], b["frames"])
            label = (
                f'{a["speaker_id"]}:{a["wav_name"]} '
                f'vs {b["speaker_id"]}:{b["wav_name"]}'
            )
            logger.info(label)
            logger.info("  utterance-level cosine (xvector): {:.4f}", utt_sim)
            logger.info("  utterance-level cosine (mean+std): {:.4f}", utt_sim_meanstd)
            logger.info("  frame-level cosine (naive): {:.4f}", frame_sim_naive)
            logger.info("  frame-level cosine (topk):  {:.4f}", frame_sim_topk)
            logger.info("")


def run_l2arctic_minimal(
    outer_zip: str = "data/raw/l2arctic_release_v5.0.zip",
    samples: Iterable[Tuple[str, str]] | None = None,
    model_name: str = "microsoft/wavlm-base-plus-sv",
    save_root: str = "data/processed/l2arctic_minimal_embeddings",
) -> List[SampleDict]:
    sample_list = prepare_sample_list(samples)

    encoder = WavLMEncoder(model_name=model_name)
    save_root_path = Path(save_root)
    save_root_path.mkdir(parents=True, exist_ok=True)

    results = []
    for speaker_id, wav_name in tqdm(
        sample_list, desc="Encoding L2ARCTIC", unit="utt"
    ):
        sample = encode_sample(encoder, outer_zip, speaker_id, wav_name)
        save_embedding(save_root_path, encoder, sample)
        results.append(sample)

    logger.info("Saved embeddings to {}", save_root_path)
    logger.info("")

    # Speaker centroids (xvector)
    by_speaker = group_by_speaker(results)
    centroids = compute_speaker_centroids(by_speaker)
    print_speaker_centroid_similarities(centroids)
    print_within_speaker_similarities(by_speaker, centroids)
    print_pairwise_similarities(results)

    return results
SampleDict = Dict[str, Union[torch.Tensor, str, int]]
