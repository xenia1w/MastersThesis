from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

from src.controllers.l2arctic_minimal_controller import L2ArcticMinimalController
from src.metrics.similarity import (
    cosine,
    frame_level_similarity_naive,
    frame_level_similarity_topk,
)
from src.models.prosody import L2ArcticSample, ProsodyEmbedding


def default_samples() -> List[L2ArcticSample]:
    return [
        L2ArcticSample(speaker_id="ABA", wav_name="arctic_a0001.wav"),
        L2ArcticSample(speaker_id="ABA", wav_name="arctic_a0002.wav"),
        L2ArcticSample(speaker_id="ABA", wav_name="arctic_a0003.wav"),
        L2ArcticSample(speaker_id="ASI", wav_name="arctic_a0001.wav"),
        L2ArcticSample(speaker_id="ASI", wav_name="arctic_a0002.wav"),
        L2ArcticSample(speaker_id="ASI", wav_name="arctic_a0003.wav"),
        L2ArcticSample(speaker_id="BWC", wav_name="arctic_a0001.wav"),
        L2ArcticSample(speaker_id="BWC", wav_name="arctic_a0002.wav"),
        L2ArcticSample(speaker_id="BWC", wav_name="arctic_a0003.wav"),
    ]


def prepare_sample_list(
    samples: Iterable[L2ArcticSample] | None,
) -> List[L2ArcticSample]:
    return list(samples) if samples is not None else default_samples()


def group_by_speaker(
    results: List[ProsodyEmbedding],
) -> Dict[str, List[ProsodyEmbedding]]:
    by_speaker: Dict[str, List[ProsodyEmbedding]] = {}
    for embedding in results:
        by_speaker.setdefault(embedding.speaker_id, []).append(embedding)
    return by_speaker


def compute_speaker_centroids(
    by_speaker: Dict[str, List[ProsodyEmbedding]],
) -> Dict[str, torch.Tensor]:
    centroids: Dict[str, torch.Tensor] = {}
    for speaker_id, embeddings in by_speaker.items():
        embs = torch.stack([it.utt_emb for it in embeddings], dim=0)
        centroids[speaker_id] = F.normalize(embs.mean(dim=0), dim=0)
    return centroids


def print_speaker_centroid_similarities(
    centroids: Dict[str, torch.Tensor],
) -> None:
    logger.info("Speaker centroid cosine similarities (xvector):")
    speakers = sorted(centroids.keys())
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            s1, s2 = speakers[i], speakers[j]
            sim = cosine(centroids[s1], centroids[s2])
            logger.info("  {} vs {}: {:.4f}", s1, s2, sim)
    logger.info("")


def print_within_speaker_similarities(
    by_speaker: Dict[str, List[ProsodyEmbedding]],
    centroids: Dict[str, torch.Tensor],
) -> None:
    logger.info("Within-speaker avg cosine to centroid (xvector):")
    for speaker_id, embeddings in sorted(by_speaker.items()):
        sims = [cosine(it.utt_emb, centroids[speaker_id]) for it in embeddings]
        avg_sim = sum(sims) / len(sims)
        logger.info("  {}: {:.4f}", speaker_id, avg_sim)
    logger.info("")


def print_pairwise_similarities(
    results: List[ProsodyEmbedding],
) -> None:
    logger.info("Pairwise comparisons (xvector + frame-level):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            utt_sim = cosine(a.utt_emb, b.utt_emb)
            utt_sim_meanstd = cosine(a.utt_emb_meanstd, b.utt_emb_meanstd)
            frame_sim_naive = frame_level_similarity_naive(a.frames, b.frames)
            frame_sim_topk = frame_level_similarity_topk(a.frames, b.frames)
            label = (
                f"{a.speaker_id}:{a.file_name} "
                f"vs {b.speaker_id}:{b.file_name}"
            )
            logger.info(label)
            logger.info("  utterance-level cosine (xvector): {:.4f}", utt_sim)
            logger.info("  utterance-level cosine (mean+std): {:.4f}", utt_sim_meanstd)
            logger.info("  frame-level cosine (naive): {:.4f}", frame_sim_naive)
            logger.info("  frame-level cosine (topk):  {:.4f}", frame_sim_topk)
            logger.info("")


def run_l2arctic_minimal(
    outer_zip: str = "data/raw/l2arctic_release_v5.0.zip",
    samples: Iterable[L2ArcticSample] | None = None,
    model_name: str = "microsoft/wavlm-base-plus-sv",
    save_root: str = "data/processed/l2arctic_minimal_embeddings",
) -> List[ProsodyEmbedding]:
    sample_list = prepare_sample_list(samples)

    controller = L2ArcticMinimalController(
        outer_zip=outer_zip,
        model_name=model_name,
        save_root=save_root,
    )
    results = []
    for sample in tqdm(sample_list, desc="Encoding L2ARCTIC", unit="utt"):
        embedding = controller.encode_sample(sample)
        controller.save_embedding(embedding)
        results.append(embedding)

    logger.info("Saved embeddings to {}", controller.save_root_path)
    logger.info("")

    # Speaker centroids (xvector)
    by_speaker = group_by_speaker(results)
    centroids = compute_speaker_centroids(by_speaker)
    print_speaker_centroid_similarities(centroids)
    print_within_speaker_similarities(by_speaker, centroids)
    print_pairwise_similarities(results)

    return results
