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
from src.models.l2arctic_minimal import (
    L2ArcticEmbedding,
    L2ArcticSample,
    SpeakerCentroid,
    SpeakerGroup,
)


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
    results: List[L2ArcticEmbedding],
) -> List[SpeakerGroup]:
    by_speaker: Dict[str, List[L2ArcticEmbedding]] = {}
    for embedding in results:
        by_speaker.setdefault(embedding.speaker_id, []).append(embedding)
    return [
        SpeakerGroup(speaker_id=speaker_id, embeddings=embeddings)
        for speaker_id, embeddings in sorted(by_speaker.items())
    ]


def compute_speaker_centroids(
    by_speaker: List[SpeakerGroup],
) -> List[SpeakerCentroid]:
    centroids: List[SpeakerCentroid] = []
    for group in by_speaker:
        embs = torch.stack([it.utt_emb for it in group.embeddings], dim=0)
        centroid = F.normalize(embs.mean(dim=0), dim=0)
        centroids.append(
            SpeakerCentroid(speaker_id=group.speaker_id, centroid=centroid)
        )
    return centroids


def print_speaker_centroid_similarities(
    centroids: List[SpeakerCentroid],
) -> None:
    logger.info("Speaker centroid cosine similarities (xvector):")
    speakers = sorted(centroids, key=lambda item: item.speaker_id)
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            s1, s2 = speakers[i], speakers[j]
            sim = cosine(s1.centroid, s2.centroid)
            logger.info("  {} vs {}: {:.4f}", s1.speaker_id, s2.speaker_id, sim)
    logger.info("")


def print_within_speaker_similarities(
    by_speaker: List[SpeakerGroup],
    centroids: List[SpeakerCentroid],
) -> None:
    logger.info("Within-speaker avg cosine to centroid (xvector):")
    centroid_lookup = {c.speaker_id: c.centroid for c in centroids}
    for group in by_speaker:
        sims = [
            cosine(it.utt_emb, centroid_lookup[group.speaker_id])
            for it in group.embeddings
        ]
        avg_sim = sum(sims) / len(sims)
        logger.info("  {}: {:.4f}", group.speaker_id, avg_sim)
    logger.info("")


def print_pairwise_similarities(
    results: List[L2ArcticEmbedding],
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
                f"{a.speaker_id}:{a.wav_name} "
                f"vs {b.speaker_id}:{b.wav_name}"
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
) -> List[L2ArcticEmbedding]:
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
