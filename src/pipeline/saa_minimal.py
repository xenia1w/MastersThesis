from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from src.controllers.saa_minimal_controller import SAAMinimalController
from src.data.saa_utils import load_saa_samples
from src.metrics.similarity import (
    cosine,
    frame_level_similarity_naive,
    frame_level_similarity_topk,
)
from src.models.prosody import ProsodyEmbedding, SAASample


def prepare_sample_list(
    outer_zip: str,
    samples: Iterable[SAASample] | None,
) -> List[SAASample]:
    return list(samples) if samples is not None else load_saa_samples(outer_zip)


def group_by_language(
    results: List[ProsodyEmbedding],
) -> Dict[str, List[ProsodyEmbedding]]:
    by_language: Dict[str, List[ProsodyEmbedding]] = {}
    for embedding in results:
        key = (
            embedding.saa_metadata.native_language
            if embedding.saa_metadata and embedding.saa_metadata.native_language
            else "unknown"
        )
        by_language.setdefault(key, []).append(embedding)
    return by_language


def compute_language_centroids(
    by_language: Dict[str, List[ProsodyEmbedding]],
) -> Dict[str, torch.Tensor]:
    centroids: Dict[str, torch.Tensor] = {}
    for language, embeddings in by_language.items():
        embs = torch.stack([it.utt_emb for it in embeddings], dim=0)
        centroids[language] = F.normalize(embs.mean(dim=0), dim=0)
    return centroids


def print_language_centroid_similarities(
    centroids: Dict[str, torch.Tensor],
    max_pairs: int = 200,
) -> None:
    if len(centroids) * (len(centroids) - 1) // 2 > max_pairs:
        logger.info(
            "Skipping language centroid similarities ({} languages).",
            len(centroids),
        )
        logger.info("")
        return

    logger.info("Language centroid cosine similarities (xvector):")
    languages = sorted(centroids.keys())
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            l1, l2 = languages[i], languages[j]
            sim = cosine(centroids[l1], centroids[l2])
            logger.info("  {} vs {}: {:.4f}", l1, l2, sim)
    logger.info("")


def print_within_language_similarities(
    by_language: Dict[str, List[ProsodyEmbedding]],
    centroids: Dict[str, torch.Tensor],
) -> None:
    logger.info("Within-language avg cosine to centroid (xvector):")
    for language, embeddings in sorted(by_language.items()):
        sims = [cosine(it.utt_emb, centroids[language]) for it in embeddings]
        avg_sim = sum(sims) / len(sims)
        logger.info("  {}: {:.4f}", language, avg_sim)
    logger.info("")


def print_pairwise_similarities(
    results: List[ProsodyEmbedding],
    max_items: int = 25,
) -> None:
    if len(results) > max_items:
        logger.info(
            "Skipping pairwise comparisons ({} items > {}).",
            len(results),
            max_items,
        )
        logger.info("")
        return

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
                f"{a.saa_metadata.native_language if a.saa_metadata else 'unknown'}:{a.file_name} "
                f"vs {b.saa_metadata.native_language if b.saa_metadata else 'unknown'}:{b.file_name}"
            )
            logger.info(label)
            logger.info("  utterance-level cosine (xvector): {:.4f}", utt_sim)
            logger.info("  utterance-level cosine (mean+std): {:.4f}", utt_sim_meanstd)
            logger.info("  frame-level cosine (naive): {:.4f}", frame_sim_naive)
            logger.info("  frame-level cosine (topk):  {:.4f}", frame_sim_topk)
            logger.info("")


def run_saa_minimal(
    outer_zip: str = "data/raw/archive.zip",
    samples: Iterable[SAASample] | None = None,
    model_name: str = "microsoft/wavlm-base-plus-sv",
    save_root: str = "data/processed/saa_minimal_embeddings",
) -> List[ProsodyEmbedding]:
    sample_list = prepare_sample_list(outer_zip, samples)

    controller = SAAMinimalController(
        outer_zip=outer_zip,
        model_name=model_name,
        save_root=save_root,
    )
    results = []
    for sample in tqdm(sample_list, desc="Encoding SAA", unit="utt"):
        embedding = controller.encode_sample(sample)
        controller.save_embedding(embedding)
        results.append(embedding)

    logger.info("Saved embeddings to {}", controller.save_root_path)
    logger.info("")

    by_language = group_by_language(results)
    centroids = compute_language_centroids(by_language)
    print_language_centroid_similarities(centroids)
    print_within_language_similarities(by_language, centroids)
    print_pairwise_similarities(results)

    return results
