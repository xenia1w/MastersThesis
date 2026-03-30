from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding, dim=0)


def running_centroids(embeddings: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    centroids: List[torch.Tensor] = []
    cumulative = None
    count = 0
    for emb in embeddings:
        count += 1
        if cumulative is None:
            cumulative = emb.clone()
        else:
            cumulative = cumulative + emb
        centroid = cumulative / float(count)
        centroids.append(F.normalize(centroid, dim=0))
    return centroids


def select_k(
    centroids: List[torch.Tensor],
    ks: Iterable[int],
) -> Dict[int, torch.Tensor]:
    selected: Dict[int, torch.Tensor] = {}
    total = len(centroids)
    for k in ks:
        if k <= 0 or k > total:
            continue
        selected[int(k)] = centroids[k - 1]
    return selected


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.clamp(F.cosine_similarity(a, b, dim=0), -1.0, 1.0))


def cosine_to_full(
    centroids: Dict[int, torch.Tensor],
    full: torch.Tensor,
) -> Dict[int, float]:
    return {int(k): _cosine(emb, full) for k, emb in centroids.items()}


def cosine_consecutive(centroids: List[torch.Tensor]) -> Dict[int, float]:
    """Cosine similarity between centroid_k and centroid_{k-1} for k >= 2."""
    return {
        k: _cosine(centroids[k - 1], centroids[k - 2])
        for k in range(2, len(centroids) + 1)
    }


def stability_point_consecutive(
    cosines_consec: Dict[int, float],
    threshold: float = 0.999,
) -> int | None:
    """First k >= 2 where s_k'' >= threshold."""
    for k in sorted(cosines_consec.keys()):
        if cosines_consec[k] >= threshold:
            return k
    return None


def stability_point(
    cosines_all: List[float],
    threshold: float = 0.95,
    epsilon: float = 0.002,
    consecutive: int = 2,
) -> int | None:
    if not cosines_all:
        return None
    deltas = [0.0]
    for i in range(1, len(cosines_all)):
        deltas.append(cosines_all[i] - cosines_all[i - 1])

    for idx in range(len(cosines_all)):
        if cosines_all[idx] < threshold:
            continue
        if consecutive <= 0:
            return idx + 1
        ok = True
        for j in range(1, consecutive + 1):
            pos = idx + j
            if pos >= len(deltas):
                ok = False
                break
            if deltas[pos] >= epsilon:
                ok = False
                break
        if ok:
            return idx + 1
    return None
