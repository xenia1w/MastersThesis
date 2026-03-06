from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


def frame_level_similarity_naive(a: torch.Tensor, b: torch.Tensor) -> float:
    min_len = min(a.shape[0], b.shape[0])
    a_trim = a[:min_len]
    b_trim = b[:min_len]
    return F.cosine_similarity(a_trim, b_trim, dim=1).mean().item()


def frame_level_similarity_topk(a: torch.Tensor, b: torch.Tensor, k: int = 200) -> float:
    """
    Compare frame sets without alignment by sampling frames and averaging
    max cosine similarity per frame (symmetric).
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    def sample(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= k:
            return x
        idx = torch.linspace(0, x.shape[0] - 1, steps=k).long()
        return x[idx]

    a_s = sample(a)
    b_s = sample(b)

    sim = a_s @ b_s.T  # [Ta, Tb]
    score_ab = sim.max(dim=1).values.mean()
    score_ba = sim.max(dim=0).values.mean()
    return 0.5 * (score_ab + score_ba).item()
