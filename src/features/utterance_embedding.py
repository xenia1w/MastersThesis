from __future__ import annotations

import torch


def mean_pool(frames: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool frame-level representations [T, C] into utterance embedding [C].
    """
    if frames.dim() != 2:
        raise ValueError(f"Expected [T, C] frames, got shape {tuple(frames.shape)}")
    return frames.mean(dim=0)


def mean_std_pool(frames: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
    """
    Mean+std pooling into a single utterance embedding [2C].
    This tends to capture more speaker characteristics than mean-only.
    """
    if frames.dim() != 2:
        raise ValueError(f"Expected [T, C] frames, got shape {tuple(frames.shape)}")
    mean = frames.mean(dim=0)
    std = frames.std(dim=0, unbiased=False)
    emb = torch.cat([mean, std], dim=0)
    if l2_normalize:
        emb = torch.nn.functional.normalize(emb, dim=0)
    return emb
