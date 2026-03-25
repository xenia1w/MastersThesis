from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from loguru import logger

from src.asr_adaptation.data.l2arctic_transcriptions import L2ArcticTranscriptSample
from src.prosodic_feature_extraction.features.utterance_embedding import mean_std_pool
from src.prosodic_feature_extraction.models.wavlm_encoder import WavLMBaseEncoder

_WAVLM_MODEL = "microsoft/wavlm-base-plus"


def compute_speaker_centroid(
    train_samples: Sequence[L2ArcticTranscriptSample],
    device: torch.device,
    model_name: str = _WAVLM_MODEL,
    cache_dir: str | None = None,
) -> torch.Tensor:
    """
    Encode training utterances with WavLM mean+std pooling and return the
    L2-normalised speaker centroid of shape [2 * hidden_size] (1536 for wavlm-base-plus).

    The encoder is loaded, used, then deleted before returning so GPU memory is
    free for the subsequent LoRA training step.
    """
    logger.info(
        "Computing speaker centroid from {} training utterances (model: {}) ...",
        len(train_samples),
        model_name,
    )
    encoder = WavLMBaseEncoder(
        model_name=model_name,
        device=str(device),
        cache_dir=cache_dir,
    )

    embs: list[torch.Tensor] = []
    for sample in train_samples:
        if len(sample.waveform) < 400:
            continue
        frames = encoder.encode_frames(sample.waveform, sample.sampling_rate)
        embs.append(mean_std_pool(frames, l2_normalize=False))

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not embs:
        raise ValueError("No valid utterances to compute speaker centroid.")

    centroid = torch.stack(embs, dim=0).mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
    logger.info("Speaker centroid computed — shape: {}", tuple(centroid.shape))
    return centroid
