from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from loguru import logger

from src.asr_adaptation.data.l2arctic_transcriptions import L2ArcticTranscriptSample
from src.acoustic_feature_extraction.features.utterance_embedding import mean_std_pool
from src.acoustic_feature_extraction.models.wav2vec2_encoder import Wav2Vec2ProfileExtractor

_WAV2VEC2_MODEL = "facebook/wav2vec2-base"


def compute_speaker_centroid_wav2vec2(
    train_samples: Sequence[L2ArcticTranscriptSample],
    device: torch.device,
    model_name: str = _WAV2VEC2_MODEL,
    profile_layer: int = -1,
    cache_dir: str | None = None,
) -> torch.Tensor:
    """Encode training utterances with a frozen Wav2Vec2Model and return the
    L2-normalised speaker centroid of shape [2 * hidden_size] (1536 for
    ``facebook/wav2vec2-base``).

    Mirrors :func:`compute_speaker_centroid` but uses
    :class:`Wav2Vec2ProfileExtractor` instead of ``WavLMBaseEncoder``, so both
    the profile extractor and the ASR backbone share the same model family.

    The encoder is loaded, used, then deleted before returning so GPU memory is
    free for the subsequent LoRA training step.

    Args:
        train_samples: Training utterances for the target speaker.
        device: Device for inference.
        model_name: Wav2Vec2 checkpoint to use.
        profile_layer: Transformer layer index whose hidden states are pooled.
            Negative indices count from the last layer (``-1`` = last).
        cache_dir: HuggingFace model cache directory.

    Returns:
        L2-normalised centroid tensor of shape [1536].
    """
    logger.info(
        "Computing Wav2Vec2 speaker centroid from {} training utterances "
        "(model: {}, layer: {}) ...",
        len(train_samples),
        model_name,
        profile_layer,
    )
    encoder = Wav2Vec2ProfileExtractor(
        model_name=model_name,
        profile_layer=profile_layer,
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
        raise ValueError("No valid utterances to compute Wav2Vec2 speaker centroid.")

    centroid = torch.stack(embs, dim=0).mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
    logger.info("Wav2Vec2 speaker centroid computed — shape: {}", tuple(centroid.shape))
    return centroid
