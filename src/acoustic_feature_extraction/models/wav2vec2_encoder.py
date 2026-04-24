from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.acoustic_feature_extraction.features.utterance_embedding import mean_std_pool


class Wav2Vec2ProfileExtractor:
    """Extract speaker profiles from a frozen Wav2Vec2Model.

    The profile is the mean+std pool of hidden states taken from a configurable
    transformer layer (default: last layer, ``profile_layer=-1``).  Output shape
    is ``[hidden_size * 2]`` — identical to the WavLM mean+std profile so the
    same ``speaker_projection`` layer in the ASR backbone can be reused.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        profile_layer: int = -1,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.profile_layer = profile_layer
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        cache_path = Path(cache_dir or "data/cache/huggingface")
        cache_path.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, cache_dir=str(cache_path)
        )
        self.model = Wav2Vec2Model.from_pretrained(model_name, cache_dir=str(cache_path))
        torch.nn.Module.to(self.model, self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_frames(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """Return frame-level hidden states of shape [T, hidden_size].

        Args:
            waveform: 1-D float32 tensor at ``sampling_rate`` Hz.
            sampling_rate: Sampling rate of the input waveform.

        Returns:
            Hidden states from ``self.profile_layer`` of the transformer,
            squeezed to [T, hidden_size] on CPU.
        """
        if waveform.dim() != 1:
            waveform = waveform.squeeze()

        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        # hidden_states is a tuple of (num_layers + 1) tensors, each [B, T, C]
        # index 0 is the CNN output; indices 1..N are transformer layer outputs
        hidden = outputs.hidden_states[self.profile_layer]  # [B, T, C]
        return hidden.squeeze(0).cpu()

    @torch.no_grad()
    def extract_profile(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """Return a mean+std speaker profile of shape [hidden_size * 2].

        Calls ``encode_frames`` then applies L2-normalised mean+std pooling,
        producing a 1536-dim vector for ``facebook/wav2vec2-base`` (hidden_size=768).
        """
        frames = self.encode_frames(waveform, sampling_rate)
        return mean_std_pool(frames, l2_normalize=True)
