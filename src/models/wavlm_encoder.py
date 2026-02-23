from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


class WavLMEncoder:
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        cache_path = Path(cache_dir or "data/cache/huggingface")
        cache_path.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, cache_dir=str(cache_path)
        )
        self.model = WavLMForXVector.from_pretrained(
            model_name, cache_dir=str(cache_path)
        )
        torch.nn.Module.to(self.model, self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_frames(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Returns frame-level representations of shape [T, C].
        """
        if waveform.dim() != 1:
            waveform = waveform.squeeze()

        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.wavlm(**inputs)
        hidden = outputs.last_hidden_state  # [B, T, C]
        return hidden.squeeze(0).cpu()

    @torch.no_grad()
    def encode_utterance(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Returns speaker embedding of shape [C] from the x-vector head.
        """
        if waveform.dim() != 1:
            waveform = waveform.squeeze()

        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.embeddings.squeeze(0).cpu()
