from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel, PeftMixedModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.asr_adaptation.models.wav2vec_lora import (
    DEFAULT_MODEL_NAME,
    _LORA_ALPHA,
    _LORA_DROPOUT,
    _LORA_R,
    _LORA_TARGET_MODULES,
    build_lora_model,
    trainable_parameter_summary,
)

__all__ = [
    "FiLMConditionedLoraModel",
    "build_film_lora_model",
    "load_film_lora_model",
    "save_film_speaker_adapter",
    "trainable_parameter_summary",
]

_N_LAYERS = 12
_HIDDEN_SIZE = 768
_MLP_HIDDEN = 128


class FiLMConditionedLoraModel(nn.Module):
    """
    Wraps a PEFT LoRA model with FiLM (Feature-wise Linear Modulation) conditioning.

    A two-layer MLP maps the 1536-dim speaker centroid to (γ_i, β_i) pairs for
    each of the 12 encoder layers.  At each layer the modulation is applied after
    final_layer_norm:  h = (1 + γ_i) * h + β_i

    The last MLP linear is zero-initialized so γ=0, β=0 at the start of training,
    preserving the model's pre-trained behaviour until the MLP learns to deviate.
    LoRA adapters and the film_mlp are both trainable.
    """

    def __init__(
        self,
        peft_model: PeftModel | PeftMixedModel,
        speaker_emb_dim: int = 1536,
        n_layers: int = _N_LAYERS,
        hidden_size: int = _HIDDEN_SIZE,
        mlp_hidden: int = _MLP_HIDDEN,
    ) -> None:
        super().__init__()
        self.model = peft_model
        self._n_layers = n_layers
        self._hidden_size = hidden_size

        film_out = nn.Linear(mlp_hidden, n_layers * hidden_size * 2)
        # Zero-init the output layer → identity transform at the start of training
        nn.init.zeros_(film_out.weight)
        nn.init.zeros_(film_out.bias)
        self.film_mlp = nn.Sequential(
            nn.Linear(speaker_emb_dim, mlp_hidden),
            nn.ReLU(),
            film_out,
        )

        self._film_params: torch.Tensor | None = None  # [N, 2, H] during forward

        hooks_registered = 0
        for name, module in peft_model.named_modules():
            for i in range(n_layers):
                if name.endswith(f"wav2vec2.encoder.layers.{i}.final_layer_norm"):
                    module.register_forward_hook(self._make_hook(i))
                    hooks_registered += 1
                    break

        if hooks_registered != n_layers:
            raise RuntimeError(
                f"Expected {n_layers} FiLM hooks but only registered {hooks_registered}. "
                "Check that the PEFT model exposes wav2vec2.encoder.layers.*.final_layer_norm."
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):  # noqa: A002
            if self._film_params is not None:
                gamma = self._film_params[layer_idx, 0]  # [H] — broadcasts to [B, T, H]
                beta = self._film_params[layer_idx, 1]   # [H]
                return (1 + gamma) * output + beta
            return output
        return hook

    @property
    def config(self):
        return self.model.config

    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        # speaker_embedding: [D] for a single speaker centroid (training / inference)
        if speaker_embedding is not None:
            params = self.film_mlp(speaker_embedding)  # [N*H*2]
            self._film_params = params.view(self._n_layers, 2, self._hidden_size)
        else:
            self._film_params = None
        try:
            return self.model(input_values=input_values, labels=labels, **kwargs)
        finally:
            self._film_params = None


def build_film_lora_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
    speaker_emb_dim: int = 1536,
    lora_r: int = _LORA_R,
    lora_alpha: int = _LORA_ALPHA,
    lora_dropout: float = _LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> tuple[FiLMConditionedLoraModel, Wav2Vec2Processor]:
    """Load wav2vec2 with LoRA adapters wrapped in FiLM speaker conditioning."""
    peft_model, processor = build_lora_model(
        model_name=model_name,
        cache_dir=cache_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    return FiLMConditionedLoraModel(peft_model, speaker_emb_dim=speaker_emb_dim), processor


def load_film_lora_model(
    speaker_id: str,
    checkpoint_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
) -> tuple[FiLMConditionedLoraModel, Wav2Vec2Processor]:
    """Load a previously saved FiLM-conditioned LoRA adapter for inference.

    Args:
        speaker_id: Subdirectory name used when saving, e.g. "ABA".
        checkpoint_dir: Root directory containing per-speaker adapter subdirectories.
        model_name: Base model identifier (must match training).
        cache_dir: Optional local cache directory for base model weights.

    Returns:
        (film_model, processor) ready for inference — no training state.
    """
    adapter_dir = Path(checkpoint_dir) / speaker_id
    base_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    film_model = FiLMConditionedLoraModel(peft_model)
    film_state = torch.load(adapter_dir / "film_mlp.pt", map_location="cpu", weights_only=True)
    film_model.film_mlp.load_state_dict(film_state)
    return film_model, processor


def save_film_speaker_adapter(
    model: FiLMConditionedLoraModel,
    speaker_id: str,
    output_dir: str | Path,
) -> Path:
    """Save LoRA adapter weights and film_mlp for one speaker."""
    save_path = Path(output_dir) / speaker_id
    save_path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(save_path))
    torch.save(model.film_mlp.state_dict(), save_path / "film_mlp.pt")
    return save_path
