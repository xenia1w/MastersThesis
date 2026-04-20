from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, PeftMixedModel, TaskType, get_peft_model
from transformers import WavLMForCTC, Wav2Vec2Processor


# Default LoRA hyperparameters — small rank keeps adapter lightweight
_LORA_R = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.05
# Target the self-attention projections in every encoder layer
_LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# WavLM-base-plus fine-tuned on LibriSpeech 100h clean with a CTC head.
# Acoustic profile extraction uses microsoft/wavlm-base-plus (pre-trained),
# kept separate to preserve speaker identity information in the embeddings.
DEFAULT_MODEL_NAME = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"


def build_lora_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
    lora_r: int = _LORA_R,
    lora_alpha: int = _LORA_ALPHA,
    lora_dropout: float = _LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> tuple[PeftModel | PeftMixedModel, Wav2Vec2Processor]:
    """
    Load a WavLM-base-plus CTC model and wrap it with LoRA adapters.

    All base model parameters are frozen; only the LoRA weights are trainable.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Optional local cache directory for model weights.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability on LoRA layers.
        target_modules: Attention sub-modules to adapt. Defaults to q_proj and v_proj.

    Returns:
        (peft_model, processor) — the LoRA-wrapped model and its feature processor.
    """
    if target_modules is None:
        target_modules = _LORA_TARGET_MODULES

    base_model = WavLMForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.enable_adapter_layers()
    return peft_model, processor


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    """Return counts of trainable vs. total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


def save_speaker_adapter(model: PeftModel | PeftMixedModel, speaker_id: str, output_dir: str | Path) -> Path:
    """
    Save LoRA adapter weights for a single speaker.

    Args:
        model: Trained PeftModel.
        speaker_id: Used as the subdirectory name, e.g. "ABA".
        output_dir: Root directory; adapter saved to output_dir/speaker_id/.

    Returns:
        Path to the saved adapter directory.
    """
    save_path = Path(output_dir) / speaker_id
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    return save_path


class SpeakerConditionedLoraModel(nn.Module):
    """
    Wraps a PEFT LoRA model with additive speaker embedding injection.

    The speaker centroid (mean+std of WavLM-base-plus frames, 1536-dim) is
    projected to WavLM's hidden size (768) and added as a constant bias to
    every time step at the feature_projection output — before the transformer
    encoder.  Both the LoRA adapter weights and the speaker_projection layer
    are trainable.
    """

    _HIDDEN_SIZE = 768  # WavLM-base hidden dim

    def __init__(
        self,
        peft_model: PeftModel | PeftMixedModel,
        speaker_emb_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.model = peft_model
        self.speaker_projection = nn.Linear(speaker_emb_dim, self._HIDDEN_SIZE, bias=False)
        self._speaker_bias: torch.Tensor | None = None

        registered = False
        for name, module in peft_model.named_modules():
            if name.endswith("wavlm.feature_projection"):
                module.register_forward_hook(self._inject_bias)
                registered = True
                break
        if not registered:
            raise RuntimeError(
                "Could not locate wavlm.feature_projection in the PEFT model."
            )

    @property
    def config(self):
        return self.model.config

    def _inject_bias(self, module, input, output):  # noqa: A002
        if self._speaker_bias is not None:
            hidden_states, extract_features = output
            return hidden_states + self._speaker_bias, extract_features
        return output

    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embedding: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        # speaker_embedding: [D] → broadcast as [1, 1, 768]
        #                    [B, D] → broadcast as [B, 1, 768]
        bias = self.speaker_projection(speaker_embedding)
        self._speaker_bias = bias.unsqueeze(0).unsqueeze(0) if bias.dim() == 1 else bias.unsqueeze(1)
        try:
            return self.model(input_values=input_values, labels=labels, **kwargs)
        finally:
            self._speaker_bias = None


def build_conditioned_lora_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
    speaker_emb_dim: int = 1536,
    lora_r: int = _LORA_R,
    lora_alpha: int = _LORA_ALPHA,
    lora_dropout: float = _LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> tuple[SpeakerConditionedLoraModel, Wav2Vec2Processor]:
    """Load WavLM-base-plus CTC with LoRA adapters wrapped in speaker-conditioning."""
    peft_model, processor = build_lora_model(
        model_name=model_name,
        cache_dir=cache_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    return SpeakerConditionedLoraModel(peft_model, speaker_emb_dim=speaker_emb_dim), processor


def save_conditioned_speaker_adapter(
    model: SpeakerConditionedLoraModel,
    speaker_id: str,
    output_dir: str | Path,
) -> Path:
    """Save LoRA adapter weights and speaker_projection for one speaker."""
    save_path = Path(output_dir) / speaker_id
    save_path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(save_path))
    torch.save(model.speaker_projection.state_dict(), save_path / "speaker_projection.pt")
    return save_path


def load_speaker_adapter(
    speaker_id: str,
    checkpoint_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
) -> tuple[PeftModel | PeftMixedModel, Wav2Vec2Processor]:
    """
    Load a previously saved speaker-specific LoRA adapter.

    Args:
        speaker_id: Subdirectory name used when saving, e.g. "ABA".
        checkpoint_dir: Root directory containing speaker adapter subdirectories.
        model_name: Base model identifier (must match what was used for training).
        cache_dir: Optional local cache directory for base model weights.

    Returns:
        (peft_model, processor) with the speaker adapter loaded.
    """
    adapter_path = Path(checkpoint_dir) / speaker_id

    base_model = WavLMForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    return peft_model, processor
