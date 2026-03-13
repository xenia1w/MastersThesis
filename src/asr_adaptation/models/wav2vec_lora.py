from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, PeftModel, PeftMixedModel, TaskType, get_peft_model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# Default LoRA hyperparameters — small rank keeps adapter lightweight
_LORA_R = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.05
# Target the self-attention projections in every encoder layer
_LORA_TARGET_MODULES = ["q_proj", "v_proj"]

DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"


def build_lora_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
    lora_r: int = _LORA_R,
    lora_alpha: int = _LORA_ALPHA,
    lora_dropout: float = _LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> tuple[PeftModel | PeftMixedModel, Wav2Vec2Processor]:
    """
    Load wav2vec2-base-960h and wrap it with LoRA adapters.

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

    base_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    return peft_model, processor


def trainable_parameter_summary(model: PeftModel | PeftMixedModel) -> dict[str, int]:
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

    base_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    return peft_model, processor
