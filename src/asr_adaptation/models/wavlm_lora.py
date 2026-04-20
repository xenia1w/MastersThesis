from __future__ import annotations

import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from transformers import WavLMForCTC, Wav2Vec2Processor
from transformers.models.wavlm.modeling_wavlm import WavLMAttention


# Default LoRA hyperparameters — small rank keeps adapter lightweight
_LORA_R = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.05
# Target the same attention projections as the wav2vec2 baseline.
# WavLM's attention normally extracts raw .weight tensors and passes them to
# F.multi_head_attention_forward, bypassing any LoRA wrapper.  We fix this by
# patching torch_multi_head_self_attention to call the projections as modules
# (see _patch_wavlm_attention_for_lora below).
_LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# WavLM-base-plus fine-tuned on LibriSpeech 100h clean with a CTC head.
# Acoustic profile extraction uses microsoft/wavlm-base-plus (pre-trained),
# kept separate to preserve speaker identity information in the embeddings.
DEFAULT_MODEL_NAME = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"


# ---------------------------------------------------------------------------
# WavLM attention patch
# ---------------------------------------------------------------------------

def _lora_aware_mhsa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gated_position_bias: torch.Tensor,
    output_attentions: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Drop-in replacement for WavLMAttention.torch_multi_head_self_attention.

    The original method passes self.q_proj.weight / self.v_proj.weight as raw
    tensors to F.multi_head_attention_forward, which silently skips any LoRA
    wrapper around those modules.  This version calls self.q_proj(x) and
    self.v_proj(x) as proper nn.Module instances so the LoRA adapter is applied.
    """
    bsz, tgt_len, _ = hidden_states.size()

    # Project through the LoRA-wrapped modules
    q = self.q_proj(hidden_states)   # [bsz, tgt_len, embed_dim]
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # [bsz, tgt_len, embed_dim] → [bsz, num_heads, tgt_len, head_dim]
    def split_heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    # Scaled dot-product attention weights: [bsz, num_heads, tgt_len, tgt_len]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

    # gated_position_bias arrives as [bsz * num_heads, tgt_len, tgt_len]
    if gated_position_bias is not None:
        attn_weights = attn_weights + gated_position_bias.view(
            bsz, self.num_heads, tgt_len, tgt_len
        )

    # Key padding mask: True = padding position → mask to -inf
    if attention_mask is not None:
        attn_weights = attn_weights.masked_fill(
            attention_mask.ne(1).unsqueeze(1).unsqueeze(2), float("-inf")
        )

    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(q.dtype)
    attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

    # Weighted sum → [bsz, tgt_len, embed_dim]
    attn_output = torch.matmul(attn_probs, v)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights if output_attentions else None


def _patch_wavlm_attention_for_lora(model: nn.Module) -> None:
    """
    Replace torch_multi_head_self_attention on every WavLMAttention instance
    so that q_proj and v_proj are called as modules (enabling LoRA).
    """
    for module in model.modules():
        if isinstance(module, WavLMAttention):
            module.torch_multi_head_self_attention = types.MethodType(  # type: ignore[method-assign]
                _lora_aware_mhsa, module
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_lora_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
    lora_r: int = _LORA_R,
    lora_alpha: int = _LORA_ALPHA,
    lora_dropout: float = _LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> tuple[WavLMForCTC, Wav2Vec2Processor]:
    """
    Load a WavLM-base-plus CTC model and attach LoRA adapters to q_proj / v_proj.

    Uses the native transformers PeftAdapterMixin API (add_adapter / enable_adapters)
    rather than PEFT's get_peft_model, which conflicts with the built-in adapter
    support in transformers >=5.x.

    WavLM's attention normally bypasses q_proj / v_proj as modules, so
    torch_multi_head_self_attention is patched on every attention layer to call
    them properly (see _patch_wavlm_attention_for_lora).

    All base model parameters are frozen; only the LoRA weights are trainable.

    Returns:
        (model, processor) — WavLMForCTC with active LoRA adapter and its processor.
    """
    if target_modules is None:
        target_modules = _LORA_TARGET_MODULES

    model = WavLMForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model.add_adapter(lora_config, adapter_name="default")
    model.enable_adapters()
    _patch_wavlm_attention_for_lora(model)

    return model, processor


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    """Return counts of trainable vs. total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


def save_speaker_adapter(model: WavLMForCTC, speaker_id: str, output_dir: str | Path) -> Path:
    """
    Save LoRA adapter weights for a single speaker.

    Only the trainable parameters (LoRA A and B matrices) are saved.

    Args:
        model: Trained WavLMForCTC with active LoRA adapter.
        speaker_id: Used as the subdirectory name, e.g. "ABA".
        output_dir: Root directory; adapter saved to output_dir/speaker_id/.

    Returns:
        Path to the saved adapter directory.
    """
    save_path = Path(output_dir) / speaker_id
    save_path.mkdir(parents=True, exist_ok=True)
    adapter_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(adapter_state, save_path / "adapter_weights.pt")
    return save_path


def load_speaker_adapter(
    speaker_id: str,
    checkpoint_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
) -> tuple[WavLMForCTC, Wav2Vec2Processor]:
    """
    Load a previously saved speaker-specific LoRA adapter.

    Args:
        speaker_id: Subdirectory name used when saving, e.g. "ABA".
        checkpoint_dir: Root directory containing speaker adapter subdirectories.
        model_name: Base model identifier (must match what was used for training).
        cache_dir: Optional local cache directory for base model weights.

    Returns:
        (model, processor) with the speaker adapter loaded and enabled.
    """
    adapter_path = Path(checkpoint_dir) / speaker_id

    model = WavLMForCTC.from_pretrained(model_name, cache_dir=cache_dir)
    processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)

    lora_config = LoraConfig(
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
        target_modules=_LORA_TARGET_MODULES,
        bias="none",
    )
    model.add_adapter(lora_config, adapter_name="default")
    model.enable_adapters()
    _patch_wavlm_attention_for_lora(model)

    adapter_state = torch.load(
        adapter_path / "adapter_weights.pt", map_location="cpu", weights_only=True
    )
    current_state = model.state_dict()
    current_state.update(adapter_state)
    model.load_state_dict(current_state)

    return model, processor
