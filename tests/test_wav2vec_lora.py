from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.asr_adaptation.models.wav2vec_lora import (
    save_speaker_adapter,
    trainable_parameter_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_peft_model(trainable_params: int = 100, frozen_params: int = 900) -> MagicMock:
    """Build a mock PeftModel with controlled parameter counts."""
    model = MagicMock()

    trainable = [torch.nn.Parameter(torch.zeros(trainable_params), requires_grad=True)]
    frozen = [torch.nn.Parameter(torch.zeros(frozen_params), requires_grad=False)]

    model.parameters.return_value = trainable + frozen
    return model


# ---------------------------------------------------------------------------
# Tests: trainable_parameter_summary
# ---------------------------------------------------------------------------

def test_trainable_parameter_summary_counts(tmp_path: Path) -> None:
    model = _make_fake_peft_model(trainable_params=50, frozen_params=200)
    summary = trainable_parameter_summary(model)

    assert summary["trainable"] == 50
    assert summary["frozen"] == 200
    assert summary["total"] == 250


def test_trainable_parameter_summary_only_lora_trainable() -> None:
    model = _make_fake_peft_model(trainable_params=10, frozen_params=990)
    summary = trainable_parameter_summary(model)

    assert summary["trainable"] < summary["total"]
    assert summary["trainable"] + summary["frozen"] == summary["total"]


# ---------------------------------------------------------------------------
# Tests: save_speaker_adapter
# ---------------------------------------------------------------------------

def test_save_speaker_adapter_creates_directory(tmp_path: Path) -> None:
    model = MagicMock()
    result_path = save_speaker_adapter(model, "ABA", tmp_path)

    assert result_path == tmp_path / "ABA"
    assert result_path.is_dir()


def test_save_speaker_adapter_calls_save_pretrained(tmp_path: Path) -> None:
    model = MagicMock()
    save_speaker_adapter(model, "ASI", tmp_path)

    model.save_pretrained.assert_called_once_with(str(tmp_path / "ASI"))


def test_save_speaker_adapter_returns_correct_path(tmp_path: Path) -> None:
    model = MagicMock()
    path = save_speaker_adapter(model, "BWC", tmp_path)

    assert path == tmp_path / "BWC"


# ---------------------------------------------------------------------------
# Tests: build_lora_model (mocked to avoid downloading weights)
# ---------------------------------------------------------------------------

def test_build_lora_model_freezes_base_parameters() -> None:
    """All non-LoRA parameters must have requires_grad=False."""
    with patch("src.asr_adaptation.models.wav2vec_lora.Wav2Vec2ForCTC") as MockModel, \
         patch("src.asr_adaptation.models.wav2vec_lora.Wav2Vec2Processor"), \
         patch("src.asr_adaptation.models.wav2vec_lora.get_peft_model") as mock_peft:

        fake_base = MagicMock()
        MockModel.from_pretrained.return_value = fake_base

        lora_param = torch.nn.Parameter(torch.zeros(10), requires_grad=True)
        base_param = torch.nn.Parameter(torch.zeros(10), requires_grad=False)
        mock_peft.return_value.parameters.return_value = [lora_param, base_param]

        from src.asr_adaptation.models.wav2vec_lora import build_lora_model
        peft_model, _ = build_lora_model()

        summary = trainable_parameter_summary(peft_model)
        assert summary["trainable"] == 10
        assert summary["frozen"] == 10


def test_build_lora_model_uses_correct_target_modules() -> None:
    with patch("src.asr_adaptation.models.wav2vec_lora.Wav2Vec2ForCTC"), \
         patch("src.asr_adaptation.models.wav2vec_lora.Wav2Vec2Processor"), \
         patch("src.asr_adaptation.models.wav2vec_lora.get_peft_model"), \
         patch("src.asr_adaptation.models.wav2vec_lora.LoraConfig") as MockLoraConfig:

        from src.asr_adaptation.models.wav2vec_lora import build_lora_model
        build_lora_model()

        call_kwargs = MockLoraConfig.call_args.kwargs
        assert "q_proj" in call_kwargs["target_modules"]
        assert "v_proj" in call_kwargs["target_modules"]
