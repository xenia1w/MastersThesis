from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from src.asr_adaptation.data.l2arctic_transcriptions import L2ArcticTranscriptSample
from src.asr_adaptation.pipeline.lora_train import (
    AdaptationRow,
    _get_hypotheses,
    _save_csv,
    _split_samples,
    _train_lora,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(utterance_id: str, transcript: str = "hello world") -> L2ArcticTranscriptSample:
    return L2ArcticTranscriptSample(
        speaker_id="ABA",
        utterance_id=utterance_id,
        waveform=torch.zeros(16000),
        sampling_rate=16000,
        transcript=transcript,
    )


def _make_samples(n: int) -> list[L2ArcticTranscriptSample]:
    return [_make_sample(f"arctic_a{i:04d}") for i in range(n)]


# ---------------------------------------------------------------------------
# Tests: _split_samples
# ---------------------------------------------------------------------------

def test_split_eval_is_always_last_n() -> None:
    samples = _make_samples(20)
    train, eval_ = _split_samples(samples, n_eval=5, n_train=None, seed=0)
    assert [s.utterance_id for s in eval_] == [s.utterance_id for s in samples[-5:]]


def test_split_n_train_limits_training_set() -> None:
    samples = _make_samples(20)
    train, eval_ = _split_samples(samples, n_eval=5, n_train=3, seed=0)
    assert len(train) == 3
    assert len(eval_) == 5


def test_split_different_seeds_give_different_subsets() -> None:
    samples = _make_samples(50)
    train0, _ = _split_samples(samples, n_eval=10, n_train=5, seed=0)
    train1, _ = _split_samples(samples, n_eval=10, n_train=5, seed=1)
    # Different seeds should (very likely) produce different subsets
    ids0 = [s.utterance_id for s in train0]
    ids1 = [s.utterance_id for s in train1]
    assert ids0 != ids1


def test_split_n_train_none_uses_all_pool() -> None:
    samples = _make_samples(20)
    train, eval_ = _split_samples(samples, n_eval=5, n_train=None, seed=0)
    assert len(train) == 15
    assert len(eval_) == 5


# ---------------------------------------------------------------------------
# Tests: _train_lora
# ---------------------------------------------------------------------------

def test_train_lora_calls_optimizer_step() -> None:
    model = MagicMock()
    output = MagicMock()
    output.loss = torch.tensor(1.0, requires_grad=True)
    model.return_value = output
    model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]

    processor = MagicMock()
    processor.return_value.input_values = torch.zeros(1, 100)
    processor.tokenizer.return_value.input_ids = torch.zeros(1, 5, dtype=torch.long)

    samples = _make_samples(4)

    with patch("src.asr_adaptation.pipeline.lora_train.torch.optim.AdamW") as MockAdam:
        mock_opt = MagicMock()
        MockAdam.return_value = mock_opt
        _train_lora(model, samples, processor, torch.device("cpu"), n_epochs=1, grad_accum_steps=4)

    mock_opt.step.assert_called()


# ---------------------------------------------------------------------------
# Tests: _save_csv
# ---------------------------------------------------------------------------

def test_save_csv_creates_file(tmp_path: Path) -> None:
    rows = [
        AdaptationRow(
            speaker_id="ABA",
            utterance_id="arctic_a0500",
            n_train=500,
            reference="hello world",
            hypothesis_baseline="hello there",
            hypothesis_adapted="hello world",
            wer_baseline=0.5,
            wer_adapted=0.0,
        )
    ]
    path = tmp_path / "results" / "ABA.csv"
    _save_csv(rows, path)
    assert path.exists()


def test_save_csv_wer_delta_is_correct(tmp_path: Path) -> None:
    rows = [
        AdaptationRow(
            speaker_id="ABA",
            utterance_id="arctic_a0500",
            n_train=500,
            reference="hello world",
            hypothesis_baseline="hello there",
            hypothesis_adapted="hello world",
            wer_baseline=0.5,
            wer_adapted=0.0,
        )
    ]
    path = tmp_path / "ABA.csv"
    _save_csv(rows, path)

    with open(path, newline="") as f:
        row = list(csv.DictReader(f))[0]

    assert float(row["wer_delta"]) == pytest.approx(-0.5)
    assert float(row["wer_baseline"]) == pytest.approx(0.5)
    assert float(row["wer_adapted"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: _get_hypotheses
# ---------------------------------------------------------------------------

def test_get_hypotheses_returns_one_per_sample() -> None:
    model = MagicMock()
    processor = MagicMock()
    samples = _make_samples(3)

    with patch("src.asr_adaptation.pipeline.lora_train.transcribe", return_value="hello") as mock_t:
        result = _get_hypotheses(model, samples, processor, torch.device("cpu"))

    assert len(result) == 3
    assert mock_t.call_count == 3
