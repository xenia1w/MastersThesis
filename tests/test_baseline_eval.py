from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from src.asr_adaptation.pipeline.baseline_eval import (
    BaselineRow,
    SAA_REFERENCE,
    _save_csv,
)


# ---------------------------------------------------------------------------
# Tests: transcribe utility (via mocked model)
# ---------------------------------------------------------------------------

def test_transcribe_returns_string() -> None:
    from src.asr_adaptation.inference.transcribe import transcribe

    processor = MagicMock()
    processor.return_value.input_values = torch.zeros(1, 800)
    processor.decode.return_value = "hello world"

    model = MagicMock()
    model.return_value.logits = torch.zeros(1, 10, 32)

    waveform = torch.zeros(800)
    result = transcribe(waveform, processor, model, torch.device("cpu"))
    assert isinstance(result, str)


def test_transcribe_chunks_long_audio() -> None:
    """Audio longer than chunk_length_s should be split into multiple calls."""
    from src.asr_adaptation.inference.transcribe import transcribe

    processor = MagicMock()
    processor.return_value.input_values = torch.zeros(1, 800)
    processor.decode.return_value = "chunk"

    model = MagicMock()
    model.return_value.logits = torch.zeros(1, 10, 32)

    # 3 seconds of audio, chunk_length_s=1 → 3 chunks
    waveform = torch.zeros(3 * 16000)
    result = transcribe(
        waveform, processor, model, torch.device("cpu"), chunk_length_s=1
    )

    assert processor.decode.call_count == 3
    assert result == "chunk chunk chunk"


# ---------------------------------------------------------------------------
# Tests: _save_csv
# ---------------------------------------------------------------------------

def test_save_csv_creates_file(tmp_path: Path) -> None:
    rows = [
        BaselineRow(
            speaker_id="ABA",
            utterance_id="arctic_a0001",
            native_language=None,
            reference="hello world",
            hypothesis="hello world",
            wer=0.0,
        )
    ]
    out = tmp_path / "subdir" / "results.csv"
    _save_csv(rows, out)
    assert out.exists()


def test_save_csv_correct_columns(tmp_path: Path) -> None:
    rows = [
        BaselineRow(
            speaker_id="ABA",
            utterance_id="arctic_a0001",
            native_language=None,
            reference="hello world",
            hypothesis="hello there",
            wer=0.5,
        )
    ]
    out = tmp_path / "results.csv"
    _save_csv(rows, out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        result_rows = list(reader)

    assert len(result_rows) == 1
    row = result_rows[0]
    assert row["speaker_id"] == "ABA"
    assert row["utterance_id"] == "arctic_a0001"
    assert row["reference"] == "hello world"
    assert row["hypothesis"] == "hello there"
    assert float(row["wer"]) == pytest.approx(0.5)


def test_save_csv_empty_optional_fields(tmp_path: Path) -> None:
    rows = [
        BaselineRow(
            speaker_id="1",
            utterance_id=None,
            native_language="mandarin",
            reference=SAA_REFERENCE,
            hypothesis="some transcription",
            wer=0.3,
        )
    ]
    out = tmp_path / "saa.csv"
    _save_csv(rows, out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        row = list(reader)[0]

    assert row["utterance_id"] == ""
    assert row["native_language"] == "mandarin"


# ---------------------------------------------------------------------------
# Tests: SAA_REFERENCE constant
# ---------------------------------------------------------------------------

def test_saa_reference_is_lowercase_no_punctuation() -> None:
    """Reference is pre-normalized so WER comparison is fair."""
    assert SAA_REFERENCE == SAA_REFERENCE.lower()
    assert "." not in SAA_REFERENCE
    assert "," not in SAA_REFERENCE


def test_saa_reference_contains_key_words() -> None:
    assert "stella" in SAA_REFERENCE
    assert "snow peas" in SAA_REFERENCE
    assert "train station" in SAA_REFERENCE
