from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.asr_adaptation.data.l2arctic_transcriptions import (
    L2ArcticTranscriptSample,
    list_l2arctic_samples_with_transcripts,
    load_l2arctic_transcript,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(num_samples: int = 8000, sr: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    audio = np.zeros(num_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _make_outer_zip(
    speaker_id: str,
    utterances: dict[str, str],  # {utterance_id: transcript_text}
    tmp_path: Path,
) -> Path:
    """
    Build a fake nested L2-ARCTIC zip at tmp_path/outer.zip.

    Structure mirrors real dataset:
      outer.zip
        {speaker_id}.zip
          {speaker_id}/wav/{utterance_id}.wav
          {speaker_id}/transcript/{utterance_id}.txt
    """
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        for utt_id, text in utterances.items():
            inner.writestr(f"{speaker_id}/wav/{utt_id}.wav", _make_wav_bytes())
            inner.writestr(f"{speaker_id}/transcript/{utt_id}.txt", text)

    outer_path = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr(f"{speaker_id}.zip", inner_buf.getvalue())

    return outer_path


# ---------------------------------------------------------------------------
# Tests: load_l2arctic_transcript
# ---------------------------------------------------------------------------

def test_load_transcript_returns_correct_text(tmp_path: Path) -> None:
    outer = _make_outer_zip("ABA", {"arctic_a0001": "Please call Stella"}, tmp_path)
    text = load_l2arctic_transcript(str(outer), "ABA", "arctic_a0001.wav")
    assert text == "Please call Stella"


def test_load_transcript_strips_whitespace(tmp_path: Path) -> None:
    outer = _make_outer_zip("ABA", {"arctic_a0001": "  hello world\n"}, tmp_path)
    text = load_l2arctic_transcript(str(outer), "ABA", "arctic_a0001.wav")
    assert text == "hello world"


def test_load_transcript_missing_file_raises(tmp_path: Path) -> None:
    outer = _make_outer_zip("ABA", {"arctic_a0001": "hello"}, tmp_path)
    with pytest.raises(KeyError):
        load_l2arctic_transcript(str(outer), "ABA", "arctic_a0099.wav")


# ---------------------------------------------------------------------------
# Tests: list_l2arctic_samples_with_transcripts
# ---------------------------------------------------------------------------

def test_list_samples_returns_all_utterances(tmp_path: Path) -> None:
    utterances = {
        "arctic_a0001": "First sentence",
        "arctic_a0002": "Second sentence",
        "arctic_a0003": "Third sentence",
    }
    outer = _make_outer_zip("ASI", utterances, tmp_path)
    samples = list_l2arctic_samples_with_transcripts(str(outer), "ASI")
    assert len(samples) == 3


def test_list_samples_correct_metadata(tmp_path: Path) -> None:
    outer = _make_outer_zip("BWC", {"arctic_a0001": "Hello world"}, tmp_path)
    samples = list_l2arctic_samples_with_transcripts(str(outer), "BWC")

    s = samples[0]
    assert isinstance(s, L2ArcticTranscriptSample)
    assert s.speaker_id == "BWC"
    assert s.utterance_id == "arctic_a0001"
    assert s.transcript == "Hello world"
    assert s.sampling_rate == 16000


def test_list_samples_sorted_by_utterance_id(tmp_path: Path) -> None:
    utterances = {
        "arctic_a0003": "C",
        "arctic_a0001": "A",
        "arctic_a0002": "B",
    }
    outer = _make_outer_zip("ABA", utterances, tmp_path)
    samples = list_l2arctic_samples_with_transcripts(str(outer), "ABA")

    ids = [s.utterance_id for s in samples]
    assert ids == sorted(ids)


def test_list_samples_waveform_is_1d_float32(tmp_path: Path) -> None:
    outer = _make_outer_zip("ABA", {"arctic_a0001": "test"}, tmp_path)
    samples = list_l2arctic_samples_with_transcripts(str(outer), "ABA")

    waveform = samples[0].waveform
    assert waveform.ndim == 1
    assert waveform.dtype == float32_dtype()


def float32_dtype():
    import torch
    return torch.float32
