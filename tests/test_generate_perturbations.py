from __future__ import annotations

from pathlib import Path

import numpy as np

from src.acoustic_feature_extraction.pipeline.generate_perturbations import (
    _build_output_path,
    _manifest_row,
    _variant_pause_insert,
)


def test_pause_insertion_increases_length() -> None:
    sr = 16000
    audio = np.ones(int(2.0 * sr), dtype=np.float32) * 0.1
    out = _variant_pause_insert(audio, sr=sr, pause_duration_sec=0.2)
    assert out.shape[0] > audio.shape[0]


def test_output_path_has_variant_tag(tmp_path: Path) -> None:
    path = _build_output_path(
        out_root=tmp_path,
        dataset="l2arctic",
        speaker_id="ABA",
        utterance_id="arctic_a0001",
        variant="rate_p10",
    )
    assert str(path).endswith("l2arctic/ABA/arctic_a0001__rate_p10.wav")


def test_manifest_row_duration() -> None:
    row = _manifest_row(
        dataset="saa",
        speaker_id="1",
        utterance_id="afrikaans1",
        variant="pitch_p2st",
        source="perturbed",
        output_path=Path("data/processed/perturbations/saa/1/afrikaans1__pitch_p2st.wav"),
        sampling_rate=16000,
        num_samples=32000,
        params={"semitones": 2.0},
    )
    assert row.duration_seconds == 2.0
    assert row.variant == "pitch_p2st"

