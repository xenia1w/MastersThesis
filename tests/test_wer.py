from __future__ import annotations

import pytest

from src.asr_adaptation.metrics.wer import compute_wer


def test_exact_match_is_zero() -> None:
    assert compute_wer(["hello world"], ["hello world"]) == 0.0


def test_full_mismatch_is_one() -> None:
    assert compute_wer(["hello world"], ["foo bar"]) == 1.0


def test_one_word_wrong_out_of_two() -> None:
    wer = compute_wer(["hello world"], ["hello there"])
    assert wer == pytest.approx(0.5)


def test_normalization_ignores_case() -> None:
    assert compute_wer(["Hello World"], ["hello world"]) == 0.0


def test_normalization_strips_punctuation() -> None:
    assert compute_wer(["Hello, world!"], ["hello world"]) == 0.0


def test_multiple_pairs() -> None:
    refs = ["hello world", "foo bar"]
    hyps = ["hello world", "foo bar"]
    assert compute_wer(refs, hyps) == 0.0


def test_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_wer([], [])


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_wer(["hello"], ["hello", "world"])
