from __future__ import annotations

import re

import jiwer


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fair WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    """
    Compute Word Error Rate between reference and hypothesis transcripts.

    Text is normalized (lowercased, punctuation stripped) before comparison.

    Args:
        references: Ground truth transcripts.
        hypotheses: ASR output transcripts.

    Returns:
        WER as a float in [0.0, 1.0] (values > 1.0 are possible with many insertions).

    Raises:
        ValueError: If the lists have different lengths or are empty.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, "
            f"got {len(references)} and {len(hypotheses)}"
        )
    if not references:
        raise ValueError("references and hypotheses must not be empty")

    refs = [_normalize(r) for r in references]
    hyps = [_normalize(h) for h in hypotheses]

    return jiwer.wer(refs, hyps)
