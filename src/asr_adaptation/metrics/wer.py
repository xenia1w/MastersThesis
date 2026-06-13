from __future__ import annotations

import re

import jiwer
from num2words import num2words


def _digits_to_words(text: str) -> str:
    def _ordinal(m: re.Match) -> str:
        return num2words(int(m.group(1)), to="ordinal")

    def _cardinal(m: re.Match) -> str:
        return num2words(int(m.group(0)))

    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _ordinal, text)
    text = re.sub(r"\b\d+\b", _cardinal, text)
    return text


def _normalize(text: str) -> str:
    """Lowercase, strip <unk> tokens, normalize numbers, and strip punctuation.

    Also collapses TED-LIUM's space-before-apostrophe contraction style
    (e.g. "it 's" → "its", "can 't" → "cant") so references and Whisper
    hypotheses are treated identically.

    Number normalization converts digit sequences to word form so that
    Whisper's digit output ("30") matches TED-LIUM word-form references
    ("thirty"). Ordinals ("20th" → "twentieth") are handled too.
    Hyphens are replaced with spaces so "twenty-one" and "twenty one"
    are treated identically.

    Limitation: four-digit years (e.g. 2014) are converted to "two thousand
    and fourteen" which may not match references that say "twenty fourteen".
    """
    text = re.sub(r"<unk>", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\s'", "'", text)   # "it 's" → "it's", "can 't" → "can't"
    text = _digits_to_words(text)
    text = re.sub(r"-", " ", text)     # "twenty-one" → "twenty one"
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
