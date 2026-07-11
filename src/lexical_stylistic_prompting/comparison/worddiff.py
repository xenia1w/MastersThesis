"""Word-level diff of two hypotheses against a shared reference.

Aligns each hypothesis to the reference with the same ``jiwer`` + ``_normalize`` path the
eval used, buckets errors by reference-token index, then diffs baseline vs prompted to see
which reference tokens the prompt *fixed* vs *degraded*. Each token is tagged as an entity
or not (reconstructed from ``load_earnings21``) so the word changes line up with the EER.
"""

from __future__ import annotations

from pathlib import Path

from jiwer import process_words

from src.asr_adaptation.metrics.wer import _normalize
from src.lexical_stylistic_prompting.data.earnings21_utils import load_earnings21

DELETED = "∅"  # placeholder shown when the reference token was deleted (not just substituted)


def _alignment(ref_norm: str, hyp_norm: str) -> list:
    return process_words(ref_norm, hyp_norm).alignments[0]


def ref_error_map(ref_norm: str, hyp_norm: str) -> dict[int, str]:
    """Map each errored reference-token index → the hypothesis text that replaced it."""
    hyp_words = hyp_norm.split()
    errors: dict[int, str] = {}
    for chunk in _alignment(ref_norm, hyp_norm):
        if chunk.type == "substitute":
            replacement = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors[i] = replacement
        elif chunk.type == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors[i] = DELETED
    return errors


def inserted_tokens(ref_norm: str, hyp_norm: str) -> list[str]:
    """Hypothesis tokens with no reference counterpart (word-gluing / hallucinations)."""
    hyp_words = hyp_norm.split()
    out: list[str] = []
    for chunk in _alignment(ref_norm, hyp_norm):
        if chunk.type == "insert":
            out.extend(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
    return out


def entity_mask_for_call(call_id: str, data_dir: Path, n_profile: int) -> list[bool]:
    """Per-reference-token entity flags for the eval window (segments after n_profile).

    Rebuilt exactly as the eval did (``segments[n_profile:]``, concatenated per-token
    masks), so index i lines up with the i-th reference token the same way EER counted it.
    """
    for call in load_earnings21(data_dir, min_tokens=5):
        if call.call_id == call_id:
            eval_segments = call.segments[n_profile:]
            return [flag for seg in eval_segments for flag in seg.entity_mask]
    raise ValueError(f"call {call_id!r} not found under {data_dir}")


def is_entity(index: int, mask: list[bool]) -> bool:
    """Entity flag for a reference-token index, tolerant of length drift from normalization."""
    return index < len(mask) and mask[index]


def format_changes(indices: list[int], ref_words: list[str], error_map: dict[int, str],
                   mask: list[bool]) -> list[str]:
    """Human-readable ``[ENTITY] ref → hyp`` lines, entity tokens first then by position."""
    ordered = sorted(indices, key=lambda i: (not is_entity(i, mask), i))
    lines: list[str] = []
    for i in ordered:
        tag = "[ENTITY] " if is_entity(i, mask) else "         "
        ref_word = ref_words[i] if i < len(ref_words) else "?"
        lines.append(f"  {tag}{ref_word!r} -> {error_map.get(i, '?')!r}")
    return lines
