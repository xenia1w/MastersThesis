"""
Post-hoc LLM correction of the v2 baseline ASR hypotheses (RQ2 extension).

Prompting Whisper via ``initial_prompt`` hurt WER (see the v2 metadata_only result). This
module tries the opposite lever: leave decoding alone and instead have an LLM fix recognition
errors in the *raw* baseline transcript of the [5:00, 15:00] evaluation window, then score the
corrected text against the same hand-annotated golden references.

Runs locally (KISSKI LLM, like profile building — no cluster). Reads ``baseline_all.csv`` and
writes ``<out-dir>/prompted_all.csv`` (plus per-call ``posthoc_<id>.csv``) in the exact shape
``earnings21_window_score.py`` expects, so scoring needs no changes::

    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_window_score \\
        --annotations manual_annotation.csv --approach posthoc_blind

Design (see arXiv 2409.09554 / 2307.04172):
- Conservative / verbatim correction — the reference is verbatim .nlp text, and we only have a
  single (1-best) hypothesis, the setup most prone to over-correction. The prompt forbids
  rewriting; an edit-ratio guard reverts to the raw hypothesis if the LLM changed too much.
- Sentence-boundary chunking (~200 words) keeps each LLM call short so verbatim discipline
  holds and output length can't drift/truncate over a long span.

Three modes:
- blind (default): the LLM sees only the eval-window chunk (writes to earnings21_window_posthoc_blind).
- context (--use-context): each chunk is corrected with the unprompted 0:00-5:00 transcript of the
  SAME call as an in-prompt reference — self-contained (no metadata, no LLM-built profile), the
  post-hoc counterpart of the transcript_only prompting method. Writes to
  earnings21_window_posthoc_context. The reference is raw ASR, so the prompt treats it as a hint
  only and forbids copying from it.
- profile (--use-profile): each chunk is corrected with a curated, spell-corrected entity-list
  profile of the SAME call as an AUTHORITATIVE reference — the post-hoc counterpart of injecting
  that same profile into Whisper's initial_prompt (one profile, two injection points). Unlike
  context, the prompt tells the model to trust the list and adopt its spellings. --profile-strategy
  selects WHICH built profile (default transcript_metadata_knowledge; also metadata_only,
  transcript_only, transcript_plus_knowledge, or any *_prose variant) and names the output dir
  earnings21_window_posthoc_profile_<strategy>, so runs over different profiles never collide.

Usage:
    # blind
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_posthoc_correct --skip-existing
    # context (0:00-5:00 self-reference)
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_posthoc_correct \\
        --use-context --skip-existing
    # profile (default transcript_metadata_knowledge)
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_posthoc_correct \\
        --use-profile --skip-existing
    # profile — switch strategy with one flag
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_posthoc_correct \\
        --use-profile --profile-strategy metadata_only --skip-existing

Score with earnings21_window_score.py --approach posthoc_blind / posthoc_context /
posthoc_profile_<strategy> (e.g. posthoc_profile_transcript_metadata_knowledge).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.models.constants import DEFAULT_LLM_MODEL
from src.lexical_stylistic_prompting.models.prompts import (
    POSTHOC_CONTEXT_SYSTEM,
    POSTHOC_CONTEXT_USER,
    POSTHOC_CORRECT_SYSTEM,
    POSTHOC_CORRECT_USER,
    POSTHOC_PROFILE_SYSTEM,
    POSTHOC_PROFILE_USER,
)
from src.lexical_stylistic_prompting.models.speaker_profile import _get_client, _llm_call

DEFAULT_BASELINE = Path(
    "data/processed/lexical_stylistic_prompting/v2/earnings21_window_baseline/baseline_all.csv"
)
DEFAULT_OUT_DIR = Path(
    "data/processed/lexical_stylistic_prompting/v2/earnings21_window_posthoc_blind"
)
# "context" mode: correct the [5:00, 15:00] window using the unprompted 0:00-5:00 transcript of
# the same call as an in-prompt reference (self-contained — no metadata, no LLM-built profile).
DEFAULT_CONTEXT_OUT_DIR = Path(
    "data/processed/lexical_stylistic_prompting/v2/earnings21_window_posthoc_context"
)
DEFAULT_PROFILE_TRANSCRIPTS_DIR = Path(
    "data/processed/lexical_stylistic_prompting/v2/profile_transcripts"
)
# "profile" mode: correct the [5:00, 15:00] window using a curated, spell-corrected entity-list
# profile of the same call as an authoritative reference (the post-hoc counterpart of injecting
# that same profile into Whisper's initial_prompt). WHICH profile is chosen by --profile-strategy:
# each strategy is a subdir of PROFILES_ROOT, so switching transcript_metadata_knowledge →
# metadata_only / transcript_only / transcript_plus_knowledge / any *_prose variant is one flag.
# The chosen strategy also names the output dir (earnings21_window_posthoc_profile_<strategy>) so
# runs over different profiles never overwrite one another.
PROFILES_ROOT = Path("data/processed/lexical_stylistic_prompting/v2/profiles")
DEFAULT_PROFILE_STRATEGY = "transcript_metadata_knowledge"
# base under which profile mode writes earnings21_window_posthoc_profile_<strategy>/
POSTHOC_OUT_ROOT = Path("data/processed/lexical_stylistic_prompting/v2")
DEFAULT_PROFILE_TAG = 300  # profile-window length in seconds; filenames are <call_id>_<tag>.json
# Sentence-grouped chunk size sent to the LLM. Larger chunks = fewer requests (KISSKI caps at
# ~10/min, 200/hour, 400/day); a ~1,400-word window at 450 words is ~3 requests instead of ~7.
CHUNK_TARGET_WORDS = 450
DEFAULT_MAX_EDIT_RATIO = 0.30  # revert to raw if the correction changed more than this fraction


class PosthocRow(BaseModel):
    call_id: str
    model: str
    raw_words: int
    corrected_words: int
    edit_ratio: float       # word-level change of the LLM output vs the raw hypothesis
    guard_applied: bool     # True → over-correction guard tripped, hypothesis reverted to raw
    hypothesis: str         # final (possibly reverted) text — what score.py reads
    raw_hypothesis: str


# ── chunking ──────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split on whitespace following sentence-terminal punctuation, keeping the punctuation."""
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in pieces if p.strip()]


def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """Group whole sentences into chunks of ~target_words (never splitting a sentence)."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > target_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── correction ────────────────────────────────────────────────────────────────

def _correct_chunk(client: OpenAI, model: str, chunk: str,
                   reference: str | None = None, profile: str | None = None) -> str:
    words = len(chunk.split())
    # allow headroom over the input length so a correct-length output never truncates
    max_tokens = max(256, int(words * 2) + 64)
    if profile:
        system = POSTHOC_PROFILE_SYSTEM
        user = POSTHOC_PROFILE_USER.format(profile=profile, chunk=chunk)
    elif reference:
        system = POSTHOC_CONTEXT_SYSTEM
        user = POSTHOC_CONTEXT_USER.format(reference=reference, chunk=chunk)
    else:
        system = POSTHOC_CORRECT_SYSTEM
        user = POSTHOC_CORRECT_USER.format(chunk=chunk)
    raw = _llm_call(client, model, system, user, max_tokens=max_tokens)
    return raw.strip().strip("\"'").strip()


def correct_hypothesis(client: OpenAI, model: str, hypothesis: str,
                       chunk_words: int = CHUNK_TARGET_WORDS, label: str = "",
                       reference: str | None = None, profile: str | None = None) -> str:
    """Correct a full window hypothesis chunk-by-chunk and reassemble it.

    When ``profile`` is given (the curated transcript_metadata_knowledge entity list of the same
    call), each chunk is corrected with that authoritative list as an in-prompt reference. Otherwise,
    when ``reference`` is given (the raw 0:00-5:00 transcript of the same call), each chunk is
    corrected with that preamble as a hint; otherwise correction is blind.
    """
    chunks = chunk_text(hypothesis, chunk_words)
    corrected: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"    {label} chunk {i}/{len(chunks)} ({len(chunk.split())} words) → KISSKI ...")
        corrected.append(_correct_chunk(client, model, chunk, reference, profile))
    return " ".join(part for part in corrected if part).strip()


def _edit_ratio(raw: str, corrected: str) -> float:
    """Word-level fraction of the raw hypothesis changed by the correction (normalized)."""
    if not raw.strip():
        return 0.0
    return compute_wer([raw], [corrected])


def process_call(client: OpenAI, model: str, call_id: str, raw: str,
                 max_edit_ratio: float, chunk_words: int = CHUNK_TARGET_WORDS,
                 reference: str | None = None, profile: str | None = None) -> PosthocRow:
    corrected = correct_hypothesis(client, model, raw, chunk_words, label=call_id,
                                   reference=reference, profile=profile)
    ratio = _edit_ratio(raw, corrected)
    guard = ratio > max_edit_ratio or not corrected.strip()
    if guard:
        logger.warning(f"{call_id}: edit_ratio={ratio:.3f} > {max_edit_ratio} — reverting to raw")
    final = raw if guard else corrected
    return PosthocRow(
        call_id=call_id,
        model=model,
        raw_words=len(raw.split()),
        corrected_words=len(corrected.split()),
        edit_ratio=round(ratio, 4),
        guard_applied=guard,
        hypothesis=final,
        raw_hypothesis=raw,
    )


# ── io ────────────────────────────────────────────────────────────────────────

def _write_row(out_dir: Path, row: PosthocRow) -> None:
    out = out_dir / f"posthoc_{row.call_id}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PosthocRow.model_fields))
        writer.writeheader()
        writer.writerow(row.model_dump())


def _merge_all(out_dir: Path) -> Path:
    files = sorted(glob.glob(str(out_dir / "posthoc_[0-9]*.csv")))
    merged = out_dir / "prompted_all.csv"
    pd.concat([pd.read_csv(f) for f in files]).to_csv(merged, index=False)
    logger.info(f"Merged {len(files)} per-call files → {merged}")
    return merged


def _load_reference(ref_dir: Path, call_id: str, tag: int) -> str:
    """Load the unprompted 0:00-5:00 transcript used as an in-prompt correction hint."""
    path = ref_dir / f"{call_id}_{tag}.json"
    if not path.exists():
        logger.warning(f"{call_id}: no reference transcript at {path} — correcting this call blind")
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("transcript", "")).strip()


def profiles_dir_for(root: Path, strategy: str) -> Path:
    """Directory of per-call profile JSONs for a strategy (a subdir of the profiles root)."""
    return root / strategy


def profile_out_dir_for(strategy: str) -> Path:
    """Output dir for a profile strategy — strategy-tagged so runs never collide."""
    return POSTHOC_OUT_ROOT / f"earnings21_window_posthoc_profile_{strategy}"


def _load_profile(profiles_dir: Path, call_id: str, tag: int) -> str:
    """Load a curated entity-list profile (its ``prompt`` field) as an authoritative reference."""
    path = profiles_dir / f"{call_id}_{tag}.json"
    if not path.exists():
        logger.warning(f"{call_id}: no profile at {path} — correcting this call blind")
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("prompt", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc LLM correction of v2 baseline hypotheses")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="baseline_all.csv with call_id + hypothesis columns")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default: posthoc_blind, or posthoc_context / posthoc_profile "
                             "in the respective modes)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--use-context", action="store_true",
                            help="correct each call using its unprompted 0:00-5:00 transcript as an "
                                 "in-prompt reference (self-contained; no metadata/profile)")
    mode_group.add_argument("--use-profile", action="store_true",
                            help="correct each call using its curated transcript_metadata_knowledge "
                                 "entity list as an authoritative in-prompt reference")
    parser.add_argument("--reference-transcripts-dir", type=Path,
                        default=DEFAULT_PROFILE_TRANSCRIPTS_DIR,
                        help="dir of 0:00-5:00 transcripts (<call_id>_<tag>.json) for --use-context")
    parser.add_argument("--profile-strategy", default=DEFAULT_PROFILE_STRATEGY,
                        help="which built profile to inject in --use-profile mode: a subdir of "
                             f"{PROFILES_ROOT} (e.g. metadata_only, transcript_only, "
                             "transcript_plus_knowledge, transcript_metadata_knowledge, or any "
                             "*_prose variant). Also names the output dir.")
    parser.add_argument("--profiles-root", type=Path, default=PROFILES_ROOT,
                        help="root dir holding the per-strategy profile subdirs")
    parser.add_argument("--profile-tag", type=int, default=DEFAULT_PROFILE_TAG,
                        help="reference/profile filename tag (profile-window seconds)")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--max-edit-ratio", type=float, default=DEFAULT_MAX_EDIT_RATIO,
                        help="revert to the raw hypothesis if the correction changed more than this")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_TARGET_WORDS,
                        help="sentence-grouped chunk size sent to the LLM (larger = fewer requests)")
    parser.add_argument("--call-id", default=None, help="Process a single call only")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip calls that already have a posthoc_<id>.csv")
    args = parser.parse_args()

    profiles_dir = profiles_dir_for(args.profiles_root, args.profile_strategy)
    if args.use_profile and not profiles_dir.is_dir():
        available = sorted(p.name for p in args.profiles_root.iterdir() if p.is_dir())
        logger.error(f"--profile-strategy {args.profile_strategy!r} not found at {profiles_dir}. "
                     f"Available: {', '.join(available) or '(none)'}")
        return

    out_dir = args.out_dir or (
        profile_out_dir_for(args.profile_strategy) if args.use_profile
        else DEFAULT_CONTEXT_OUT_DIR if args.use_context
        else DEFAULT_OUT_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = (
        f"profile ({args.profile_strategy} entity list)" if args.use_profile
        else "context (0:00-5:00 reference)" if args.use_context
        else "blind"
    )
    logger.info(f"Post-hoc correction mode: {mode} → {out_dir}")
    df = pd.read_csv(args.baseline)
    df["call_id"] = df["call_id"].astype(str)
    if args.call_id:
        df = df[df["call_id"] == args.call_id]
        if df.empty:
            logger.error(f"Call {args.call_id!r} not found in {args.baseline}")
            return

    client = _get_client()
    corrected, skipped, reverted = 0, 0, 0
    total = len(df)

    for idx, (_, r) in enumerate(df.iterrows(), 1):
        call_id = str(r["call_id"])
        out_path = out_dir / f"posthoc_{call_id}.csv"
        if args.skip_existing and out_path.exists():
            logger.info(f"[{idx}/{total}] {call_id}: skipping (cached)")
            skipped += 1
            continue
        raw = str(r["hypothesis"]) if pd.notna(r["hypothesis"]) else ""
        reference = (
            _load_reference(args.reference_transcripts_dir, call_id, args.profile_tag)
            if args.use_context else None
        )
        profile = (
            _load_profile(profiles_dir, call_id, args.profile_tag)
            if args.use_profile else None
        )
        n_chunks = len(chunk_text(raw, args.chunk_words))
        hint = profile or reference
        ref_note = f", {'profile' if profile else 'ref'}={len(hint.split())}w" if hint else ""
        logger.info(f"[{idx}/{total}] {call_id}: correcting {len(raw.split())} words "
                    f"in {n_chunks} chunks{ref_note} ...")
        row = process_call(client, args.model, call_id, raw, args.max_edit_ratio,
                           args.chunk_words, reference=reference, profile=profile)
        _write_row(out_dir, row)
        corrected += 1
        reverted += int(row.guard_applied)
        logger.success(f"[{idx}/{total}] {call_id}: done — edit_ratio={row.edit_ratio} "
                       f"guard={row.guard_applied} ({row.raw_words}→{row.corrected_words} words)")

    logger.success(f"Finished — corrected {corrected} (reverted {reverted}), skipped {skipped} / {total}")
    _merge_all(out_dir)


if __name__ == "__main__":
    main()
