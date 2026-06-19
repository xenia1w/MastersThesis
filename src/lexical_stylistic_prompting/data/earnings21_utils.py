from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel

SAMPLE_RATE = 16_000
RTTM_MERGE_GAP = 2.0   # merge consecutive same-speaker RTTM segments within this gap
MAX_SEGMENT_DURATION = 30.0  # split turns longer than this for Whisper

# Entity types that represent genuine vocabulary challenges for ASR
VOCABULARY_ENTITY_TYPES = {
    "ORG", "PERSON", "PRODUCT", "GPE", "LOC", "FAC",
    "WEBSITE", "ABBREVIATION", "MONEY", "DATE", "TIME", "YEAR",
}


# ── Data models ───────────────────────────────────────────────────────────────

class Earnings21Token(BaseModel):
    text: str
    speaker: int
    entity_type: str | None


class Earnings21Segment(BaseModel):
    segment_id: str
    call_id: str
    speaker: int
    tokens: list[Earnings21Token]
    start_ts: float
    end_ts: float

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def entity_mask(self) -> list[bool]:
        return [t.entity_type in VOCABULARY_ENTITY_TYPES for t in self.tokens]


class Earnings21Call(BaseModel):
    call_id: str
    company_name: str
    sector: str
    financial_quarter: str
    audio_path: Path
    segments: list[Earnings21Segment]

    model_config = {"arbitrary_types_allowed": True}


# ── Parsers ───────────────────────────────────────────────────────────────────

def _load_wer_tags(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v["entity_type"] for k, v in raw.items()}


def _parse_wer_tag_ids(cell: str) -> list[str]:
    cell = cell.strip()
    if not cell or cell == "[]":
        return []
    try:
        parsed = ast.literal_eval(cell)
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _parse_nlp_speaker_turns(
    nlp_path: Path,
    entity_map: dict[str, str],
) -> list[tuple[int, list[Earnings21Token]]]:
    """Return ordered list of (speaker_id, tokens) for each speaker turn."""
    turns: list[tuple[int, list[Earnings21Token]]] = []
    current_speaker: int | None = None
    current_tokens: list[Earnings21Token] = []

    with open(nlp_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            text = row.get("token", "").strip()
            if not text:
                continue
            try:
                speaker = int(row.get("speaker", "0"))
            except ValueError:
                continue

            tag_ids = _parse_wer_tag_ids(row.get("wer_tags", "[]"))
            entity_type: str | None = None
            for tid in tag_ids:
                etype = entity_map.get(tid)
                if etype in VOCABULARY_ENTITY_TYPES:
                    entity_type = etype
                    break

            tok = Earnings21Token(text=text, speaker=speaker, entity_type=entity_type)

            if speaker != current_speaker:
                if current_tokens and current_speaker is not None:
                    turns.append((current_speaker, current_tokens))
                current_speaker = speaker
                current_tokens = [tok]
            else:
                current_tokens.append(tok)

    if current_tokens and current_speaker is not None:
        turns.append((current_speaker, current_tokens))

    return turns


def _parse_rttm_turns(
    rttm_path: Path,
) -> list[tuple[int, float, float]]:
    """Parse RTTM and merge same-speaker segments within RTTM_MERGE_GAP.

    Returns list of (speaker_id, start_ts, end_ts) ordered by start time.
    """
    raw: list[tuple[float, float, int]] = []
    with open(rttm_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = int(parts[7])
            raw.append((start, start + dur, spk))

    raw.sort()
    merged: list[list] = []
    for start, end, spk in raw:
        if merged and merged[-1][2] == spk and (start - merged[-1][1]) < RTTM_MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, spk])

    return [(m[2], m[0], m[1]) for m in merged]


def _split_long_turn(
    speaker: int,
    tokens: list[Earnings21Token],
    start_ts: float,
    end_ts: float,
    call_id: str,
    seg_idx: int,
) -> list[tuple[int, list[Earnings21Token], float, float, int]]:
    """Split a long turn into ≤MAX_SEGMENT_DURATION chunks by token count."""
    duration = end_ts - start_ts
    if duration <= MAX_SEGMENT_DURATION or not tokens:
        return [(speaker, tokens, start_ts, end_ts, seg_idx)]

    n_chunks = int(np.ceil(duration / MAX_SEGMENT_DURATION))
    chunk_size = max(1, len(tokens) // n_chunks)
    chunk_dur = duration / n_chunks
    results = []
    for i in range(n_chunks):
        t_start = start_ts + i * chunk_dur
        t_end = min(start_ts + (i + 1) * chunk_dur, end_ts)
        tok_start = i * chunk_size
        tok_end = tok_start + chunk_size if i < n_chunks - 1 else len(tokens)
        if tok_start >= len(tokens):
            break
        results.append((speaker, tokens[tok_start:tok_end], t_start, t_end, seg_idx + i))
    return results


def _build_segments(
    nlp_turns: list[tuple[int, list[Earnings21Token]]],
    rttm_turns: list[tuple[int, float, float]],
    call_id: str,
    min_tokens: int,
) -> list[Earnings21Segment]:
    """Match NLP speaker turns to RTTM timing by (speaker, per-speaker order)."""
    # Index RTTM turns per speaker in order
    rttm_by_speaker: dict[int, list[tuple[float, float]]] = {}
    for spk, start, end in rttm_turns:
        rttm_by_speaker.setdefault(spk, []).append((start, end))

    speaker_counters: dict[int, int] = {}
    segments: list[Earnings21Segment] = []
    seg_idx = 0

    for spk, tokens in nlp_turns:
        if len(tokens) < min_tokens:
            continue
        idx = speaker_counters.get(spk, 0)
        speaker_counters[spk] = idx + 1

        rttm_list = rttm_by_speaker.get(spk, [])
        if idx < len(rttm_list):
            start_ts, end_ts = rttm_list[idx]
        else:
            # No matching RTTM turn — skip (can't slice audio without timing)
            logger.debug(f"{call_id}: no RTTM turn {idx} for speaker {spk}, skipping")
            continue

        for sub_spk, sub_tokens, sub_start, sub_end, si in _split_long_turn(
            spk, tokens, start_ts, end_ts, call_id, seg_idx
        ):
            segments.append(Earnings21Segment(
                segment_id=f"{call_id}_{si:04d}",
                call_id=call_id,
                speaker=sub_spk,
                tokens=sub_tokens,
                start_ts=sub_start,
                end_ts=sub_end,
            ))
            seg_idx = si + 1

    return segments


# ── Metadata ─────────────────────────────────────────────────────────────────

def _load_metadata(data_dir: Path) -> dict[str, dict]:
    meta_path = data_dir / "earnings21-file-metadata.csv"
    result: dict[str, dict] = {}
    if not meta_path.exists():
        logger.warning(f"Metadata file not found: {meta_path}")
        return result
    with open(meta_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = row["file_id"].strip()
            result[fid] = {
                "company_name": row.get("company_name", ""),
                "sector": row.get("sector", ""),
                "financial_quarter": row.get("financial_quarter", ""),
            }
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def load_earnings21(
    data_dir: Path,
    min_tokens: int = 5,
) -> list[Earnings21Call]:
    """Load all calls from the earnings21 data directory.

    Expected layout:
        data_dir/transcripts/nlp_references/<id>.nlp
        data_dir/transcripts/wer_tags/<id>.wer_tag.json
        data_dir/rttms/<id>.rttm
        data_dir/media/<id>.mp3
        data_dir/earnings21-file-metadata.csv
    """
    nlp_dir = data_dir / "transcripts" / "nlp_references"
    wer_tags_dir = data_dir / "transcripts" / "wer_tags"
    rttm_dir = data_dir / "rttms"
    media_dir = data_dir / "media"

    if not nlp_dir.exists():
        raise FileNotFoundError(f"NLP references directory not found: {nlp_dir}")

    metadata = _load_metadata(data_dir)
    calls: list[Earnings21Call] = []

    for nlp_path in sorted(nlp_dir.glob("*.nlp")):
        call_id = nlp_path.stem
        audio_path = media_dir / f"{call_id}.mp3"
        rttm_path = rttm_dir / f"{call_id}.rttm"

        if not audio_path.exists():
            logger.warning(f"{call_id}: no audio, skipping")
            continue
        if not rttm_path.exists():
            logger.warning(f"{call_id}: no RTTM, skipping")
            continue

        wer_tags_path = wer_tags_dir / f"{call_id}.wer_tag.json"
        entity_map = _load_wer_tags(wer_tags_path) if wer_tags_path.exists() else {}

        nlp_turns = _parse_nlp_speaker_turns(nlp_path, entity_map)
        rttm_turns = _parse_rttm_turns(rttm_path)
        segments = _build_segments(nlp_turns, rttm_turns, call_id, min_tokens)

        meta = metadata.get(call_id, {})
        calls.append(Earnings21Call(
            call_id=call_id,
            company_name=meta.get("company_name", ""),
            sector=meta.get("sector", ""),
            financial_quarter=meta.get("financial_quarter", ""),
            audio_path=audio_path,
            segments=segments,
        ))
        logger.debug(f"{call_id} ({meta.get('company_name', '')}): {len(segments)} segments")

    logger.info(
        f"Loaded {len(calls)} calls, "
        f"{sum(len(c.segments) for c in calls)} segments total"
    )
    return calls


def load_audio_segment(audio_path: Path, start_ts: float, end_ts: float) -> dict:
    """Load a time-sliced segment from an MP3 in Whisper pipeline format."""
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        offset=start_ts,
        duration=end_ts - start_ts,
        mono=True,
    )
    return {"raw": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE}
