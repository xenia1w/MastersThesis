"""
v2 local scoring: turn hand-annotated 5:00 / 15:00 boundaries into exact references,
then score the saved window hypotheses (baseline + each prompting approach).

The cluster jobs only saved hypotheses (transcriptions of the [5:00, 15:00] audio slice).
This script builds the ground-truth reference for that slice from your annotation CSV — for
each call you note the phrase heard at 5:00 and at 15:00; we locate each in the ordered .nlp
transcript to get word indices i5 / i15, and the reference is tokens[i5:i15]. Then we compute
per-call WER and entity-EER for every hypothesis and the baseline deltas.

Annotation CSV: one row per call. Columns are auto-detected (override with flags):
  - call id column   (name contains "call")
  - 5:00 phrase column   (header contains "5:00" but not "15:00")
  - 15:00 phrase column  (header contains "15:00")
`<unknown>` markers in a phrase are ignored; a run of a few ordinary consecutive words is
enough to locate the boundary.

Usage:
    uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_window_score \\
        --data-dir     data/raw/earnings21 \\
        --annotations  data/processed/lexical_stylistic_prompting/v2/annotations.csv \\
        --base-dir     data/processed/lexical_stylistic_prompting/v2 \\
        --approach     metadata_only --approach transcript_only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon

from src.asr_adaptation.metrics.wer import _normalize, compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    VOCABULARY_ENTITY_TYPES,
    Earnings21Token,
    _load_wer_tags,
    _parse_nlp_speaker_turns,
)
from src.lexical_stylistic_prompting.pipeline.earnings21_fullcall_eval import (
    _compute_entity_errors,
)

MIN_ANCHOR = 3          # shortest word-run we trust to locate a boundary
WINDOW_MINUTES = 10.0   # eval window length (5:00 -> 15:00), for the wpm sanity check
PLAUSIBLE_WPM = (80.0, 220.0)


# ── boundary location ─────────────────────────────────────────────────────────

def _norm_word(w: str) -> str:
    return w.lower().strip(".,;:!?\"'()[]")


def _phrase_words(phrase: str) -> list[str]:
    return [w for w in (_norm_word(x) for x in phrase.split()) if w and w != "<unknown>"]


def _find_all(tokens: list[str], sub: list[str]) -> list[int]:
    n = len(sub)
    return [i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == sub]


def locate_boundary(tokens_norm: list[str], phrase: str) -> tuple[int | None, str]:
    """Return (token index of the phrase's first word, status).

    Tries the longest contiguous sub-phrase that occurs uniquely; falls back to a shorter
    run (flagged). Status is one of ok / ambiguous / weak / not_found / empty.
    """
    words = _phrase_words(phrase)
    if not words:
        return None, "empty"

    n = len(words)
    for length in range(n, MIN_ANCHOR - 1, -1):
        for start in range(0, n - length + 1):
            occ = _find_all(tokens_norm, words[start:start + length])
            if len(occ) == 1:
                return max(occ[0] - start, 0), "ok"
    # nothing unique — take the first occurrence of the longest run we can find, flag it
    for length in range(min(n, MIN_ANCHOR), 1, -1):
        for start in range(0, n - length + 1):
            occ = _find_all(tokens_norm, words[start:start + length])
            if occ:
                return max(occ[0] - start, 0), ("ambiguous" if len(occ) > 1 else "weak")
    return None, "not_found"


# ── reference building ──────────────────────────────────────────────────────────

class Reference:
    def __init__(self, call_id: str, tokens: list[Earnings21Token], i5: int, i15: int,
                 status5: str, status15: str) -> None:
        self.call_id = call_id
        self.i5, self.i15 = i5, i15
        self.status5, self.status15 = status5, status15
        self.eval_tokens = tokens[i5:i15]
        self.profile_tokens = tokens[:i5]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.eval_tokens)

    @property
    def entity_mask(self) -> list[bool]:
        return [t.entity_type in VOCABULARY_ENTITY_TYPES for t in self.eval_tokens]

    @property
    def wpm(self) -> float:
        return len(self.eval_tokens) / WINDOW_MINUTES


def _call_tokens(data_dir: Path, call_id: str) -> list[Earnings21Token]:
    nlp = data_dir / "transcripts" / "nlp_references" / f"{call_id}.nlp"
    wer_tags = data_dir / "transcripts" / "wer_tags" / f"{call_id}.wer_tag.json"
    entity_map = _load_wer_tags(wer_tags) if wer_tags.exists() else {}
    return [t for _, turn in _parse_nlp_speaker_turns(nlp, entity_map) for t in turn]


def build_reference(data_dir: Path, call_id: str, phrase5: str, phrase15: str) -> Reference:
    tokens = _call_tokens(data_dir, call_id)
    norm = [_norm_word(t.text) for t in tokens]
    i5, s5 = locate_boundary(norm, phrase5)
    i15, s15 = locate_boundary(norm, phrase15)
    if i5 is None:
        i5, s5 = 0, s5 + "!"
    if i15 is None:
        i15, s15 = len(tokens), s15 + "!"
    if i15 <= i5:
        s15 += ";out-of-order"
    return Reference(call_id, tokens, i5, i15, s5, s15)


def _match_indices(tokens: list[Earnings21Token], eval_text: str) -> tuple[int, int] | None:
    """Recover (i5, i15) by exact consecutive-token match of a golden eval snippet.

    The golden snippet is a single-space join of consecutive token texts, so this is an
    exact (not fuzzy) match — it cannot land on the wrong repeated phrase the way the
    locator can. Returns None if the snippet is not found verbatim in the token stream.
    """
    target = eval_text.split()
    if not target:
        return None
    texts = [t.text for t in tokens]
    n = len(target)
    for i in range(len(texts) - n + 1):
        if texts[i:i + n] == target:
            return i, i + n
    return None


def build_reference_golden(data_dir: Path, call_id: str, eval_text: str) -> Reference | None:
    """Build a Reference from a hand-verified golden eval snippet (authoritative).

    Recovers token indices by exact match so entity tags are preserved for entity-EER.
    Returns None if the snippet cannot be matched verbatim (caller should surface this).
    """
    tokens = _call_tokens(data_dir, call_id)
    match = _match_indices(tokens, eval_text)
    if match is None:
        return None
    i5, i15 = match
    return Reference(call_id, tokens, i5, i15, "golden", "golden")


# ── annotation parsing ────────────────────────────────────────────────────────

class AnnotationColumns(NamedTuple):
    call: str
    start: str            # 5:00 phrase
    end: str              # 15:00 phrase
    ref_eval: str | None  # golden "Reference 5:00 - 15:00" snippet (if present)
    ref_prof: str | None  # golden "Reference 0:00 - 5:00" snippet (if present)


def _detect_columns(fieldnames: list[str]) -> AnnotationColumns:
    low = {c: c.lower() for c in fieldnames}
    ref_eval = next((c for c in fieldnames
                     if "reference" in low[c] and "5:00" in low[c] and "15:00" in low[c]), None)
    ref_prof = next((c for c in fieldnames
                     if "reference" in low[c] and "0:00" in low[c]), None)
    # the phrase columns are the ones that are NOT the golden reference columns
    end = next((c for c in fieldnames if "15:00" in low[c] and "reference" not in low[c]), None)
    start = next((c for c in fieldnames
                  if "5:00" in low[c] and "15:00" not in low[c] and "reference" not in low[c]), None)
    call = next((c for c in fieldnames if "call" in low[c]), fieldnames[0])
    if start is None or end is None:
        raise SystemExit(f"Could not auto-detect 5:00 / 15:00 columns in {fieldnames}. "
                         "Pass --call-col/--start-col/--end-col.")
    return AnnotationColumns(call, start, end, ref_eval, ref_prof)


def _sniff_delimiter(header_line: str) -> str:
    return ";" if header_line.count(";") >= header_line.count(",") else ","


class AnnotationRow(NamedTuple):
    call_id: str
    phrase5: str
    phrase15: str
    ref_eval: str  # golden eval snippet, "" if absent/blank
    ref_prof: str  # golden profile snippet, "" if absent/blank


def load_annotations(path: Path, call_col: str | None, start_col: str | None,
                     end_col: str | None) -> tuple[list[AnnotationRow], bool]:
    """Return (rows, golden_mode). golden_mode is True when the CSV carries a
    hand-verified 'Reference 5:00 - 15:00' column that should be treated as authoritative."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter=_sniff_delimiter(first))
        fields = list(reader.fieldnames or [])
        cols = _detect_columns(fields)
        if call_col and start_col and end_col:
            cols = cols._replace(call=call_col, start=start_col, end=end_col)
        golden_mode = cols.ref_eval is not None
        logger.info(f"Annotation columns → call={cols.call!r} 5:00={cols.start!r} 15:00={cols.end!r} "
                    f"golden_ref={cols.ref_eval!r} (golden_mode={golden_mode})")
        rows = []
        for r in reader:
            cid = str(r[cols.call]).strip()
            if cid:
                rows.append(AnnotationRow(
                    cid, str(r.get(cols.start, "")), str(r.get(cols.end, "")),
                    str(r.get(cols.ref_eval, "")).strip() if cols.ref_eval else "",
                    str(r.get(cols.ref_prof, "")).strip() if cols.ref_prof else "",
                ))
    return rows, golden_mode


# ── scoring ─────────────────────────────────────────────────────────────────────

def _load_hypotheses(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    df["call_id"] = df["call_id"].astype(str)
    return dict(zip(df["call_id"], df["hypothesis"].fillna("").astype(str)))


def score(refs: dict[str, Reference], hyps: dict[str, str]) -> pd.DataFrame:
    rows = []
    for cid, ref in refs.items():
        if cid not in hyps:
            continue
        hyp = hyps[cid]
        wer = compute_wer([ref.text], [hyp])
        n_err, n_tok = _compute_entity_errors(ref.text, hyp, ref.entity_mask)
        rows.append({
            "call_id": cid,
            "ref_words": len(ref.eval_tokens),
            "wer": round(wer, 4),
            "entity_eer": round(n_err / n_tok, 4) if n_tok else 0.0,
            "n_entity_tokens": n_tok,
            "n_entity_errors": n_err,
        })
    return pd.DataFrame(rows)


def _micro_macro(base: pd.DataFrame, appr: pd.DataFrame, refs: dict[str, Reference],
                 hyps_b: dict[str, str], hyps_a: dict[str, str], name: str) -> dict:
    m = base.merge(appr, on="call_id", suffixes=("_b", "_a"))
    all_ref = [refs[c].text for c in m["call_id"]]
    micro_b = compute_wer(all_ref, [hyps_b[c] for c in m["call_id"]])
    micro_a = compute_wer(all_ref, [hyps_a[c] for c in m["call_id"]])
    eer_b = m["n_entity_errors_b"].sum() / max(m["n_entity_tokens_b"].sum(), 1)
    eer_a = m["n_entity_errors_a"].sum() / max(m["n_entity_tokens_a"].sum(), 1)
    return {
        "approach": name, "n": len(m),
        "wer_base": round(micro_b, 4), "wer_prom": round(micro_a, 4),
        "wer_d_micro": round(micro_a - micro_b, 4),
        "wer_d_macro": round((m["wer_a"] - m["wer_b"]).mean(), 4),
        "eer_base": round(eer_b, 4), "eer_prom": round(eer_a, 4),
        "eer_d_micro": round(eer_a - eer_b, 4),
        "eer_d_macro": round((m["entity_eer_a"] - m["entity_eer_b"]).mean(), 4),
        "p_wer": _wilcoxon(m["wer_a"], m["wer_b"]),
        "p_eer": _wilcoxon(m["entity_eer_a"], m["entity_eer_b"]),
    }


def _wilcoxon(a: pd.Series, b: pd.Series) -> float:
    if (a - b).abs().sum() == 0:
        return 1.0
    return float(wilcoxon(a, b).pvalue)


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/earnings21"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path,
                        default=Path("data/processed/lexical_stylistic_prompting/v2"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="baseline_all.csv (default <base-dir>/earnings21_window_baseline/baseline_all.csv)")
    parser.add_argument("--approach", action="append", default=None,
                        help="approach name, repeatable (dir <base-dir>/earnings21_window_<name>)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--call-col", default=None)
    parser.add_argument("--start-col", default=None)
    parser.add_argument("--end-col", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.base_dir / "scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = args.baseline or (args.base_dir / "earnings21_window_baseline" / "baseline_all.csv")
    approaches = args.approach or ["metadata_only", "transcript_only",
                                   "transcript_plus_knowledge", "transcript_metadata_knowledge"]

    # 1. build references from annotations
    #    golden_mode: the CSV carries hand-verified "Reference 5:00 - 15:00" snippets — use them
    #    verbatim (authoritative); a blank snippet means the call is not ready and is skipped.
    #    Otherwise fall back to locating the boundaries from the 5:00/15:00 phrases.
    annos, golden_mode = load_annotations(args.annotations, args.call_col, args.start_col, args.end_col)
    refs: dict[str, Reference] = {}
    ref_rows = []
    for row in annos:
        cid = row.call_id
        if golden_mode:
            if not row.ref_eval:
                logger.warning(f"{cid}: golden reference blank — skipping (re-annotate to include)")
                continue
            ref = build_reference_golden(args.data_dir, cid, row.ref_eval)
            if ref is None:
                logger.error(f"{cid}: golden snippet not found verbatim in the .nlp tokens — "
                             f"skipping (check the snippet matches the reference token text)")
                continue
        else:
            ref = build_reference(args.data_dir, cid, row.phrase5, row.phrase15)
        refs[cid] = ref
        flag = "" if (PLAUSIBLE_WPM[0] <= ref.wpm <= PLAUSIBLE_WPM[1]
                      and "!" not in ref.status5 + ref.status15) else "  <-- CHECK"
        ref_rows.append({"call_id": cid, "i5": ref.i5, "i15": ref.i15,
                         "profile_words": len(ref.profile_tokens), "eval_words": len(ref.eval_tokens),
                         "wpm": round(ref.wpm, 1), "status_5": ref.status5, "status_15": ref.status15,
                         "reference_5_15": ref.text, "reference_0_5": " ".join(t.text for t in ref.profile_tokens)})
        logger.info(f"{cid}: i5={ref.i5} i15={ref.i15} eval_words={len(ref.eval_tokens)} "
                    f"wpm={ref.wpm:.0f} [{ref.status5}/{ref.status15}]{flag}")
    pd.DataFrame(ref_rows).to_csv(out_dir / "references.csv", index=False)
    logger.success(f"Wrote references → {out_dir / 'references.csv'} "
                   f"({sum('CHECK' in r for r in [str(x) for x in ref_rows])} flagged — inspect the wpm/status)")

    # 2. score baseline + each approach
    hyps_b = _load_hypotheses(baseline_csv)
    base_scores = score(refs, hyps_b)
    base_scores.to_csv(out_dir / "baseline_scored.csv", index=False)

    summary = []
    for name in approaches:
        appr_csv = args.base_dir / f"earnings21_window_{name}" / "prompted_all.csv"
        if not appr_csv.exists():
            logger.warning(f"{name}: no {appr_csv} — skipping")
            continue
        hyps_a = _load_hypotheses(appr_csv)
        appr_scores = score(refs, hyps_a)
        appr_scores.to_csv(out_dir / f"{name}_scored.csv", index=False)
        summary.append(_micro_macro(base_scores, appr_scores, refs, hyps_b, hyps_a, name))

    if summary:
        table = pd.DataFrame(summary)
        table.to_csv(out_dir / "cross_approach_summary.csv", index=False)
        logger.info("══ v2 cross-approach summary (baseline vs prompted) ══")
        for line in table.to_string(index=False).splitlines():
            logger.info("  " + line)
        logger.success(f"Wrote summary → {out_dir / 'cross_approach_summary.csv'}")


if __name__ == "__main__":
    main()
