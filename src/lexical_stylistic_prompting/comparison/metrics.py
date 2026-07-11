"""WER / entity-EER aggregation over a baseline-vs-prompted joined frame.

A joined frame (see ``loaders.join_to_baseline``) has ``base_*`` and ``prom_*`` columns
and one row per common call. Two averaging conventions are reported everywhere:

  * micro (corpus) — Σerrors / Σwords, so long calls count more (the standard ASR number);
  * macro          — the mean of per-call values, so every call counts equally.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon


def reference_mismatches(merged: pd.DataFrame) -> int:
    """Calls whose baseline/prompted reference differs — must be 0 for apples-to-apples."""
    return int((merged["base_reference_words"] != merged["prom_reference_words"]).sum())


def micro_wer(df: pd.DataFrame, wer_col: str, words_col: str) -> float:
    total = df[words_col].sum()
    return float((df[wer_col] * df[words_col]).sum() / total) if total else 0.0


def micro_eer(df: pd.DataFrame, err_col: str, tok_col: str) -> float:
    total = df[tok_col].sum()
    return float(df[err_col].sum() / total) if total else 0.0


def macro_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean())


def per_call_table(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per call: baseline/prompted WER and EER with their deltas."""
    out = pd.DataFrame({
        "call_id": merged["call_id"],
        "base_wer": merged["base_wer"],
        "prom_wer": merged["prom_wer"],
        "d_wer": merged["prom_wer"] - merged["base_wer"],
        "base_eer": merged["base_entity_eer"],
        "prom_eer": merged["prom_entity_eer"],
        "d_eer": merged["prom_entity_eer"] - merged["base_entity_eer"],
    })
    return out.sort_values("d_eer").reset_index(drop=True)


def paired_pvalue(merged: pd.DataFrame, base_col: str, prom_col: str) -> float:
    """Wilcoxon signed-rank p-value for the paired per-call metric (1.0 if all-equal)."""
    if (merged[base_col] == merged[prom_col]).all():
        return 1.0
    return float(wilcoxon(merged[prom_col], merged[base_col]).pvalue)


def summary_row(merged: pd.DataFrame, approach: str) -> dict[str, float | int | str]:
    """One flat row of baseline/prompted WER+EER (micro & macro deltas) for a table."""
    bw, pw = micro_wer(merged, "base_wer", "base_reference_words"), micro_wer(merged, "prom_wer", "prom_reference_words")
    be, pe = micro_eer(merged, "base_n_entity_errors", "base_n_entity_tokens"), micro_eer(merged, "prom_n_entity_errors", "prom_n_entity_tokens")
    return {
        "approach": approach,
        "n": len(merged),
        "wer_base": round(bw, 4),
        "wer_prom": round(pw, 4),
        "wer_d_micro": round(pw - bw, 4),
        "wer_d_macro": round(macro_mean(merged, "prom_wer") - macro_mean(merged, "base_wer"), 4),
        "eer_base": round(be, 4),
        "eer_prom": round(pe, 4),
        "eer_d_micro": round(pe - be, 4),
        "eer_d_macro": round(macro_mean(merged, "prom_entity_eer") - macro_mean(merged, "base_entity_eer"), 4),
        "p_wer": round(paired_pvalue(merged, "base_wer", "prom_wer"), 4),
        "p_eer": round(paired_pvalue(merged, "base_entity_eer", "prom_entity_eer"), 4),
    }


def summarize(merged: pd.DataFrame, label: str) -> dict[str, float]:
    """Log micro + macro WER/EER deltas and paired p-values; return them as a dict."""
    bw, pw = micro_wer(merged, "base_wer", "base_reference_words"), micro_wer(merged, "prom_wer", "prom_reference_words")
    be, pe = micro_eer(merged, "base_n_entity_errors", "base_n_entity_tokens"), micro_eer(merged, "prom_n_entity_errors", "prom_n_entity_tokens")
    mac_bw, mac_pw = macro_mean(merged, "base_wer"), macro_mean(merged, "prom_wer")
    mac_be, mac_pe = macro_mean(merged, "base_entity_eer"), macro_mean(merged, "prom_entity_eer")
    p_wer = paired_pvalue(merged, "base_wer", "prom_wer")
    p_eer = paired_pvalue(merged, "base_entity_eer", "prom_entity_eer")

    logger.info(f"── {label} ({len(merged)} calls) ──")
    logger.info(f"  WER micro : {bw:.4f} → {pw:.4f}  (Δ {pw - bw:+.4f})   macro : {mac_bw:.4f} → {mac_pw:.4f}  (Δ {mac_pw - mac_bw:+.4f})")
    logger.info(f"  EER micro : {be:.4f} → {pe:.4f}  (Δ {pe - be:+.4f})   macro : {mac_be:.4f} → {mac_pe:.4f}  (Δ {mac_pe - mac_be:+.4f})")
    logger.info(f"  paired Wilcoxon p: WER={p_wer:.4f}  EER={p_eer:.4f}")
    return {"micro_d_wer": pw - bw, "macro_d_wer": mac_pw - mac_bw,
            "micro_d_eer": pe - be, "macro_d_eer": mac_pe - mac_be,
            "p_wer": p_wer, "p_eer": p_eer}
