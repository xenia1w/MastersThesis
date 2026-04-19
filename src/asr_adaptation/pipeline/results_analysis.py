"""
Analyse per-speaker LoRA adaptation results.

Usage:
    python -m src.asr_adaptation.pipeline.results_analysis \
        --results-dir data/processed/asr_adaptation/adaptation_results
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SpeakerStats:
    speaker: str
    n_train: int
    n_eval: int
    wer_baseline: float
    wer_adapted: float
    wer_delta: float
    pct_rel: float          # relative improvement (positive = better)
    n_improved: int
    n_same: int
    n_declined: int


def load_results(results_dir: Path) -> list[SpeakerStats]:
    stats = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv"):
            continue
        speaker = fname[:-4]
        rows = list(csv.DictReader(open(results_dir / fname)))
        wer_b = [float(r["wer_baseline"]) for r in rows]
        wer_a = [float(r["wer_adapted"]) for r in rows]
        baseline = sum(wer_b) / len(wer_b)
        adapted = sum(wer_a) / len(wer_a)
        delta = adapted - baseline
        pct_rel = -delta / baseline * 100  # positive = improvement
        stats.append(SpeakerStats(
            speaker=speaker,
            n_train=int(rows[0]["n_train"]),
            n_eval=len(rows),
            wer_baseline=baseline,
            wer_adapted=adapted,
            wer_delta=delta,
            pct_rel=pct_rel,
            n_improved=sum(1 for b, a in zip(wer_b, wer_a) if a < b),
            n_same=sum(1 for b, a in zip(wer_b, wer_a) if a == b),
            n_declined=sum(1 for b, a in zip(wer_b, wer_a) if a > b),
        ))
    return sorted(stats, key=lambda s: s.wer_delta)   # most improved first


def print_report(stats: list[SpeakerStats]) -> None:
    # ── per-speaker table ────────────────────────────────────────────────────
    col = "{:<6}  {:>7}  {:>10}  {:>9}  {:>9}  {:>8}  {:>9}  {:>9}  {:>9}"
    header = col.format(
        "Spkr", "n_train",
        "WER base", "WER adpt", "Δ WER",
        "Rel imp%",
        "↑ impr", "= same", "↓ decl",
    )
    sep = "-" * len(header)
    print(sep)
    print("  Per-speaker LoRA adaptation results  (sorted by Δ WER, best first)")
    print(sep)
    print(header)
    print(sep)
    for s in stats:
        arrow = "▲" if s.wer_delta < -0.05 else ("▼" if s.wer_delta > 0 else " ")
        print(col.format(
            s.speaker,
            s.n_train,
            f"{s.wer_baseline:.4f}",
            f"{s.wer_adapted:.4f}",
            f"{s.wer_delta:+.4f}",
            f"{s.pct_rel:+.1f}%",
            f"{s.n_improved}/100",
            f"{s.n_same}/100",
            f"{s.n_declined}/100",
        ) + f"  {arrow}")
    print(sep)

    # ── aggregate statistics ─────────────────────────────────────────────────
    deltas = [s.wer_delta for s in stats]
    pcts   = [s.pct_rel   for s in stats]
    baselines = [s.wer_baseline for s in stats]
    adapteds  = [s.wer_adapted  for s in stats]

    macro_baseline = sum(baselines) / len(baselines)
    macro_adapted  = sum(adapteds)  / len(adapteds)
    macro_delta    = macro_adapted - macro_baseline

    declined = [s for s in stats if s.wer_delta > 0]
    neutral  = [s for s in stats if abs(s.wer_delta) < 1e-4]
    improved = [s for s in stats if s.wer_delta < -1e-4]

    print()
    print("  Summary statistics")
    print(sep)
    print(f"  Speakers evaluated       : {len(stats)}")
    print(f"  Utterances per speaker   : 100  (held-out eval set)")
    print()
    print(f"  Macro-avg WER baseline   : {macro_baseline:.4f}")
    print(f"  Macro-avg WER adapted    : {macro_adapted:.4f}")
    print(f"  Macro-avg Δ WER          : {macro_delta:+.4f}")
    print(f"  Macro-avg rel. impr.     : {-macro_delta/macro_baseline*100:+.1f}%")
    print()
    print(f"  Mean   Δ WER             : {statistics.mean(deltas):+.4f}")
    print(f"  Median Δ WER             : {statistics.median(deltas):+.4f}")
    print(f"  Std    Δ WER             : {statistics.stdev(deltas):.4f}")
    print()
    print(f"  Mean   rel. improvement  : {statistics.mean(pcts):+.1f}%")
    print(f"  Median rel. improvement  : {statistics.median(pcts):+.1f}%")
    print()
    best  = min(stats, key=lambda s: s.wer_delta)
    worst = max(stats, key=lambda s: s.wer_delta)
    print(f"  Best   speaker           : {best.speaker}  (Δ {best.wer_delta:+.4f}, {best.pct_rel:+.1f}% rel)")
    print(f"  Worst  speaker           : {worst.speaker}  (Δ {worst.wer_delta:+.4f}, {worst.pct_rel:+.1f}% rel)")
    print()
    print(f"  Speakers with improvement: {len(improved)} / {len(stats)}")
    print(f"  Speakers neutral (≈0)    : {len(neutral)} / {len(stats)}")
    print(f"  Speakers with decline    : {len(declined)} / {len(stats)}")

    if declined:
        print()
        print("  Speakers where WER got worse:")
        for s in declined:
            print(f"    {s.speaker}  Δ {s.wer_delta:+.4f}  ({s.pct_rel:+.1f}%)")

    # ── utterance-level decline rate ─────────────────────────────────────────
    total_improved = sum(s.n_improved for s in stats)
    total_same     = sum(s.n_same     for s in stats)
    total_declined = sum(s.n_declined for s in stats)
    total_utts     = total_improved + total_same + total_declined
    print()
    print(f"  Utterance-level breakdown (all speakers, {total_utts} total):")
    print(f"    Improved : {total_improved:4d}  ({total_improved/total_utts*100:.1f}%)")
    print(f"    Same     : {total_same:4d}  ({total_same/total_utts*100:.1f}%)")
    print(f"    Declined : {total_declined:4d}  ({total_declined/total_utts*100:.1f}%)")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse LoRA adaptation WER results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/processed/asr_adaptation/adaptation_results"),
    )
    args = parser.parse_args()
    stats = load_results(args.results_dir)
    print_report(stats)


if __name__ == "__main__":
    main()
