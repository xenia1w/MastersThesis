from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SPEAKER_COLORS = {
    "RRBI": "#2A6F97",
    "ERMS": "#BC4749",
    "LXC":  "#6A994E",
    "HQTV": "#E07B39",
}


def _load_results(results_dir: Path) -> Dict[str, List[dict]]:
    by_speaker: Dict[str, List[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*_seed*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_speaker[row["speaker_id"]].append(
                    {
                        "n_train":     int(row["n_train"]),
                        "seed":        int(row["seed"]),
                        "wer_baseline": float(row["wer_baseline"]),
                        "wer_adapted":  float(row["wer_adapted"]),
                        "wer_delta":    float(row["wer_delta"]),
                    }
                )
    for rows in by_speaker.values():
        rows.sort(key=lambda r: (r["n_train"], r["seed"]))
    return dict(by_speaker)


def _aggregate(rows: List[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_n: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_n[r["n_train"]].append(r)

    ns = np.array(sorted(by_n))
    baseline = np.array([np.mean([r["wer_baseline"] for r in by_n[n]]) for n in ns])
    adapted  = np.array([np.mean([r["wer_adapted"]  for r in by_n[n]]) for n in ns])
    delta    = np.array([np.mean([r["wer_delta"]    for r in by_n[n]]) for n in ns])
    return ns, baseline, adapted, delta


def plot(results_dir: Path, output_path: Path) -> None:
    data = _load_results(results_dir)
    if not data:
        raise FileNotFoundError(f"No result CSVs found in {results_dir}")

    fig, (ax_wer, ax_delta) = plt.subplots(1, 2, figsize=(11, 4.5))

    for speaker, rows in sorted(data.items()):
        ns, baseline, adapted, delta = _aggregate(rows)
        color = SPEAKER_COLORS.get(speaker, "#888888")

        ax_wer.axhline(baseline.mean(), color=color, linewidth=1, linestyle="--", alpha=0.5)
        ax_wer.plot(ns, adapted, marker="o", color=color, label=speaker)

        ax_delta.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_delta.plot(ns, delta, marker="o", color=color, label=speaker)

    ax_wer.set_xlabel("Training utterances per speaker")
    ax_wer.set_ylabel("WER")
    ax_wer.set_title("Adapted WER vs training size")
    ax_wer.set_xscale("log")
    ax_wer.set_xticks(ns)
    ax_wer.set_xticklabels(ns)
    ax_wer.legend(title="Speaker")

    ax_delta.set_xlabel("Training utterances per speaker")
    ax_delta.set_ylabel("WER delta (adapted − baseline)")
    ax_delta.set_title("WER improvement vs training size")
    ax_delta.set_xscale("log")
    ax_delta.set_xticks(ns)
    ax_delta.set_xticklabels(ns)
    ax_delta.legend(title="Speaker")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WER vs training size sweep")
    parser.add_argument(
        "--results-dir",
        default="data/processed/asr_adaptation/data_size_curves",
        help="Directory containing individual *_seed*.csv files",
    )
    parser.add_argument(
        "--output",
        default="src/asr_adaptation/plots/data_size_curves.png",
        help="Output image path",
    )
    args = parser.parse_args()
    plot(Path(args.results_dir), Path(args.output))
