from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

PERTURBATION_ORDER = ["rate_p10", "rate_m10", "pitch_p2st", "pitch_m2st", "pause_ins"]
METRIC_COLUMNS = [
    ("cosine_base_mean", "Base WavLM (mean)", "#2A6F97"),
    ("cosine_base_meanstd", "Base WavLM (mean+std)", "#6A994E"),
    ("cosine_sv_xvector", "SV x-vector", "#BC4749"),
]


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def compute_metric_stats(
    rows: Iterable[Dict[str, str]],
) -> Dict[str, Dict[str, tuple[float, float, int]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        perturbation = row["perturbation_type"]
        grouped.setdefault(perturbation, {})
        for metric_name, _, _ in METRIC_COLUMNS:
            grouped[perturbation].setdefault(metric_name, [])
            grouped[perturbation][metric_name].append(float(row[metric_name]))

    stats: Dict[str, Dict[str, tuple[float, float, int]]] = {}
    for perturbation, by_metric in grouped.items():
        stats[perturbation] = {}
        for metric_name, values in by_metric.items():
            arr = np.asarray(values, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            stats[perturbation][metric_name] = (mean, std, int(arr.shape[0]))
    return stats


def _ordered_perturbations(stats: Dict[str, Dict[str, tuple[float, float, int]]]) -> List[str]:
    ordered = [name for name in PERTURBATION_ORDER if name in stats]
    for name in sorted(stats.keys()):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _save_plot(fig: plt.Figure, out_stem: Path, formats: List[str]) -> None:
    for fmt in formats:
        fig.savefig(out_stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_stats_csv(
    out_path: Path,
    stats: Dict[str, Dict[str, tuple[float, float, int]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["perturbation_type", "embedding_type", "mean_cosine", "std_cosine", "n"],
        )
        writer.writeheader()
        for perturbation in _ordered_perturbations(stats):
            for metric_name, metric_label, _ in METRIC_COLUMNS:
                mean, std, n = stats[perturbation][metric_name]
                writer.writerow(
                    {
                        "perturbation_type": perturbation,
                        "embedding_type": metric_label,
                        "mean_cosine": mean,
                        "std_cosine": std,
                        "n": n,
                    }
                )


def _plot_grouped_bars(
    dataset: str,
    stats: Dict[str, Dict[str, tuple[float, float, int]]],
    out_dir: Path,
    formats: List[str],
) -> None:
    perturbations = _ordered_perturbations(stats)
    x = np.arange(len(perturbations), dtype=np.float64)
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for idx, (metric_name, label, color) in enumerate(METRIC_COLUMNS):
        means = [stats[p][metric_name][0] for p in perturbations]
        stds = [stats[p][metric_name][1] for p in perturbations]
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            label=label,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            yerr=stds,
            capsize=4,
            error_kw={"elinewidth": 1.1, "alpha": 0.95},
        )

    ax.set_title(f"Perturbation Sensitivity by Embedding Type ({dataset})")
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("Cosine Similarity (orig vs perturbed)")
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=20, ha="right")
    ax.set_ylim(0.5, 1.01)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    out_stem = out_dir / "sensitivity_grouped_mean_std"
    _save_plot(fig, out_stem, formats=formats)


def _plot_deltas(
    dataset: str,
    rows: List[Dict[str, str]],
    out_dir: Path,
    formats: List[str],
) -> None:
    by_perturbation: Dict[str, List[float]] = {}
    for row in rows:
        perturbation = row["perturbation_type"]
        by_perturbation.setdefault(perturbation, [])
        by_perturbation[perturbation].append(float(row["delta_xvector_minus_meanstd"]))

    perturbations = _ordered_perturbations(
        {k: {"delta_xvector_minus_meanstd": (0.0, 0.0, 0)} for k in by_perturbation}
    )
    means = [float(np.mean(np.asarray(by_perturbation[p], dtype=np.float64))) for p in perturbations]
    stds = [float(np.std(np.asarray(by_perturbation[p], dtype=np.float64), ddof=0)) for p in perturbations]

    x = np.arange(len(perturbations), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(
        x,
        means,
        color="#5E548E",
        alpha=0.9,
        yerr=stds,
        capsize=4,
        error_kw={"elinewidth": 1.1, "alpha": 0.95},
    )
    ax.axhline(0.0, color="#2b2d42", linewidth=1.0, alpha=0.7)
    ax.set_title(f"SV x-vector Delta vs Base mean+std ({dataset})")
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("Delta Cosine (x-vector - mean+std)")
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    out_stem = out_dir / "sensitivity_delta_xvector_vs_meanstd"
    _save_plot(fig, out_stem, formats=formats)


def _plot_single_dataset(
    dataset: str,
    sensitivity_root: Path,
    out_root: Path,
    formats: List[str],
) -> None:
    csv_path = sensitivity_root / dataset / "sensitivity_detail.csv"
    rows = _load_rows(csv_path)
    stats = compute_metric_stats(rows)

    dataset_out = out_root / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    _plot_grouped_bars(dataset=dataset, stats=stats, out_dir=dataset_out, formats=formats)
    _plot_deltas(dataset=dataset, rows=rows, out_dir=dataset_out, formats=formats)
    _write_stats_csv(dataset_out / "sensitivity_plot_stats.csv", stats=stats)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot perturbation sensitivity results.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["l2arctic", "saa", "all"],
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--sensitivity-root",
        default="data/processed/perturbation_sensitivity",
        help="Root directory containing sensitivity CSV outputs.",
    )
    parser.add_argument(
        "--out-root",
        default="src/acoustic_feature_extraction/plots/perturbation_sensitivity",
        help="Output root for plot figures.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    datasets = ["l2arctic", "saa"] if args.dataset == "all" else [str(args.dataset)]
    formats = [fmt.strip().lower() for fmt in str(args.formats).split(",") if fmt.strip()]
    sensitivity_root = Path(args.sensitivity_root)
    out_root = Path(args.out_root)

    for dataset in datasets:
        _plot_single_dataset(
            dataset=dataset,
            sensitivity_root=sensitivity_root,
            out_root=out_root,
            formats=formats,
        )

    print(f"Saved perturbation sensitivity plots to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))

