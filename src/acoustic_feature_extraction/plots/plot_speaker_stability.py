from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _group_curves(
    rows: Iterable[Dict[str, str]],
    metric: str = "cosine_to_full",
) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    grouped: DefaultDict[str, DefaultDict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        representation = row["representation"]
        speaker_id = row["speaker_id"]
        seconds = float(row["cumulative_seconds"])
        raw = row.get(metric, "")
        if not raw or raw.lower() in ("none", "nan"):
            continue
        cosine = float(raw)
        grouped[representation][speaker_id].append((seconds, cosine))

    out: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for representation, by_speaker in grouped.items():
        out[representation] = {}
        for speaker_id, curve in by_speaker.items():
            out[representation][speaker_id] = sorted(curve, key=lambda x: x[0])
    return out


def _compute_stability_means(
    rows: List[Dict[str, str]],
    col: str,
) -> Dict[str, float]:
    sums: DefaultDict[str, List[float]] = defaultdict(list)
    for row in rows:
        val = row.get(col, "")
        if val and val.lower() not in ("", "none", "nan"):
            sums[row["representation"]].append(float(val))
    return {rep: sum(vs) / len(vs) for rep, vs in sums.items() if vs}


def _aggregate_stats(
    curves: Dict[str, List[Tuple[float, float]]],
    max_seconds: float | None,
) -> List[Tuple[float, float, float]]:
    observed_max_seconds = max(
        (points[-1][0] for points in curves.values() if points),
        default=0.0,
    )
    if observed_max_seconds <= 0.0:
        return []
    target_max_seconds = observed_max_seconds
    if max_seconds is not None and max_seconds > 0.0:
        target_max_seconds = min(observed_max_seconds, max_seconds)
    grid = np.linspace(0.0, target_max_seconds, 120, dtype=np.float64)
    interpolated: List[np.ndarray] = []
    for points in curves.values():
        if not points:
            continue
        xs = np.array([x for x, _ in points], dtype=np.float64)
        ys = np.array([y for _, y in points], dtype=np.float64)
        interpolated.append(np.interp(grid, xs, ys, left=ys[0], right=ys[-1]))
    if not interpolated:
        return []

    stacked = np.vstack(interpolated)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0, ddof=0)
    return [
        (float(grid[i]), float(means[i]), float(stds[i])) for i in range(grid.shape[0])
    ]


def _y_limits(
    curves: Dict[str, List[Tuple[float, float]]],
    max_seconds: float | None = None,
) -> tuple[float, float]:
    values: List[float] = []
    for points in curves.values():
        for seconds, cosine in points:
            if max_seconds is not None and seconds > max_seconds:
                continue
            values.append(cosine)
    if not values:
        return (0.0, 1.0)
    min_value = min(values)
    max_value = max(values)
    pad = max(0.005, (max_value - min_value) * 0.15)
    y_min = max(0.0, min_value - pad)
    y_max = min(1.01, max_value + pad)
    if y_max - y_min < 0.02:
        y_min = max(0.0, y_min - 0.01)
        y_max = min(1.01, y_max + 0.01)
    return y_min, y_max


def _save_plot(
    fig: plt.Figure,
    out_stem: Path,
    formats: List[str],
) -> None:
    for fmt in formats:
        fig.savefig(out_stem.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_aggregate_csv(
    out_path: Path,
    stats_by_representation: Dict[str, List[Tuple[float, float, float]]],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["representation", "seconds", "cosine_mean", "cosine_std"],
        )
        writer.writeheader()
        for representation, stats in sorted(stats_by_representation.items()):
            for seconds, mean_value, std_value in stats:
                writer.writerow(
                    {
                        "representation": representation,
                        "seconds": seconds,
                        "cosine_mean": mean_value,
                        "cosine_std": std_value,
                    }
                )


def _plot_representation_curves(
    representation: str,
    curves: Dict[str, List[Tuple[float, float]]],
    stats: List[Tuple[float, float, float]],
    out_dir: Path,
    formats: List[str],
    x_max_seconds: float,
    file_suffix: str = "",
    stability_mean: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, points in curves.items():
        ks = [k for k, _ in points]
        cosines = [c for _, c in points]
        ax.plot(ks, cosines, color="#9aa0a6", alpha=0.25, linewidth=1.0)

    ks = [k for k, _, _ in stats]
    means = [m for _, m, _ in stats]
    stds = [s for _, _, s in stats]
    lower = np.clip(np.array(means) - np.array(stds), 0.0, 1.0)
    upper = np.clip(np.array(means) + np.array(stds), 0.0, 1.0)
    ax.plot(ks, means, color="#1f77b4", linewidth=2.4, label="Mean")
    ax.fill_between(
        ks,
        lower,
        upper,
        color="#1f77b4",
        alpha=0.2,
        label="Mean ± Std",
    )

    if stability_mean is not None:
        ax.axvline(
            stability_mean,
            color="#e05c5c",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean stability ≈ {stability_mean:.1f}s",
        )

    ax.set_title(f"Embedding Stability by Speaker: {representation}")
    ax.set_xlabel("Cumulative Speech (seconds)")
    ax.set_ylabel("Cosine Similarity to E_full")
    ax.set_xlim(0.0, x_max_seconds)
    y_min, y_max = _y_limits(curves, max_seconds=x_max_seconds)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    stem = f"stability_{_sanitize_filename(representation)}{file_suffix}"
    file_stem = out_dir / stem
    _save_plot(fig, file_stem, formats)


def _plot_aggregate_comparison(
    stats_by_representation: Dict[str, List[Tuple[float, float, float]]],
    curves_by_representation: Dict[str, Dict[str, List[Tuple[float, float]]]],
    out_dir: Path,
    formats: List[str],
    x_max_seconds: float,
    file_stem_name: str = "stability_aggregate_mean_std",
    stability_means: Optional[Dict[str, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    all_curves: Dict[str, List[Tuple[float, float]]] = {}
    rep_colors: Dict[str, str] = {}
    for representation, stats in sorted(stats_by_representation.items()):
        if not stats:
            continue
        ks = [k for k, _, _ in stats]
        means = np.array([m for _, m, _ in stats], dtype=np.float64)
        stds = np.array([s for _, _, s in stats], dtype=np.float64)
        lower = np.clip(means - stds, 0.0, 1.0)
        upper = np.clip(means + stds, 0.0, 1.0)
        (line,) = ax.plot(ks, means, linewidth=2.3, label=f"{representation} mean")
        rep_colors[representation] = str(line.get_color())
        ax.fill_between(ks, lower, upper, alpha=0.18, color=line.get_color())
        for speaker_id, points in curves_by_representation[representation].items():
            all_curves[f"{representation}:{speaker_id}"] = points

    if stability_means:
        for representation, mean_s in sorted(stability_means.items()):
            color = rep_colors.get(representation, "#888888")
            ax.axvline(
                mean_s,
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{representation} stability ≈ {mean_s:.1f}s",
            )

    ax.set_title("Embedding Stability: Aggregate Mean ± Std")
    ax.set_xlabel("Cumulative Speech (seconds)")
    ax.set_ylabel("Cosine Similarity to E_full")
    ax.set_xlim(0.0, x_max_seconds)
    y_min, y_max = _y_limits(all_curves, max_seconds=x_max_seconds)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    file_stem = out_dir / file_stem_name
    _save_plot(fig, file_stem, formats)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot speaker stability curves.")
    parser.add_argument(
        "--csv-path",
        default="data/processed/stability/l2arctic_speaker_stability/speaker_stability_all.csv",
        help="Path to aggregated speaker stability CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots. Defaults to <csv_dir>/plots.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--x-max-seconds",
        type=float,
        default=30.0,
        help="Maximum x-axis value in seconds.",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = _load_rows(csv_path)
    dataset = rows[0]["dataset"] if rows else "unknown"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("src") / "acoustic_feature_extraction" / "plots" / dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lower() for fmt in str(args.formats).split(",") if fmt.strip()]
    x_max_seconds = float(args.x_max_seconds)

    approaches = [
        {
            "metric": "cosine_to_full",
            "stability_col": "stability_seconds",
            "file_suffix": "",
            "aggregate_stem": "stability_aggregate_mean_std",
        },
        {
            "metric": "cosine_consecutive",
            "stability_col": "stability_consecutive_seconds",
            "file_suffix": "_consecutive",
            "aggregate_stem": "stability_aggregate_consecutive",
        },
    ]

    for approach in approaches:
        curves_by_rep = _group_curves(rows, metric=approach["metric"])
        stability_means = _compute_stability_means(rows, approach["stability_col"])
        stats_by_rep: Dict[str, List[Tuple[float, float, float]]] = {}
        for representation, curves in curves_by_rep.items():
            stats = _aggregate_stats(curves, max_seconds=x_max_seconds)
            stats_by_rep[representation] = stats
            _plot_representation_curves(
                representation=representation,
                curves=curves,
                stats=stats,
                out_dir=out_dir,
                formats=formats,
                x_max_seconds=x_max_seconds,
                file_suffix=approach["file_suffix"],
                stability_mean=stability_means.get(representation),
            )

        _plot_aggregate_comparison(
            stats_by_representation=stats_by_rep,
            curves_by_representation=curves_by_rep,
            out_dir=out_dir,
            formats=formats,
            x_max_seconds=x_max_seconds,
            file_stem_name=approach["aggregate_stem"],
            stability_means=stability_means,
        )
        if approach["file_suffix"] == "":
            _write_aggregate_csv(out_dir / "stability_aggregate_stats.csv", stats_by_rep)

    print(f"Saved plots and aggregate stats to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
