from __future__ import annotations

from src.acoustic_feature_extraction.plots.plot_perturbation_sensitivity import (
    compute_metric_stats,
)


def test_compute_metric_stats() -> None:
    rows = [
        {
            "perturbation_type": "rate_p10",
            "cosine_base_mean": "0.8",
            "cosine_base_meanstd": "0.9",
            "cosine_sv_xvector": "0.95",
        },
        {
            "perturbation_type": "rate_p10",
            "cosine_base_mean": "0.9",
            "cosine_base_meanstd": "1.0",
            "cosine_sv_xvector": "0.85",
        },
    ]
    stats = compute_metric_stats(rows)
    mean_value, std_value, n = stats["rate_p10"]["cosine_base_mean"]
    assert abs(mean_value - 0.85) < 1e-6
    assert abs(std_value - 0.05) < 1e-6
    assert n == 2

