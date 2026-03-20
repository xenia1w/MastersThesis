from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from src.asr_adaptation.pipeline.data_size_analysis import (
    N_VALUES,
    SEEDS,
    merge_results,
    run_data_size_single,
)
from src.asr_adaptation.pipeline.lora_train import AdaptationRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(n: int, wer_baseline: float, wer_adapted: float) -> list[AdaptationRow]:
    return [
        AdaptationRow(
            speaker_id="ABA",
            utterance_id=f"arctic_a{i:04d}",
            n_train=n,
            reference="hello world",
            hypothesis_baseline="hello there",
            hypothesis_adapted="hello world",
            wer_baseline=wer_baseline,
            wer_adapted=wer_adapted,
        )
        for i in range(10)
    ]


def _write_individual_csv(
    path: Path,
    speaker_id: str,
    n_train: int,
    seed: int,
    wer_baseline: float,
    wer_adapted: float,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["speaker_id", "n_train", "seed",
                           "wer_baseline", "wer_adapted", "wer_delta"]
        )
        writer.writeheader()
        writer.writerow(dict(
            speaker_id=speaker_id,
            n_train=n_train,
            seed=seed,
            wer_baseline=round(wer_baseline, 4),
            wer_adapted=round(wer_adapted, 4),
            wer_delta=round(wer_adapted - wer_baseline, 4),
        ))


# ---------------------------------------------------------------------------
# Tests: run_data_size_single
# ---------------------------------------------------------------------------

def test_run_data_size_single_creates_csv(tmp_path: Path) -> None:
    with patch(
        "src.asr_adaptation.pipeline.data_size_analysis.run_lora_train",
        return_value=_make_rows(10, wer_baseline=0.4, wer_adapted=0.2),
    ):
        out = run_data_size_single(
            speaker_id="ABA",
            n_train=10,
            seed=0,
            l2arctic_zip="dummy.zip",
            output_dir=tmp_path,
        )

    assert out.exists()
    assert out.name == "ABA_0010_seed0.csv"


def test_run_data_size_single_csv_content(tmp_path: Path) -> None:
    with patch(
        "src.asr_adaptation.pipeline.data_size_analysis.run_lora_train",
        return_value=_make_rows(10, wer_baseline=0.4, wer_adapted=0.2),
    ):
        out = run_data_size_single("ABA", 10, 0, "dummy.zip", tmp_path)

    with open(out, newline="") as f:
        row = list(csv.DictReader(f))[0]

    assert row["speaker_id"] == "ABA"
    assert int(row["n_train"]) == 10
    assert int(row["seed"]) == 0
    assert float(row["wer_baseline"]) == pytest.approx(0.4)
    assert float(row["wer_adapted"]) == pytest.approx(0.2)
    assert float(row["wer_delta"]) == pytest.approx(-0.2)


def test_run_data_size_single_passes_seed_to_lora_train(tmp_path: Path) -> None:
    with patch(
        "src.asr_adaptation.pipeline.data_size_analysis.run_lora_train",
        return_value=_make_rows(5, 0.3, 0.1),
    ) as mock_train:
        run_data_size_single("ABA", 5, 2, "dummy.zip", tmp_path)

    call_kwargs = mock_train.call_args.kwargs
    assert call_kwargs["seed"] == 2
    assert call_kwargs["n_train"] == 5


# ---------------------------------------------------------------------------
# Tests: merge_results
# ---------------------------------------------------------------------------

def test_merge_results_combines_all_csvs(tmp_path: Path) -> None:
    for n in [10, 20]:
        for seed in [0, 1]:
            _write_individual_csv(
                tmp_path / f"ABA_{n:04d}_seed{seed}.csv",
                "ABA", n, seed, 0.4, 0.2,
            )

    merge_results(tmp_path)

    merged = tmp_path / "ABA_wer_vs_n.csv"
    assert merged.exists()

    with open(merged, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4  # 2 n_values × 2 seeds


def test_merge_results_sorted_by_n_then_seed(tmp_path: Path) -> None:
    for n, seed in [(20, 1), (10, 0), (20, 0), (10, 1)]:
        _write_individual_csv(
            tmp_path / f"ABA_{n:04d}_seed{seed}.csv",
            "ABA", n, seed, 0.4, 0.2,
        )

    merge_results(tmp_path)

    with open(tmp_path / "ABA_wer_vs_n.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    n_seed_pairs = [(int(r["n_train"]), int(r["seed"])) for r in rows]
    assert n_seed_pairs == sorted(n_seed_pairs)


def test_merge_results_raises_if_no_csvs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        merge_results(tmp_path)


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

def test_n_values_covers_expected_range() -> None:
    assert N_VALUES == [1, 5, 10, 20, 50, 100, 200]


def test_seeds_has_three_values() -> None:
    assert len(SEEDS) == 3
