"""
One-time script: filter TED-LIUM to the selected speakers and save to disk.

Run once after speakers_selected.txt is finalised:
    uv run src/lexical_stylistic_prompting/data/prepare_dataset.py

Produces data/processed/lexical_stylistic_prompting/tedlium_selected/
which baseline_eval.py loads directly on every subsequent run.
"""

from pathlib import Path

from datasets import Dataset, load_dataset
from loguru import logger

SPEAKERS_FILE = Path("src/lexical_stylistic_prompting/data/speaker_selection/speakers_selected.txt")
OUTPUT_DIR    = Path("data/processed/lexical_stylistic_prompting/tedlium_selected")


def main() -> None:
    selected = {
        line.strip()
        for line in SPEAKERS_FILE.read_text().splitlines()
        if line.strip()
    }
    logger.info(f"Filtering to {len(selected)} selected speakers ...")

    logger.info("Loading full TED-LIUM dataset (train) ...")
    dataset = load_dataset(
        "distil-whisper/tedlium", "release3",
        split="train",
        trust_remote_code=True,
    )
    assert isinstance(dataset, Dataset)
    logger.info(f"Full dataset: {len(dataset):,} segments")

    filtered = dataset.filter(
        lambda ex: ex["speaker_id"] in selected,
        desc="Filtering",
    )
    assert isinstance(filtered, Dataset)
    logger.info(f"Filtered dataset: {len(filtered):,} segments")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(str(OUTPUT_DIR))
    logger.info(f"Saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
