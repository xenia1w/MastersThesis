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

_DEFAULT_SPEAKERS_FILE = Path("src/lexical_stylistic_prompting/data/speaker_selection/speakers_selected.txt")
_DEFAULT_OUTPUT_DIR    = Path("data/processed/lexical_stylistic_prompting/tedlium_selected")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Filter TED-LIUM to selected speakers and save to disk")
    parser.add_argument("--speakers-file", type=Path, default=_DEFAULT_SPEAKERS_FILE,
                        help="File with speaker IDs, one per line")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
                        help="Where to save the filtered dataset")
    parser.add_argument("--match-mode", choices=["exact", "base_name"], default="exact",
                        help="'exact' matches full talk IDs (e.g. AlGore_2006); "
                             "'base_name' strips the year suffix before matching")
    args = parser.parse_args()

    names = {line.strip() for line in args.speakers_file.read_text().splitlines() if line.strip()}
    logger.info(f"Filtering to {len(names)} speakers (match_mode={args.match_mode}) ...")

    logger.info("Loading full TED-LIUM dataset (train) ...")
    dataset = load_dataset(
        "distil-whisper/tedlium", "release3",
        split="train",
        trust_remote_code=True,
    )
    assert isinstance(dataset, Dataset)
    logger.info(f"Full dataset: {len(dataset):,} segments")

    if args.match_mode == "exact":
        filtered = dataset.filter(lambda ex: ex["speaker_id"] in names, desc="Filtering")
    else:
        filtered = dataset.filter(lambda ex: ex["speaker_id"].rsplit("_", 1)[0] in names, desc="Filtering")

    assert isinstance(filtered, Dataset)
    logger.info(f"Filtered dataset: {len(filtered):,} segments")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(str(args.output_dir))
    logger.info(f"Saved → {args.output_dir}")


if __name__ == "__main__":
    main()
