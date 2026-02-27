from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.pipeline.prosody_pipeline import ProsodyPipelineConfig, run_prosody_pipeline


def parse_args(argv: Sequence[str]) -> ProsodyPipelineConfig:
    parser = argparse.ArgumentParser(
        description="Run prosody feature extraction for L2-ARCTIC or SAA."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["l2arctic", "saa"],
        help="Dataset pipeline to run.",
    )
    parser.add_argument(
        "--outer-zip",
        default=None,
        help="Path to dataset zip archive (overrides default).",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/wavlm-base-plus-sv",
        help="Model identifier for feature extraction.",
    )
    parser.add_argument(
        "--save-root",
        default=None,
        help="Directory for saving embeddings (overrides default).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit for number of samples.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include missing SAA recordings when loading metadata.",
    )
    parser.add_argument(
        "--no-validate-files",
        action="store_false",
        dest="validate_files",
        help="Skip checking whether SAA recordings exist in the archive.",
    )
    parser.set_defaults(validate_files=True)

    parsed = parser.parse_args(list(argv))
    return ProsodyPipelineConfig(
        dataset=parsed.dataset,
        outer_zip=parsed.outer_zip,
        model_name=parsed.model_name,
        save_root=parsed.save_root,
        max_items=parsed.max_items,
        include_missing=parsed.include_missing,
        validate_files=parsed.validate_files,
    )


def main(argv: Sequence[str]) -> int:
    # Example: uv run python main.py --dataset l2arctic (or --dataset saa)
    # SAA with a quick limit: uv run python main.py --dataset saa --max-items 20
    config = parse_args(argv)
    run_prosody_pipeline(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
