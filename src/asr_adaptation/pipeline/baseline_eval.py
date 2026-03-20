from __future__ import annotations

import argparse
import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.asr_adaptation.data.l2arctic_transcriptions import (
    list_l2arctic_samples_with_transcripts,
)
from src.asr_adaptation.inference.transcribe import transcribe
from src.asr_adaptation.metrics.wer import compute_wer
from src.prosodic_feature_extraction.data.audio_utils import load_saa_mp3
from src.prosodic_feature_extraction.data.saa_utils import load_saa_samples

# The fixed passage all Speech Accent Archive speakers read aloud
SAA_REFERENCE = (
    "please call stella ask her to bring these things with her from the store "
    "six spoons of fresh snow peas five thick slabs of blue cheese and maybe a snack "
    "for her brother bob we also need a small plastic snake and a big toy frog for the kids "
    "she can scoop these things into three red bags and we can go meet her wednesday "
    "at the train station"
)

MODEL_NAME = "facebook/wav2vec2-base-960h"


@dataclass
class BaselineRow:
    speaker_id: str
    utterance_id: str | None
    native_language: str | None
    reference: str
    hypothesis: str
    wer: float


def _list_l2arctic_speakers(outer_zip_path: str) -> list[str]:
    with zipfile.ZipFile(outer_zip_path) as outer:
        return sorted(
            Path(name).stem
            for name in outer.namelist()
            if name.endswith(".zip")
        )


def run_l2arctic_baseline(
    l2arctic_zip: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
    speaker_filter: str | None = None,
) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    speakers = _list_l2arctic_speakers(l2arctic_zip)
    if speaker_filter:
        speakers = [s for s in speakers if s == speaker_filter]

    for speaker_id in speakers:
        logger.info(f"L2-ARCTIC | speaker {speaker_id}")
        samples = list_l2arctic_samples_with_transcripts(l2arctic_zip, speaker_id)

        for sample in samples:
            hypothesis = transcribe(sample.waveform, processor, model, device)
            wer = compute_wer([sample.transcript], [hypothesis])
            rows.append(
                BaselineRow(
                    speaker_id=speaker_id,
                    utterance_id=sample.utterance_id,
                    native_language=None,
                    reference=sample.transcript,
                    hypothesis=hypothesis,
                    wer=wer,
                )
            )

    return rows


def run_saa_baseline(
    saa_zip: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    samples = load_saa_samples(saa_zip)

    for sample in samples:
        logger.info(f"SAA | speaker {sample.speaker_id} ({sample.native_language})")
        waveform, _ = load_saa_mp3(saa_zip, sample.filename)
        if waveform.numel() < 400:
            logger.warning(f"SAA | speaker {sample.speaker_id} — audio too short ({waveform.numel()} samples), skipping")
            continue
        hypothesis = transcribe(waveform, processor, model, device)
        wer = compute_wer([SAA_REFERENCE], [hypothesis])
        rows.append(
            BaselineRow(
                speaker_id=sample.speaker_id,
                utterance_id=None,
                native_language=sample.native_language,
                reference=SAA_REFERENCE,
                hypothesis=hypothesis,
                wer=wer,
            )
        )

    return rows


def _save_csv(rows: list[BaselineRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker_id", "utterance_id", "native_language",
                        "reference", "hypothesis", "wer"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    speaker_id=row.speaker_id,
                    utterance_id=row.utterance_id or "",
                    native_language=row.native_language or "",
                    reference=row.reference,
                    hypothesis=row.hypothesis,
                    wer=round(row.wer, 4),
                )
            )
    logger.info(f"Saved {len(rows)} rows → {path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading {MODEL_NAME} ...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    torch.nn.Module.to(model, device)
    model.eval()

    output_dir = Path(args.output_dir)

    if args.l2arctic_zip:
        l2arctic_rows = run_l2arctic_baseline(
            args.l2arctic_zip, processor, model, device,
            speaker_filter=args.speaker,
        )
        _save_csv(l2arctic_rows, output_dir / "l2arctic_baseline.csv")

    if args.saa_zip:
        saa_rows = run_saa_baseline(args.saa_zip, processor, model, device)
        _save_csv(saa_rows, output_dir / "saa_baseline.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline WER evaluation (no adaptation)")
    parser.add_argument("--l2arctic-zip", default=None, help="Path to l2arctic_release_v5.0.zip")
    parser.add_argument("--saa-zip", default=None, help="Path to archive.zip (SAA)")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSVs")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model cache directory")
    parser.add_argument("--speaker", default=None, help="Evaluate a single L2-ARCTIC speaker only, e.g. ABA")
    main(parser.parse_args())
