"""Wrong-speaker control experiment for FiLM-conditioned LoRA.

Loads a pre-trained FiLM+LoRA checkpoint for speaker A and re-transcribes their
existing eval set using a different speaker B's centroid.  The correct-speaker
results are read directly from the already-saved film_adaptation_results CSV,
so only one new inference pass (with the wrong centroid) is needed.

If FiLM has learned to use speaker-specific content, WER should increase when
the wrong centroid is injected.  If WER is unchanged, FiLM helps structurally
only and does not encode speaker-specific information.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel, computed_field

from src.asr_adaptation.data.l2arctic_transcriptions import list_l2arctic_samples_with_transcripts
from src.asr_adaptation.data.wav2vec2_speaker_embeddings import compute_speaker_centroid_wav2vec2
from src.asr_adaptation.inference.transcribe import transcribe
from src.asr_adaptation.metrics.wer import compute_wer
from src.asr_adaptation.models.film_lora import load_film_lora_model
from src.asr_adaptation.pipeline.lora_train import _split_samples

_N_EVAL = 100
_N_TRAIN_DEFAULT = None


class WrongSpeakerRow(BaseModel):
    speaker_id: str
    utterance_id: str
    wrong_speaker_id: str
    reference: str
    hypothesis_correct: str
    hypothesis_wrong: str
    wer_correct: float
    wer_wrong: float

    @computed_field
    @property
    def wer_delta(self) -> float:
        """Positive = wrong centroid hurt WER (model is using speaker content)."""
        return self.wer_wrong - self.wer_correct


def _load_existing_results(csv_path: Path) -> dict[str, dict]:
    """Read the correct-speaker results CSV keyed by utterance_id."""
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["utterance_id"]] = row
    return rows


def _save_csv(rows: list[WrongSpeakerRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(WrongSpeakerRow.model_fields) + ["wer_delta"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            d = row.model_dump()
            d["wer_correct"] = round(d["wer_correct"], 4)
            d["wer_wrong"] = round(d["wer_wrong"], 4)
            d["wer_delta"] = round(d["wer_delta"], 4)
            writer.writerow(d)
    logger.info(f"Saved {len(rows)} rows → {path}")


def run_film_wrong_speaker(
    speaker_id: str,
    wrong_speaker_id: str,
    l2arctic_zip: str,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    cache_dir: str | None = None,
    n_train: int | None = _N_TRAIN_DEFAULT,
    n_eval: int = _N_EVAL,
    seed: int = 0,
    profile_layer: int = -1,
) -> list[WrongSpeakerRow]:
    """Run wrong-speaker control inference for one speaker pair (A model, B centroid).

    Args:
        speaker_id: Target speaker whose trained model and test set are used.
        wrong_speaker_id: Speaker whose centroid is injected as the wrong profile.
        l2arctic_zip: Path to l2arctic_release_v5.0.zip.
        checkpoint_dir: Root directory containing film_lora_weights/ and film_adaptation_results/.
        output_dir: Root for writing results CSV.
        cache_dir: HuggingFace model cache directory.
        n_train: Must match training value — used to compute the wrong speaker's centroid
                 from the same number of utterances. (default: all available)
        n_eval: Must match training value (default: 100).
        seed: Must match training seed so the split is identical (default: 0).
        profile_layer: Encoder layer for centroid extraction (default -1 = last).

    Returns:
        List of WrongSpeakerRow, one per eval utterance.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Wrong-speaker control: {speaker_id} (model) ← {wrong_speaker_id} (centroid) | device: {device}")

    # Load the existing correct-speaker results — no need to rerun inference
    correct_csv = checkpoint_dir / "film_adaptation_results" / f"{speaker_id}.csv"
    logger.info(f"Reading correct-speaker results from {correct_csv} ...")
    correct_results = _load_existing_results(correct_csv)
    eval_utterance_ids = set(correct_results.keys())
    logger.info(f"Found {len(eval_utterance_ids)} eval utterances in existing CSV")

    # Load speaker A's audio for the eval utterances only
    logger.info(f"Loading audio for {speaker_id} eval set ...")
    all_samples = list_l2arctic_samples_with_transcripts(l2arctic_zip, speaker_id)
    eval_samples = [s for s in all_samples if s.utterance_id in eval_utterance_ids]
    if not eval_samples:
        raise ValueError(f"No eval samples found for {speaker_id} matching the existing CSV utterance IDs.")

    # Compute the wrong speaker's centroid from their training utterances
    logger.info(f"Loading utterances for wrong speaker {wrong_speaker_id} ...")
    wrong_all = list_l2arctic_samples_with_transcripts(l2arctic_zip, wrong_speaker_id)
    wrong_train, _ = _split_samples(wrong_all, n_eval, n_train, seed)
    logger.info(f"Computing wrong centroid from {len(wrong_train)} utterances ...")
    wrong_centroid = compute_speaker_centroid_wav2vec2(
        wrong_train, device=device, profile_layer=profile_layer, cache_dir=cache_dir
    ).to(device)

    # Load pre-trained FiLM+LoRA — no retraining
    logger.info(f"Loading FiLM+LoRA checkpoint for {speaker_id} ...")
    model, processor = load_film_lora_model(
        speaker_id,
        checkpoint_dir=checkpoint_dir / "film_lora_weights",
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()

    # Single inference pass with the wrong centroid
    logger.info(f"Transcribing eval set with {wrong_speaker_id}'s centroid ...")
    hyps_wrong = [
        transcribe(s.waveform, processor, model, device, speaker_embedding=wrong_centroid)
        for s in eval_samples
    ]

    rows: list[WrongSpeakerRow] = []
    for sample, hyp_wrong in zip(eval_samples, hyps_wrong):
        correct = correct_results[sample.utterance_id]
        rows.append(
            WrongSpeakerRow(
                speaker_id=speaker_id,
                utterance_id=sample.utterance_id,
                wrong_speaker_id=wrong_speaker_id,
                reference=sample.transcript,
                hypothesis_correct=correct["hypothesis_adapted"],
                hypothesis_wrong=hyp_wrong,
                wer_correct=float(correct["wer_adapted"]),
                wer_wrong=compute_wer([sample.transcript], [hyp_wrong]),
            )
        )

    avg_correct = sum(r.wer_correct for r in rows) / len(rows)
    avg_wrong = sum(r.wer_wrong for r in rows) / len(rows)
    logger.info(
        f"{speaker_id}: correct-centroid WER={avg_correct:.3f}  "
        f"wrong-centroid WER={avg_wrong:.3f}  "
        f"delta={avg_wrong - avg_correct:+.3f}"
    )

    out_path = output_dir / "film_wrong_speaker_results" / f"{speaker_id}_wrong_{wrong_speaker_id}.csv"
    _save_csv(rows, out_path)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FiLM wrong-speaker control inference (no retraining).")
    parser.add_argument("--speaker-id", required=True, help="Target speaker (model owner and test set)")
    parser.add_argument("--wrong-speaker-id", required=True, help="Speaker whose centroid is injected")
    parser.add_argument("--l2arctic-zip", required=True, help="Path to l2arctic_release_v5.0.zip")
    parser.add_argument("--checkpoint-dir", required=True, help="Root dir containing film_lora_weights/ and film_adaptation_results/")
    parser.add_argument("--output-dir", required=True, help="Root dir for results CSV")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model cache directory")
    parser.add_argument("--n-train", type=int, default=_N_TRAIN_DEFAULT, help="Must match training value (for wrong speaker centroid computation)")
    parser.add_argument("--n-eval", type=int, default=_N_EVAL, help="Must match training value")
    parser.add_argument("--seed", type=int, default=0, help="Must match training seed")
    parser.add_argument("--profile-layer", type=int, default=-1, help="Encoder layer for centroid extraction")

    args = parser.parse_args()
    run_film_wrong_speaker(
        speaker_id=args.speaker_id,
        wrong_speaker_id=args.wrong_speaker_id,
        l2arctic_zip=args.l2arctic_zip,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
        profile_layer=args.profile_layer,
    )
