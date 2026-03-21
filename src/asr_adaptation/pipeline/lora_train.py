from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel, PeftMixedModel
from transformers import Wav2Vec2Processor

from src.asr_adaptation.data.l2arctic_transcriptions import (
    L2ArcticTranscriptSample,
    list_l2arctic_samples_with_transcripts,
)
from src.asr_adaptation.inference.transcribe import transcribe
from src.asr_adaptation.metrics.wer import compute_wer
from src.asr_adaptation.models.wav2vec_lora import (
    build_lora_model,
    save_speaker_adapter,
    trainable_parameter_summary,
)

# Default training hyperparameters
_N_EVAL = 100
_N_TRAIN_DEFAULT = 500
_N_EPOCHS = 10
_LEARNING_RATE = 1e-4
_GRAD_ACCUM_STEPS = 4


@dataclass
class AdaptationRow:
    speaker_id: str
    utterance_id: str
    n_train: int
    reference: str
    hypothesis_baseline: str
    hypothesis_adapted: str
    wer_baseline: float
    wer_adapted: float


def _split_samples(
    samples: list[L2ArcticTranscriptSample],
    n_eval: int,
    n_train: int | None,
    seed: int,
) -> tuple[list[L2ArcticTranscriptSample], list[L2ArcticTranscriptSample]]:
    """
    Split samples into train and eval sets.

    Eval is always the last n_eval samples (held out from shuffling).
    The training pool is shuffled with `seed` before taking n_train samples,
    so different seeds yield different subsets when n_train < pool size.
    """
    eval_samples = samples[-n_eval:]
    train_pool = list(samples[:-n_eval])

    rng = random.Random(seed)
    rng.shuffle(train_pool)

    if n_train is not None:
        train_pool = train_pool[:n_train]

    return train_pool, eval_samples


def _train_lora(
    model: PeftModel | PeftMixedModel,
    train_samples: list[L2ArcticTranscriptSample],
    processor: Wav2Vec2Processor,
    device: torch.device,
    n_epochs: int = _N_EPOCHS,
    learning_rate: float = _LEARNING_RATE,
    grad_accum_steps: int = _GRAD_ACCUM_STEPS,
) -> None:
    """Fine-tune LoRA weights on the training samples using CTC loss."""
    # Prevent NaN loss from crashing training when a sample's label sequence is
    # longer than the encoder output (CTC constraint violation).
    model.config.ctc_zero_infinity = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, sample in enumerate(train_samples):
            inputs = processor(
                sample.waveform.numpy(),
                sampling_rate=sample.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)

            # processor.tokenizer exists at runtime but is absent from HF type stubs
            labels = getattr(processor, "tokenizer")(
                sample.transcript,
                return_tensors="pt",
            ).input_ids.to(device)

            output = model(input_values=input_values, labels=labels)
            if torch.isnan(output.loss):
                logger.warning(f"NaN loss at epoch {epoch + 1} step {step} — skipping")
                continue
            loss = output.loss / grad_accum_steps
            loss.backward()
            total_loss += output.loss.item()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining accumulated gradients at end of epoch
        if len(train_samples) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(len(train_samples), 1)
        logger.info(f"Epoch {epoch + 1}/{n_epochs} | avg loss: {avg_loss:.4f}")


def _get_hypotheses(
    model: PeftModel | PeftMixedModel,
    eval_samples: list[L2ArcticTranscriptSample],
    processor: Wav2Vec2Processor,
    device: torch.device,
) -> list[str]:
    """Transcribe all eval samples and return hypotheses."""
    model.eval()
    return [
        transcribe(sample.waveform, processor, model, device)
        for sample in eval_samples
    ]


def run_lora_train(
    speaker_id: str,
    l2arctic_zip: str,
    output_dir: str | Path,
    cache_dir: str | None = None,
    n_train: int | None = _N_TRAIN_DEFAULT,
    n_eval: int = _N_EVAL,
    n_epochs: int = _N_EPOCHS,
    learning_rate: float = _LEARNING_RATE,
    grad_accum_steps: int = _GRAD_ACCUM_STEPS,
    seed: int = 0,
) -> list[AdaptationRow]:
    """
    Fine-tune a LoRA adapter for one speaker and evaluate WER before and after.

    Args:
        speaker_id: L2-ARCTIC speaker ID, e.g. "ABA".
        l2arctic_zip: Path to l2arctic_release_v5.0.zip.
        output_dir: Root for saving LoRA weights and results CSV.
        cache_dir: HuggingFace model cache directory.
        n_train: Number of utterances for training. None = use all available.
        n_eval: Number of utterances held out for evaluation.
        n_epochs: Training epochs.
        learning_rate: AdamW learning rate.
        grad_accum_steps: Gradient accumulation steps (simulates larger batch).
        seed: Random seed for shuffling the training pool.

    Returns:
        List of AdaptationRow, one per eval utterance.
    """
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Speaker {speaker_id} | device: {device}")

    # Load data
    logger.info(f"Loading utterances for {speaker_id} ...")
    all_samples = list_l2arctic_samples_with_transcripts(l2arctic_zip, speaker_id)

    if len(all_samples) <= n_eval:
        raise ValueError(
            f"Speaker {speaker_id} has only {len(all_samples)} utterances "
            f"but n_eval={n_eval} — not enough data."
        )

    train_samples, eval_samples = _split_samples(all_samples, n_eval, n_train, seed)
    actual_n_train = len(train_samples)
    logger.info(f"Train: {actual_n_train} utterances | Eval: {len(eval_samples)} utterances")

    # Build model
    logger.info("Building LoRA model ...")
    model, processor = build_lora_model(cache_dir=cache_dir)
    torch.nn.Module.to(model, device)
    summary = trainable_parameter_summary(model)
    logger.info(
        f"Trainable params: {summary['trainable']:,} / {summary['total']:,} "
        f"({100 * summary['trainable'] / summary['total']:.2f}%)"
    )

    # Baseline evaluation (before training)
    logger.info("Evaluating baseline (no adaptation) ...")
    hypotheses_baseline = _get_hypotheses(model, eval_samples, processor, device)

    # Fine-tune
    logger.info(f"Fine-tuning on {actual_n_train} utterances ...")
    _train_lora(model, train_samples, processor, device, n_epochs, learning_rate, grad_accum_steps)

    # Adapted evaluation (after training)
    logger.info("Evaluating adapted model ...")
    hypotheses_adapted = _get_hypotheses(model, eval_samples, processor, device)

    # Save LoRA adapter
    adapter_path = save_speaker_adapter(model, speaker_id, output_dir / "lora_weights")
    logger.info(f"Saved LoRA adapter → {adapter_path}")

    # Build result rows
    rows: list[AdaptationRow] = []
    for sample, hyp_base, hyp_adapted in zip(eval_samples, hypotheses_baseline, hypotheses_adapted):
        rows.append(
            AdaptationRow(
                speaker_id=speaker_id,
                utterance_id=sample.utterance_id,
                n_train=actual_n_train,
                reference=sample.transcript,
                hypothesis_baseline=hyp_base,
                hypothesis_adapted=hyp_adapted,
                wer_baseline=compute_wer([sample.transcript], [hyp_base]),
                wer_adapted=compute_wer([sample.transcript], [hyp_adapted]),
            )
        )

    avg_base = sum(r.wer_baseline for r in rows) / len(rows)
    avg_adapted = sum(r.wer_adapted for r in rows) / len(rows)
    logger.info(
        f"Speaker {speaker_id} | "
        f"baseline WER: {avg_base:.3f} → adapted WER: {avg_adapted:.3f} "
        f"(delta: {avg_adapted - avg_base:+.3f})"
    )

    # Save per-speaker CSV
    _save_csv(rows, output_dir / "adaptation_results" / f"{speaker_id}.csv")

    return rows


def _save_csv(rows: list[AdaptationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id", "utterance_id", "n_train",
                "reference", "hypothesis_baseline", "hypothesis_adapted",
                "wer_baseline", "wer_adapted", "wer_delta",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(
                    speaker_id=row.speaker_id,
                    utterance_id=row.utterance_id,
                    n_train=row.n_train,
                    reference=row.reference,
                    hypothesis_baseline=row.hypothesis_baseline,
                    hypothesis_adapted=row.hypothesis_adapted,
                    wer_baseline=round(row.wer_baseline, 4),
                    wer_adapted=round(row.wer_adapted, 4),
                    wer_delta=round(row.wer_adapted - row.wer_baseline, 4),
                )
            )
    logger.info(f"Saved {len(rows)} rows → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-speaker LoRA fine-tuning")
    parser.add_argument("--speaker",       required=True,        help="L2-ARCTIC speaker ID, e.g. ABA")
    parser.add_argument("--l2arctic-zip",  required=True,        help="Path to l2arctic_release_v5.0.zip")
    parser.add_argument("--output-dir",    required=True,        help="Root output directory")
    parser.add_argument("--cache-dir",     default=None,         help="HuggingFace model cache directory")
    parser.add_argument("--n-train",       type=int, default=_N_TRAIN_DEFAULT, help="Training utterances (default: 500)")
    parser.add_argument("--n-eval",        type=int, default=_N_EVAL,          help="Eval utterances (default: 100)")
    parser.add_argument("--n-epochs",      type=int, default=_N_EPOCHS,        help="Training epochs (default: 10)")
    parser.add_argument("--seed",          type=int, default=0,                help="Random seed (default: 0)")
    args = parser.parse_args()

    run_lora_train(
        speaker_id=args.speaker,
        l2arctic_zip=args.l2arctic_zip,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )
