from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from src.asr_adaptation.data.l2arctic_transcriptions import list_l2arctic_samples_with_transcripts
from src.asr_adaptation.data.speaker_embeddings import compute_speaker_centroid
from src.asr_adaptation.data.wav2vec2_speaker_embeddings import compute_speaker_centroid_wav2vec2
from src.asr_adaptation.metrics.wer import compute_wer
from src.asr_adaptation.models.film_lora import (
    build_film_lora_model,
    save_film_speaker_adapter,
    trainable_parameter_summary,
)
from src.asr_adaptation.pipeline.lora_train import (
    AdaptationRow,
    _get_hypotheses,
    _save_csv,
    _split_samples,
    _train_lora,
)

_N_EVAL = 100
_N_TRAIN_DEFAULT = None  # use all available training utterances
_N_EPOCHS = 10
_LEARNING_RATE = 1e-5
_GRAD_ACCUM_STEPS = 4


def run_film_train(
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
    wavlm_model: str = "microsoft/wavlm-base-plus",
    profile_extractor: str = "wav2vec2",
    profile_layer: int = -1,
    wrong_speaker_id: str | None = None,
    no_profile: bool = False,
) -> list[AdaptationRow]:
    """
    Fine-tune a FiLM-conditioned LoRA adapter for one speaker and evaluate WER.

    Args:
        speaker_id: L2-ARCTIC speaker ID, e.g. "ABA".
        l2arctic_zip: Path to l2arctic_release_v5.0.zip.
        output_dir: Root for saving LoRA weights and results CSV.
        cache_dir: HuggingFace model cache directory.
        n_train: Number of utterances for training. None = use all available.
        n_eval: Number of utterances held out for evaluation.
        n_epochs: Training epochs.
        learning_rate: AdamW learning rate.
        grad_accum_steps: Gradient accumulation steps.
        seed: Random seed for shuffling the training pool.
        profile_extractor: "wav2vec2" (default) or "wavlm".
        profile_layer: Which encoder layer to extract the profile from (default -1 = last).
        wrong_speaker_id: If set, use this speaker's centroid at eval time only (control experiment).
        no_profile: Disable profile injection — LoRA-only mode.

    Returns:
        List of AdaptationRow, one per eval utterance.
    """
    output_dir = Path(output_dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Speaker {speaker_id} | device: {device}")

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

    # Compute speaker centroid from training utterances
    if no_profile:
        speaker_centroid = None
        eval_centroid = None
        logger.info("Profile injection disabled — running LoRA-only mode.")
    elif profile_extractor == "wavlm":
        speaker_centroid = compute_speaker_centroid(
            train_samples, device=device, model_name=wavlm_model, cache_dir=cache_dir
        ).to(device)
        eval_centroid = speaker_centroid
    else:
        speaker_centroid = compute_speaker_centroid_wav2vec2(
            train_samples, device=device, profile_layer=profile_layer, cache_dir=cache_dir
        ).to(device)
        eval_centroid = speaker_centroid

    # Wrong-speaker control: compute a different centroid for eval (inference only)
    if wrong_speaker_id is not None and not no_profile:
        logger.info(f"Wrong-speaker control: loading centroid for {wrong_speaker_id} ...")
        wrong_samples = list_l2arctic_samples_with_transcripts(l2arctic_zip, wrong_speaker_id)
        wrong_train, _ = _split_samples(wrong_samples, n_eval, n_train, seed)
        if profile_extractor == "wavlm":
            eval_centroid = compute_speaker_centroid(
                wrong_train, device=device, model_name=wavlm_model, cache_dir=cache_dir
            ).to(device)
        else:
            eval_centroid = compute_speaker_centroid_wav2vec2(
                wrong_train, device=device, profile_layer=profile_layer, cache_dir=cache_dir
            ).to(device)
        logger.info(f"Eval will use {wrong_speaker_id}'s centroid (wrong-speaker control).")

    logger.info("Building FiLM-conditioned LoRA model ...")
    model, processor = build_film_lora_model(cache_dir=cache_dir)
    if no_profile:
        for param in model.film_mlp.parameters():
            param.requires_grad = False
    torch.nn.Module.to(model, device)
    summary = trainable_parameter_summary(model)
    logger.info(
        f"Trainable params: {summary['trainable']:,} / {summary['total']:,} "
        f"({100 * summary['trainable'] / summary['total']:.2f}%)"
    )

    logger.info("Evaluating baseline (no adaptation) ...")
    hypotheses_baseline = _get_hypotheses(model, eval_samples, processor, device, eval_centroid)

    logger.info(f"Fine-tuning on {actual_n_train} utterances ...")
    _train_lora(model, train_samples, processor, device, speaker_centroid, n_epochs, learning_rate, grad_accum_steps)

    logger.info("Evaluating adapted model ...")
    hypotheses_adapted = _get_hypotheses(model, eval_samples, processor, device, eval_centroid)

    adapter_path = save_film_speaker_adapter(model, speaker_id, output_dir / "film_lora_weights")
    logger.info(f"Saved FiLM adapter → {adapter_path}")

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

    _save_csv(rows, output_dir / "film_adaptation_results" / f"{speaker_id}.csv")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FiLM-conditioned LoRA speaker adaptation for wav2vec2.")
    parser.add_argument("--speaker-id", required=True, help="L2-ARCTIC speaker ID, e.g. ABA")
    parser.add_argument("--l2arctic-zip", required=True, help="Path to l2arctic_release_v5.0.zip")
    parser.add_argument("--output-dir", required=True, help="Root directory for saving weights and results")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model cache directory")
    parser.add_argument("--n-train", type=int, default=_N_TRAIN_DEFAULT, help="Max training utterances")
    parser.add_argument("--n-eval", type=int, default=_N_EVAL, help="Held-out evaluation utterances")
    parser.add_argument("--n-epochs", type=int, default=_N_EPOCHS, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=_LEARNING_RATE, help="AdamW learning rate")
    parser.add_argument("--grad-accum-steps", type=int, default=_GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus", help="WavLM model name (used with --profile-extractor wavlm)")
    parser.add_argument("--profile-extractor", default="wav2vec2", help="Profile extractor: 'wav2vec2' (default) or 'wavlm'")
    parser.add_argument("--profile-layer", type=int, default=-1, help="Encoder layer for profile extraction (default -1 = last)")
    parser.add_argument("--wrong-speaker-id", default=None, help="Use this speaker's centroid at eval time (wrong-speaker control)")
    parser.add_argument("--no-profile", action="store_true", help="LoRA-only mode: disable FiLM conditioning")

    args = parser.parse_args()
    run_film_train(
        speaker_id=args.speaker_id,
        l2arctic_zip=args.l2arctic_zip,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        wavlm_model=args.wavlm_model,
        profile_extractor=args.profile_extractor,
        profile_layer=args.profile_layer,
        wrong_speaker_id=args.wrong_speaker_id,
        no_profile=args.no_profile,
    )
