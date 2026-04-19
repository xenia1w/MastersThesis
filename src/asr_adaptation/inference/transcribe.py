from __future__ import annotations

import torch
from transformers import Wav2Vec2Processor


def transcribe(
    waveform: torch.Tensor,
    processor: Wav2Vec2Processor,
    model: torch.nn.Module,
    device: torch.device,
    speaker_embedding: torch.Tensor | None = None,
    chunk_length_s: int = 30,
    sampling_rate: int = 16000,
) -> str:
    """
    Transcribe a waveform using a WavLM CTC model.

    Long recordings are split into chunks to avoid memory issues.
    Chunk transcriptions are joined with a space.

    Args:
        waveform: 1D float32 tensor at `sampling_rate`.
        processor: Matching Wav2Vec2Processor.
        model: WavLMForCTC (base or LoRA-wrapped).
        device: Target device (cpu / cuda).
        chunk_length_s: Max seconds per chunk.
        sampling_rate: Expected sample rate of the waveform.

    Returns:
        Decoded transcript string.
    """
    chunk_size = chunk_length_s * sampling_rate
    chunks = [
        waveform[start : start + chunk_size]
        for start in range(0, len(waveform), chunk_size)
    ]

    # WavLM feature extractor requires a minimum input length (~400 samples);
    # a short trailing chunk can crash the conv layers.
    MIN_CHUNK_SAMPLES = 400

    parts: list[str] = []
    for chunk in chunks:
        if len(chunk) < MIN_CHUNK_SAMPLES:
            continue
        inputs = processor(
            chunk.numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            kwargs: dict = {"input_values": input_values}
            if speaker_embedding is not None:
                kwargs["speaker_embedding"] = speaker_embedding.to(device)
            logits = model(**kwargs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        parts.append(processor.decode(predicted_ids[0]))

    return " ".join(parts).strip()
