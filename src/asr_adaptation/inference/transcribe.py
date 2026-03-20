from __future__ import annotations

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def transcribe(
    waveform: torch.Tensor,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
    chunk_length_s: int = 30,
    sampling_rate: int = 16000,
) -> str:
    """
    Transcribe a waveform using a wav2vec2 model.

    Long recordings are split into chunks to avoid memory issues.
    Chunk transcriptions are joined with a space.

    Args:
        waveform: 1D float32 tensor at `sampling_rate`.
        processor: Matching Wav2Vec2Processor.
        model: Wav2Vec2ForCTC (base or LoRA-wrapped).
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

    parts: list[str] = []
    for chunk in chunks:
        inputs = processor(
            chunk.numpy(),
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        parts.append(processor.decode(predicted_ids[0]))

    return " ".join(parts).strip()
