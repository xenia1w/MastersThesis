from __future__ import annotations

import io
import zipfile
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch


def load_l2arctic_wav(
    outer_zip_path: str,
    speaker_id: str,
    wav_filename: str,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load a single L2ARCTIC wav from the nested zip without full extraction.

    Args:
        outer_zip_path: Path to l2arctic_release_v5.0.zip
        speaker_id: Speaker ID, e.g., "ABA"
        wav_filename: File name like "arctic_a0001.wav"
        target_sr: Target sampling rate

    Returns:
        (waveform, sampling_rate) where waveform is 1D float32 torch tensor
    """
    inner_zip_name = f"{speaker_id}.zip"
    wav_path = f"{speaker_id}/wav/{wav_filename}"

    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_bytes = outer.read(inner_zip_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        wav_bytes = inner.read(wav_path)

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    waveform = torch.from_numpy(audio.astype(np.float32))
    return waveform, sr
