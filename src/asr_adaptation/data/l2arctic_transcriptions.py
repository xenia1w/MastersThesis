from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch


@dataclass
class L2ArcticTranscriptSample:
    speaker_id: str
    utterance_id: str
    waveform: torch.Tensor  # 1D float32, 16 kHz
    sampling_rate: int
    transcript: str


def load_l2arctic_transcript(
    outer_zip_path: str,
    speaker_id: str,
    wav_filename: str,
) -> str:
    """
    Load a single transcript for an L2-ARCTIC utterance.

    Args:
        outer_zip_path: Path to l2arctic_release_v5.0.zip
        speaker_id: Speaker ID, e.g. "ABA"
        wav_filename: WAV filename, e.g. "arctic_a0001.wav"

    Returns:
        Transcript as a plain string (stripped of whitespace).
    """
    utterance_id = Path(wav_filename).stem
    transcript_path = f"{speaker_id}/transcript/{utterance_id}.txt"
    inner_zip_name = f"{speaker_id}.zip"

    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_bytes = outer.read(inner_zip_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        text = inner.read(transcript_path).decode("utf-8").strip()

    return text


def list_l2arctic_samples_with_transcripts(
    outer_zip_path: str,
    speaker_id: str,
    target_sr: int = 16000,
) -> List[L2ArcticTranscriptSample]:
    """
    Load all (audio, transcript) pairs for a speaker in a single zip pass.

    Opens the speaker's inner zip once and loads all WAV files together with
    their matching transcript files. Utterances are returned sorted by name.

    Args:
        outer_zip_path: Path to l2arctic_release_v5.0.zip
        speaker_id: Speaker ID, e.g. "ABA"
        target_sr: Target sampling rate (default 16 kHz)

    Returns:
        List of L2ArcticTranscriptSample, one per utterance.
    """
    inner_zip_name = f"{speaker_id}.zip"
    wav_prefix = f"{speaker_id}/wav/"
    transcript_prefix = f"{speaker_id}/transcript/"

    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_bytes = outer.read(inner_zip_name)

    samples: List[L2ArcticTranscriptSample] = []

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        wav_names = sorted(
            Path(m).name
            for m in inner.namelist()
            if m.startswith(wav_prefix) and m.endswith(".wav")
        )

        for wav_name in wav_names:
            utterance_id = Path(wav_name).stem

            # Load audio
            wav_bytes = inner.read(f"{wav_prefix}{wav_name}")
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            waveform = torch.from_numpy(audio.astype(np.float32))

            # Load transcript
            transcript_path = f"{transcript_prefix}{utterance_id}.txt"
            text = inner.read(transcript_path).decode("utf-8").strip()

            samples.append(
                L2ArcticTranscriptSample(
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    waveform=waveform,
                    sampling_rate=sr,
                    transcript=text,
                )
            )

    return samples
