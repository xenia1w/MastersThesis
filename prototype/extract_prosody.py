#!/usr/bin/env python3
"""Simple prosodic feature extractor.

Usage:
  python extract_prosody.py path/to/audio.wav --output features.json

The script computes classic prosodic features from audio:
- pitch (f0) statistics
- energy (RMS) statistics
- voiced fraction
- pause statistics
- speech rate proxy (voiced time / total time)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import librosa


DEFAULT_SR = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 256


def _safe_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def extract_prosody(path: Path) -> Dict[str, Any]:
    y, sr = librosa.load(path, sr=DEFAULT_SR, mono=True)

    duration_sec = float(len(y) / sr) if len(y) > 0 else 0.0

    # Pitch (f0) using YIN
    f0 = librosa.yin(
        y,
        fmin=50,
        fmax=500,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    # YIN returns NaN for unvoiced frames
    voiced_mask = np.isfinite(f0)
    f0_voiced = f0[voiced_mask]

    pitch_stats = _safe_stats(f0_voiced)
    pitch_stats["range"] = float(pitch_stats["max"] - pitch_stats["min"])

    voiced_fraction = float(np.sum(voiced_mask) / len(f0)) if len(f0) > 0 else 0.0

    # Energy (RMS)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_stats = _safe_stats(rms)

    # Energy in dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max) if rms.size > 0 else np.array([])
    rms_db_stats = _safe_stats(rms_db)

    # Pause statistics using energy-based splitting
    # top_db controls silence sensitivity; 30 dB is a reasonable default
    intervals = librosa.effects.split(y, top_db=30, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    speech_durations = []
    for start, end in intervals:
        speech_durations.append((end - start) / sr)

    total_speech = float(np.sum(speech_durations)) if speech_durations else 0.0
    total_silence = max(duration_sec - total_speech, 0.0)

    # Pauses are the gaps between speech segments
    pause_durations = []
    if len(intervals) >= 2:
        for (s1, e1), (s2, _) in zip(intervals[:-1], intervals[1:]):
            pause = (s2 - e1) / sr
            if pause > 0:
                pause_durations.append(pause)

    pause_stats = _safe_stats(np.array(pause_durations))

    speech_rate_proxy = (total_speech / duration_sec) if duration_sec > 0 else 0.0

    return {
        "file": str(path),
        "sample_rate": sr,
        "duration_sec": duration_sec,
        "pitch_hz": pitch_stats,
        "voiced_fraction": voiced_fraction,
        "rms": rms_stats,
        "rms_db": rms_db_stats,
        "speech": {
            "total_speech_sec": total_speech,
            "total_silence_sec": total_silence,
            "speech_rate_proxy": speech_rate_proxy,
        },
        "pauses": {
            "count": len(pause_durations),
            **pause_stats,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract basic prosodic features from audio.")
    parser.add_argument("audio", type=Path, help="Path to input audio file (wav, mp3, flac, etc.)")
    parser.add_argument("--output", "-o", type=Path, help="Write features to JSON file")
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

    features = extract_prosody(args.audio)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
    else:
        print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
