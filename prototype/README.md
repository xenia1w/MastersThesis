# Prosodic Feature Prototype

This is a small Python prototype that extracts classic prosodic features from an audio file.

## Setup (uv)

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Usage

```bash
uv run extract_prosody.py path/to/audio.wav --output features.json
```

If you omit `--output`, the JSON is printed to stdout.

## Features

- Pitch (f0) stats: mean, std, min, max, range
- Voiced fraction
- Energy (RMS) stats (linear + dB)
- Pause stats: count, mean, std, min, max
- Total speech and total silence duration
- Speech rate proxy (total speech / total duration)

## Notes

- Audio is resampled to 16 kHz mono.
- This is a lightweight baseline; it can be extended with WavLM embeddings later.
