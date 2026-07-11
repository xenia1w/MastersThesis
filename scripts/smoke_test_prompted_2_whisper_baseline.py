"""
Smoke test 2/3 — Whisper loads and transcribes one segment without a prompt.

Verifies that the ASR pipeline and audio loading work correctly before
adding the prompt injection layer in step 3.

Run: uv run scripts/smoke_test_prompted_2_whisper_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from transformers import pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    load_audio_segment,
    load_earnings21,
)

DATA_DIR = Path("data/raw/earnings21")
MODEL = "openai/whisper-medium"
CALL_INDEX = 0   # Monro Inc
SEG_INDEX = 22   # first segment after the 20-segment profile window

calls = load_earnings21(DATA_DIR, min_tokens=5)
call = calls[CALL_INDEX]
seg = call.segments[SEG_INDEX]

logger.info(f"Call:      {call.call_id} — {call.company_name}")
logger.info(f"Segment:   {seg.segment_id} | {seg.start_ts:.1f}–{seg.end_ts:.1f}s")
logger.info(f"Reference: {seg.text}")
logger.info(f"Entities:  {[t.text for t in seg.tokens if t.entity_type]}")

audio = load_audio_segment(call.audio_path, seg.start_ts, seg.end_ts)

logger.info(f"Loading {MODEL} (this may take a moment on first run) ...")
pipe = hf_pipeline("automatic-speech-recognition", model=MODEL)

logger.info("Transcribing ...")
result = pipe(audio)
assert isinstance(result, dict)
hypothesis = result["text"].strip()
wer = compute_wer([seg.text], [hypothesis])

logger.info(f"Hypothesis: {hypothesis}")
logger.info(f"WER:        {wer:.3f}")
logger.info("Smoke test 2 PASSED — run smoke_test_prompted_3_whisper_prompted.py next")
