"""
Smoke test 3/3 — Whisper with the profile prompt injected via prompt_ids.

Loads the profile saved by step 1, runs Whisper on the same segment as
step 2, and prints a side-by-side comparison.

Requires: smoke_test_prompted_1_kisski.py to have run first.

Run: uv run scripts/smoke_test_prompted_3_whisper_prompted.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger
from transformers import WhisperProcessor, pipeline as hf_pipeline

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    load_audio_segment,
    load_earnings21,
)
from src.lexical_stylistic_prompting.models.speaker_profile import (
    ProfileStrategy,
    load_profile,
)

DATA_DIR = Path("data/raw/earnings21")
MODEL = "openai/whisper-medium"
CALL_INDEX = 0
SEG_INDEX = 22
N_PROFILE = 20

calls = load_earnings21(DATA_DIR, min_tokens=5)
call = calls[CALL_INDEX]
seg = call.segments[SEG_INDEX]

logger.info(f"Call:      {call.call_id} — {call.company_name}")
logger.info(f"Segment:   {seg.segment_id} | {seg.start_ts:.1f}–{seg.end_ts:.1f}s")
logger.info(f"Reference: {seg.text}")

profile = load_profile(call.call_id, N_PROFILE, ProfileStrategy.METADATA_ONLY)
logger.info(f"Loaded profile prompt ({len(profile.prompt)} chars): {profile.prompt[:120]} ...")

audio = load_audio_segment(call.audio_path, seg.start_ts, seg.end_ts)

logger.info(f"Loading {MODEL} ...")
pipe = hf_pipeline("automatic-speech-recognition", model=MODEL)
processor = WhisperProcessor.from_pretrained(MODEL)
prompt_ids = torch.tensor(processor.get_prompt_ids(profile.prompt)).to(pipe.model.device)

logger.info("Transcribing without prompt ...")
result_baseline = pipe(audio["raw"])
assert isinstance(result_baseline, dict)
hyp_baseline = result_baseline["text"].strip()

logger.info("Transcribing with prompt ...")
result_prompted = pipe(audio["raw"], generate_kwargs={"prompt_ids": prompt_ids})
assert isinstance(result_prompted, dict)
hyp_prompted = result_prompted["text"].strip()

wer_baseline = compute_wer([seg.text], [hyp_baseline])
wer_prompted = compute_wer([seg.text], [hyp_prompted])

logger.info("=== Results ===")
logger.info(f"Reference : {seg.text}")
logger.info(f"Baseline  : {hyp_baseline}  (WER {wer_baseline:.3f})")
logger.info(f"Prompted  : {hyp_prompted}  (WER {wer_prompted:.3f})")
logger.info(f"Delta WER : {wer_prompted - wer_baseline:+.3f}")

logger.info("Entity check:")
for tok in seg.tokens:
    if tok.entity_type:
        base_ok = tok.text.lower() in hyp_baseline.lower()
        prom_ok = tok.text.lower() in hyp_prompted.lower()
        logger.info(
            f"  [{tok.entity_type:15s}] {tok.text!r:30s}"
            f"  baseline={'OK  ' if base_ok else 'MISS'}"
            f"  prompted={'OK  ' if prom_ok else 'MISS'}"
        )

logger.info("Smoke test 3 PASSED — full prompted pipeline verified")
