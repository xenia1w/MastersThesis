"""Quick smoke test for the Earnings21 pipeline — runs Whisper on one segment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from transformers import pipeline as hf_pipeline


class _ASROutput(BaseModel):
    text: str

from src.asr_adaptation.metrics.wer import compute_wer
from src.lexical_stylistic_prompting.data.earnings21_utils import (
    load_audio_segment,
    load_earnings21,
)

DATA_DIR = Path("data/raw/earnings21")
MODEL = "openai/whisper-medium"
CALL_INDEX = 0   # Monro Inc
SEG_INDEX = 2    # first real speaker (skip operator intro)

calls = load_earnings21(DATA_DIR, min_tokens=5)
c = calls[CALL_INDEX]
seg = c.segments[SEG_INDEX]

print(f"Call:      {c.call_id} — {c.company_name} ({c.sector})")
print(f"Segment:   {seg.segment_id} | spk={seg.speaker} | {seg.start_ts:.1f}-{seg.end_ts:.1f}s")
print(f"Reference: {seg.text}")
print(f"Entities:  {[t.text for t in seg.tokens if t.entity_type]}")
print()

audio = load_audio_segment(c.audio_path, seg.start_ts, seg.end_ts)

print(f"Loading {MODEL} ...")
pipe = hf_pipeline("automatic-speech-recognition", model=MODEL, chunk_length_s=30)
hyp = _ASROutput.model_validate(pipe(audio)).text.strip()
wer = compute_wer([seg.text], [hyp])

print(f"Hypothesis: {hyp}")
print(f"WER:        {wer:.3f}")
print()
print("Entity check:")
hyp_lower = hyp.lower()
for tok in seg.tokens:
    if tok.entity_type:
        found = tok.text.lower() in hyp_lower
        status = "OK   " if found else "MISSED"
        print(f"  [{tok.entity_type:15s}] {tok.text!r:30s} → {status}")
