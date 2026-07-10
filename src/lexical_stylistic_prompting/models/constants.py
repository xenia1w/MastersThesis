"""Shared constants for the lexical/stylistic prompting pipeline (metadata_only)."""

from pathlib import Path

# Where LLM-generated ASR context profiles are cached (one subdir per strategy).
PROFILES_DIR = Path("data/processed/lexical_stylistic_prompting/profiles")

# Noisy unprompted Whisper transcripts of each call's first n_profile turns, consumed by the
# transcript_only / transcript_plus_knowledge profile builders.
PROFILE_TRANSCRIPTS_DIR = Path("data/processed/lexical_stylistic_prompting/profile_transcripts")

# KISSKI / SAIA academic-cloud OpenAI-compatible endpoint used to build profiles.
KISSKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_LLM_MODEL = "qwen3-30b-a3b-instruct-2507"

LLM_TIMEOUT_SECONDS = 120
LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT_SECONDS = 10

# Whisper's decoder prompt window is ~224 tokens and end-weighted. Keep the injected
# keyword list comfortably under that so it always fits and biasing stays effective.
MAX_PROMPT_TOKENS = 200
MAX_PROMPT_TERMS = 40
# A keyword is a short phrase; anything longer is prose that slipped through and is dropped.
MAX_WORDS_PER_TERM = 5
# Word cap for the prose (natural-sentence) format. Kept well under MAX_PROMPT_TOKENS so the
# passage always fits Whisper's prompt window; the token budget is still enforced as a backstop.
MAX_PROMPT_WORDS = 60
