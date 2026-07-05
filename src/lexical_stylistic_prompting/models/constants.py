"""Shared constants for the lexical/stylistic prompting pipeline (metadata_only)."""

from pathlib import Path

# Where LLM-generated ASR context profiles are cached (one subdir per strategy).
PROFILES_DIR = Path("data/processed/lexical_stylistic_prompting/profiles")

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
