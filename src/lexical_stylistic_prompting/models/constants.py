"""Shared constants for the lexical/stylistic prompting pipeline."""

from pathlib import Path

PROFILES_DIR = Path("data/processed/lexical_stylistic_prompting/profiles")

KISSKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"

LLM_TIMEOUT_SECONDS = 120
LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT_SECONDS = 10
