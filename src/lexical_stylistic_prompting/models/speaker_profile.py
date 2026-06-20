"""Build and cache call-level ASR context profiles."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import APITimeoutError, OpenAI
from pydantic import BaseModel

from src.lexical_stylistic_prompting.models.prompts import (
    METADATA_ONLY_SYSTEM,
    METADATA_ONLY_USER,
)

load_dotenv()

PROFILES_DIR = Path("data/processed/lexical_stylistic_prompting/profiles")
KISSKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"


class ProfileStrategy(str, Enum):
    METADATA_ONLY = "metadata_only"


class SpeakerProfile(BaseModel):
    speaker_id: str
    n_profile: int
    strategy: ProfileStrategy
    model: str
    prompt: str
    created_at: str



def _get_client() -> OpenAI:
    api_key = os.environ.get("KISSKI_API_KEY") or os.environ.get("SAIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set KISSKI_API_KEY or SAIA_API_KEY in your environment or .env file."
        )
    return OpenAI(
        base_url=KISSKI_BASE_URL,
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )


LLM_TIMEOUT_SECONDS = 120
LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT_SECONDS = 10


def _llm_call(client: OpenAI, model: str, system: str, user: str) -> str:
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info(f"Sending request to KISSKI ({model}), attempt {attempt}/{LLM_MAX_RETRIES} ...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=200,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content
            assert content is not None, "LLM returned empty content"
            return content.strip()
        except APITimeoutError:
            if attempt == LLM_MAX_RETRIES:
                raise
            logger.warning(f"KISSKI timed out, retrying in {LLM_RETRY_WAIT_SECONDS}s ...")
            time.sleep(LLM_RETRY_WAIT_SECONDS)
    raise RuntimeError("unreachable")


def build_profile(
    speaker_id: str,
    strategy: ProfileStrategy,
    n_profile: int,
    company_name: str,
    sector: str,
    financial_quarter: str,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> SpeakerProfile:
    if client is None:
        client = _get_client()

    if strategy == ProfileStrategy.METADATA_ONLY:
        user_msg = METADATA_ONLY_USER.format(
            company_name=company_name,
            sector=sector,
            financial_quarter=financial_quarter,
        )
        prompt = _llm_call(client, model, METADATA_ONLY_SYSTEM, user_msg)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return SpeakerProfile(
        speaker_id=speaker_id,
        n_profile=n_profile,
        strategy=strategy,
        model=model,
        prompt=prompt,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_profile(profile: SpeakerProfile, profiles_dir: Path = PROFILES_DIR) -> Path:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    out = profiles_dir / f"{profile.speaker_id}_{profile.n_profile}_{profile.strategy.value}.json"
    out.write_text(profile.model_dump_json(indent=2))
    logger.info(f"Saved profile → {out}")
    return out


def load_profile(
    speaker_id: str,
    n_profile: int,
    strategy: ProfileStrategy,
    profiles_dir: Path = PROFILES_DIR,
) -> SpeakerProfile:
    path = profiles_dir / f"{speaker_id}_{n_profile}_{strategy.value}.json"
    return SpeakerProfile.model_validate_json(path.read_text())

