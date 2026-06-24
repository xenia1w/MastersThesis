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
    TRANSCRIPT_ONLY_SYSTEM,
    TRANSCRIPT_ONLY_USER,
    TRANSCRIPT_PLUS_KNOWLEDGE_SYSTEM,
    TRANSCRIPT_PLUS_KNOWLEDGE_USER,
)

from src.lexical_stylistic_prompting.models.constants import (
    DEFAULT_MODEL,
    KISSKI_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_RETRY_WAIT_SECONDS,
    LLM_TIMEOUT_SECONDS,
    PROFILES_DIR,
)

load_dotenv()


class ProfileStrategy(str, Enum):
    METADATA_ONLY = "metadata_only"
    TRANSCRIPT_ONLY = "transcript_only"
    TRANSCRIPT_PLUS_KNOWLEDGE = "transcript_plus_knowledge"


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
                frequency_penalty=1.2,
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


def _build_transcript_prompt(
    segments: list[str],
    system: str,
    user_template: str,
    client: OpenAI,
    model: str,
) -> str:
    transcript = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(segments))
    user_msg = user_template.format(n_segments=len(segments), transcript=transcript)
    return _llm_call(client, model, system, user_msg)


def build_profile(
    speaker_id: str,
    strategy: ProfileStrategy,
    n_profile: int,
    company_name: str = "",
    sector: str = "",
    financial_quarter: str = "",
    segments: list[str] | None = None,
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

    elif strategy == ProfileStrategy.TRANSCRIPT_ONLY:
        if not segments:
            raise ValueError("TRANSCRIPT_ONLY strategy requires non-empty segments")
        prompt = _build_transcript_prompt(
            segments, TRANSCRIPT_ONLY_SYSTEM, TRANSCRIPT_ONLY_USER, client, model
        )

    elif strategy == ProfileStrategy.TRANSCRIPT_PLUS_KNOWLEDGE:
        if not segments:
            raise ValueError("TRANSCRIPT_PLUS_KNOWLEDGE strategy requires non-empty segments")
        prompt = _build_transcript_prompt(
            segments, TRANSCRIPT_PLUS_KNOWLEDGE_SYSTEM, TRANSCRIPT_PLUS_KNOWLEDGE_USER, client, model
        )

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
    out_dir = profiles_dir / profile.strategy.value
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{profile.speaker_id}_{profile.n_profile}.json"
    out.write_text(profile.model_dump_json(indent=2))
    logger.info(f"Saved profile → {out}")
    return out


def load_profile(
    speaker_id: str,
    n_profile: int,
    strategy: ProfileStrategy,
    profiles_dir: Path = PROFILES_DIR,
) -> SpeakerProfile:
    path = profiles_dir / strategy.value / f"{speaker_id}_{n_profile}.json"
    return SpeakerProfile.model_validate_json(path.read_text())
