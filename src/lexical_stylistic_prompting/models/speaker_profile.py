"""
Build a speaker profile from profile-set transcripts and cache it as JSON.

Four strategies:
  free_form         — LLM summarises vocabulary, domain, register (prose description)
  keyword_list      — LLM returns a comma-separated list of observed domain terms / rare words
  keyword_expansion — LLM returns observed terms + predicted related terms from domain knowledge
  raw_context       — no LLM; profile segments concatenated directly

Usage:
  uv run -m src.lexical_stylistic_prompting.models.speaker_profile \\
      --speaker <speaker_id> \\
      --transcripts-file path/to/transcripts.txt \\
      --strategy free_form keyword_list keyword_expansion raw_context \\
      --n-profile 20
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROFILES_DIR = Path("data/processed/lexical_stylistic_prompting/profiles")

# ── KISSKI / SAIA client ──────────────────────────────────────────────────────
KISSKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL   = "meta-llama-3.1-8b-instruct"


class ProfileStrategy(str, Enum):
    FREE_FORM         = "free_form"
    KEYWORD_LIST      = "keyword_list"
    KEYWORD_EXPANSION = "keyword_expansion"
    RAW_CONTEXT       = "raw_context"


class SpeakerProfile(BaseModel):
    speaker_id: str
    n_profile:  int
    strategy:   ProfileStrategy
    model:      str | None       # None for raw_context
    prompt:     str
    created_at: str


# ── LLM prompts ───────────────────────────────────────────────────────────────
_FREE_FORM_SYSTEM = (
    "You are an expert linguist assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR."
)
_FREE_FORM_USER = """\
Below are transcribed speech segments from a single speaker.
Write a concise description (2-3 sentences) of this speaker's domain, vocabulary style, \
and recurring technical terms or named entities.
Return only the description — no preamble, no bullet points.

Transcripts:
{transcripts}"""

_KEYWORD_LIST_SYSTEM = (
    "You are an expert linguist assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR."
)
_KEYWORD_LIST_USER = """\
Below are transcribed speech segments from a single speaker.
Extract a comma-separated list of domain-specific terms, named entities, \
technical vocabulary, and rare words that characterise this speaker.
Return only the comma-separated list — no explanation, no numbering.

Transcripts:
{transcripts}"""


_KEYWORD_EXPANSION_SYSTEM = (
    "You are an expert linguist assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR."
)
_KEYWORD_EXPANSION_USER = """\
Below are transcribed speech segments from a single speaker.
First extract domain-specific terms, named entities, technical vocabulary, and rare words \
that appear in these transcripts.
Then use your domain knowledge to add further related technical terms that this speaker \
is likely to use, even if not explicitly mentioned in the transcripts.
Return only a single comma-separated list combining both observed and predicted terms \
— no explanation, no labels, no numbering.

Transcripts:
{transcripts}"""


def _get_client() -> OpenAI:
    api_key = os.environ.get("KISSKI_API_KEY") or os.environ.get("SAIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set KISSKI_API_KEY or SAIA_API_KEY in your environment or .env file."
        )
    return OpenAI(base_url=KISSKI_BASE_URL, api_key=api_key)


def _llm_call(client: OpenAI, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    assert content is not None, "LLM returned empty content"
    return content.strip()


def build_profile(
    speaker_id:  str,
    transcripts: list[str],
    strategy:    ProfileStrategy,
    n_profile:   int,
    model:       str = DEFAULT_MODEL,
    client:      OpenAI | None = None,
) -> SpeakerProfile:
    """Build a single profile for (speaker, strategy, n_profile)."""
    texts = transcripts[:n_profile]
    joined = "\n".join(f"- {t}" for t in texts)

    if strategy == ProfileStrategy.RAW_CONTEXT:
        prompt = " ".join(texts)
        used_model = None
    else:
        if client is None:
            client = _get_client()
        if strategy == ProfileStrategy.FREE_FORM:
            prompt = _llm_call(client, model, _FREE_FORM_SYSTEM,
                               _FREE_FORM_USER.format(transcripts=joined))
        elif strategy == ProfileStrategy.KEYWORD_LIST:
            prompt = _llm_call(client, model, _KEYWORD_LIST_SYSTEM,
                               _KEYWORD_LIST_USER.format(transcripts=joined))
        else:  # KEYWORD_EXPANSION
            prompt = _llm_call(client, model, _KEYWORD_EXPANSION_SYSTEM,
                               _KEYWORD_EXPANSION_USER.format(transcripts=joined))
        used_model = model

    return SpeakerProfile(
        speaker_id=speaker_id,
        n_profile=n_profile,
        strategy=strategy,
        model=used_model,
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
    speaker_id:  str,
    n_profile:   int,
    strategy:    ProfileStrategy,
    profiles_dir: Path = PROFILES_DIR,
) -> SpeakerProfile:
    path = profiles_dir / f"{speaker_id}_{n_profile}_{strategy.value}.json"
    return SpeakerProfile.model_validate_json(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache speaker profiles")
    parser.add_argument("--speaker",          required=True, help="Speaker ID")
    parser.add_argument("--transcripts-file", required=True,
                        help="Text file with one profile-segment transcript per line")
    parser.add_argument("--strategy",         nargs="+",
                        choices=[s.value for s in ProfileStrategy],
                        default=[s.value for s in ProfileStrategy],
                        help="One or more strategies (default: all)")
    parser.add_argument("--n-profile",        type=int, default=20,
                        help="Number of profile segments to use (first N lines of --transcripts-file)")
    parser.add_argument("--kisski-model",     default=DEFAULT_MODEL,
                        help="Model name on KISSKI/SAIA")
    parser.add_argument("--profiles-dir",     default=str(PROFILES_DIR),
                        help="Output directory for profile JSON files")
    parser.add_argument("--skip-existing",    action="store_true",
                        help="Skip strategies that already have a cached profile")
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    strategies   = [ProfileStrategy(s) for s in args.strategy]

    transcripts_path = Path(args.transcripts_file)
    transcripts = [l.strip() for l in transcripts_path.read_text().splitlines() if l.strip()]
    transcripts = transcripts[:args.n_profile]
    logger.info(f"Loaded {len(transcripts)} profile transcripts for {args.speaker}")

    client = _get_client()

    for strategy in strategies:
        out_path = profiles_dir / f"{args.speaker}_{args.n_profile}_{strategy.value}.json"
        if args.skip_existing and out_path.exists():
            logger.info(f"Skipping {strategy.value} (cached)")
            continue

        logger.info(f"Building profile: strategy={strategy.value} ...")
        profile = build_profile(
            speaker_id=args.speaker,
            transcripts=transcripts,
            strategy=strategy,
            n_profile=args.n_profile,
            model=args.kisski_model,
            client=client if strategy != ProfileStrategy.RAW_CONTEXT else None,
        )
        save_profile(profile, profiles_dir)
        logger.info(f"  prompt preview: {profile.prompt[:120]} ...")


if __name__ == "__main__":
    main()
