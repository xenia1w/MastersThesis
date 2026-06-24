"""LLM prompt templates for ASR context profile strategies."""

# ── metadata_only ─────────────────────────────────────────────────────────────

METADATA_ONLY_SYSTEM = (
    "You are a financial domain expert assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR."
)

METADATA_ONLY_USER = """\
This is an earnings call for {company_name}, a company operating in the {sector} sector \
(Q{financial_quarter}).
Generate a comma-separated list of named entities, ticker symbols, financial abbreviations, \
executive names, product names, and domain-specific terms likely to appear in this call. \
Focus on rare or uncommon words that ASR systems commonly mis-transcribe.
Limit your list to at most 50 terms so it fits within the ASR system's context window.
Do not repeat any term.
Return only the comma-separated list: no explanation, no numbering."""

# ── transcript_only ───────────────────────────────────────────────────────────

TRANSCRIPT_ONLY_SYSTEM = (
    "You are a financial domain expert assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR."
)

TRANSCRIPT_ONLY_USER = """\
The following is a partial transcript of a financial earnings call ({n_segments} speaker turns):

{transcript}

Based on this transcript, generate a comma-separated list of named entities, ticker symbols, \
financial abbreviations, executive names, product names, and domain-specific terms that appear \
or are likely to appear in this call. \
Focus on rare or uncommon words that ASR systems commonly mis-transcribe.
Limit your list to at most 50 terms so it fits within the ASR system's context window.
Return only the comma-separated list — no explanation, no numbering."""

# ── transcript_plus_knowledge ─────────────────────────────────────────────────

TRANSCRIPT_PLUS_KNOWLEDGE_SYSTEM = (
    "You are a financial domain expert assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR. "
    "Remember that the ASR model might have mis-transcribed named entities. "
    "Use your domain knowledge to verify and correct their spelling where possible."
)

# User prompt is identical to transcript_only — the difference is in the system prompt only.
TRANSCRIPT_PLUS_KNOWLEDGE_USER = TRANSCRIPT_ONLY_USER
