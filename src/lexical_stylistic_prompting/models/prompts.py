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
Return only the comma-separated list — no explanation, no numbering."""
