"""LLM prompt templates for ASR context-profile strategies.

Only ``metadata_only`` is implemented in this fresh rewrite: given a company's
name, sector, and fiscal quarter, the LLM emits a short, deduplicated keyword
list of likely rare/entity terms. That list is injected into Whisper via
``initial_prompt`` to softly bias decoding toward the domain vocabulary.
"""

# ── metadata_only ─────────────────────────────────────────────────────────────

METADATA_ONLY_SYSTEM = (
    "You are a financial-domain expert assisting an automatic speech recognition (ASR) "
    "system. Your output is used verbatim as Whisper's initial_prompt, so it must be a "
    "compact keyword list — never prose."
)

METADATA_ONLY_USER = """\
This is an earnings call for {company_name}, a company in the {sector} sector (Q{financial_quarter}).

List the rare or domain-specific terms most likely to be spoken in this call and most likely \
to be mis-transcribed by ASR: the company name, its ticker symbol, executive names, product/brand \
names, subsidiaries, and financial abbreviations.

Rules:
- Output ONLY a single comma-separated list of terms.
- Each item is a short term (1-4 words). No sentences, no explanations, no numbering.
- At most {max_terms} terms. Do not repeat any term.
"""
