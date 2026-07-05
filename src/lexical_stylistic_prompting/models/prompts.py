"""LLM prompt templates for ASR context-profile strategies.

Three strategies, all producing a short, deduplicated keyword list that is injected into
Whisper via ``initial_prompt`` to softly bias decoding toward the domain vocabulary:

- ``metadata_only``: the LLM sees only the company name, sector, and fiscal quarter.
- ``transcript_only``: the LLM sees a noisy, unprompted Whisper transcript of the call's
  first ``n_profile`` speaker turns and extracts terms from it (data-dependent).
- ``transcript_plus_knowledge``: identical user prompt to ``transcript_only``; the system
  prompt additionally asks the LLM to correct the spelling of mis-transcribed entities.
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
or are likely to appear in this call. Focus on rare or uncommon words that ASR systems commonly \
mis-transcribe.

Rules:
- Output ONLY a single comma-separated list of terms.
- Each item is a short term (1-4 words). No sentences, no explanations, no numbering.
- At most {max_terms} terms. Do not repeat any term.
"""

# ── transcript_plus_knowledge ─────────────────────────────────────────────────
# Identical user prompt; the system prompt adds a spelling-correction instruction so the
# LLM does not simply echo Whisper's mis-transcriptions from the noisy input transcript.

TRANSCRIPT_PLUS_KNOWLEDGE_SYSTEM = (
    "You are a financial domain expert assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR. "
    "Remember that the ASR model might have mis-transcribed named entities. "
    "Use your domain knowledge to verify and correct their spelling where possible."
)

TRANSCRIPT_PLUS_KNOWLEDGE_USER = TRANSCRIPT_ONLY_USER
