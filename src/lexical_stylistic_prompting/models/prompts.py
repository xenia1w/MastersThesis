"""LLM prompt templates for ASR context-profile strategies.

Four *content* strategies decide what information the LLM sees; an orthogonal *format*
(list vs prose) decides the surface form of the text injected into Whisper's
``initial_prompt``. The format instruction is factored into the ``*_FORMAT_RULES``
constants below and appended to every user prompt, so a strategy can be rendered as
either a comma-separated keyword list or a natural-language passage while everything
else about the prompt stays identical (the RQ2.1 format comparison).

Content strategies:
- ``metadata_only``: the LLM sees only the company name, sector, and fiscal quarter.
- ``transcript_only``: the LLM sees a noisy, unprompted Whisper transcript of the call's
  first ``n_profile`` speaker turns and extracts terms from it (data-dependent).
- ``transcript_plus_knowledge``: identical user prompt to ``transcript_only``; the system
  prompt additionally asks the LLM to correct the spelling of mis-transcribed entities.
- ``transcript_metadata_knowledge``: transcript + company metadata + spelling correction.
"""

# ── output-format rules (appended verbatim to every user prompt) ───────────────
# The trailing instruction that fixes the surface form of the initial_prompt. Keeping
# it as a swappable constant means list vs prose is the ONLY thing that changes across
# the RQ2.1 comparison — the prompt bodies above are format-neutral ("identify …").

LIST_FORMAT_RULES = """\
Rules:
- Output ONLY a single comma-separated list of terms.
- Each item is a short term (1-4 words). No sentences, no explanations, no numbering.
- At most {max_terms} terms. Do not repeat any term."""

PROSE_FORMAT_RULES = """\
Rules:
- Output ONLY one short, natural-sounding passage (1-3 sentences) written in the fluent \
spoken style of an earnings-call speaker, as if it were the opening words of the call.
- Weave the rare or domain-specific terms (company, product/brand names, ticker symbol, \
subsidiaries, financial abbreviations) naturally into the sentences, so each appears in \
realistic spoken context rather than as a bare label.
- Ground every detail in the information provided above. Do NOT invent or guess numbers, \
dates, percentages, statistics, tickers, or product/brand names that are not given above.
- Only state a person's name if that exact name appears in the information above; otherwise \
refer to them by role ("our CEO", "the CFO"). Never use bracketed placeholders such as \
"[Name]" — if a detail is unknown, phrase around it or leave it out.
- Write flowing prose only: no lists, bullet points, headings, numbering, or explanations.
- At most {max_words} words."""

# ── metadata_only ─────────────────────────────────────────────────────────────

METADATA_ONLY_SYSTEM = (
    "You are a financial-domain expert assisting an automatic speech recognition (ASR) "
    "system. Your output is used verbatim as Whisper's initial_prompt to bias decoding "
    "toward the call's domain vocabulary."
)

METADATA_ONLY_USER = """\
This is an earnings call for {company_name}, a company in the {sector} sector (Q{financial_quarter}).

Identify the rare or domain-specific terms most likely to be spoken in this call and most likely \
to be mis-transcribed by ASR: the company name, its ticker symbol, executive names, product/brand \
names, subsidiaries, and financial abbreviations.

{format_rules}
"""

# ── transcript_only ───────────────────────────────────────────────────────────

TRANSCRIPT_ONLY_SYSTEM = (
    "You are a financial domain expert assisting an automatic speech recognition system. "
    "Your output will be used verbatim as an initial context prompt for Whisper ASR."
)

TRANSCRIPT_ONLY_USER = """\
The following is a partial transcript of a financial earnings call ({n_segments} speaker turns):

{transcript}

Based on this transcript, identify the named entities, ticker symbols, financial abbreviations, \
executive names, product names, and domain-specific terms that appear or are likely to appear in \
this call. Focus on rare or uncommon words that ASR systems commonly mis-transcribe.

{format_rules}
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

# ── transcript_metadata_knowledge ─────────────────────────────────────────────
# Combines the noisy transcript, the company metadata, and the spelling-correction
# instruction (knowledge system prompt) in a single request.

TRANSCRIPT_METADATA_KNOWLEDGE_SYSTEM = TRANSCRIPT_PLUS_KNOWLEDGE_SYSTEM

TRANSCRIPT_METADATA_KNOWLEDGE_USER = """\
This is an earnings call for {company_name}, a company in the {sector} sector (Q{financial_quarter}).

The following is a partial transcript of this call ({n_segments} speaker turns):

{transcript}

Using BOTH the company metadata above and the transcript, identify the named entities, ticker \
symbols, financial abbreviations, executive names, product names, and domain-specific terms that \
appear or are likely to appear in this call. Focus on rare or uncommon words that ASR systems \
commonly mis-transcribe.

{format_rules}
"""