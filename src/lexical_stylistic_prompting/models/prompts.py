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


# ── post-hoc correction (RQ2 extension) ────────────────────────────────────────
# Conservative, verbatim ASR error-correction of a single (1-best) Whisper hypothesis.
# The reference is verbatim .nlp text (disfluencies, exact spoken wording), so the model
# must NOT rewrite/clean up — only fix clear recognition errors. See arXiv 2409.09554 /
# 2307.04172: over-correction (rewriting already-correct words) is the main failure mode
# with a single hypothesis, so edit freedom is tightly constrained.

POSTHOC_CORRECT_SYSTEM = """\
You correct automatic-speech-recognition (ASR) errors in earnings-call transcripts.

You are given a chunk of a raw ASR transcript. Return the SAME text with only genuine \
recognition errors fixed. This is a verbatim transcript, not an edit for readability.

Rules:
- Preserve wording exactly: same words, same order. Do NOT paraphrase, rephrase, summarize, \
translate, reorder, or "clean up" grammar.
- Keep disfluencies and false starts verbatim (um, uh, "we-we", repeated words, incomplete \
sentences). Do NOT remove filler words.
- Do NOT add or delete content. The output length must closely match the input.
- Only fix words that were clearly MIS-HEARD: misrecognized common words, company/product/person \
names, ticker symbols, financial terms and abbreviations, and numbers.
- Keep the same casing and punctuation style as the input; do not re-punctuate.
- If a passage is already correct, return it unchanged.
- Output ONLY the corrected transcript text, with no preamble, notes, or quotation marks."""

POSTHOC_CORRECT_USER = """\
Raw ASR transcript chunk:

{chunk}"""

# Self-contained "context" variant: the same strict verbatim correction, but the chunk is
# accompanied by a REFERENCE transcript — the unprompted Whisper output for the first five
# minutes (0:00-5:00) of the SAME call. That preamble (operator script, speaker introductions,
# company name) is entity-dense, so it hints at how names/terms recur in this call. Crucially
# the reference is itself raw ASR and may carry the same errors, so the prompt forbids copying
# from it and treats it as a hint only. This is the post-hoc counterpart of the transcript_only
# prompting method (same 0-5 transcript, injected after decoding instead of via initial_prompt).

POSTHOC_CONTEXT_SYSTEM = """\
You correct automatic-speech-recognition (ASR) errors in earnings-call transcripts.

You are given a chunk of a raw ASR transcript to correct, together with a REFERENCE transcript \
of the opening of the SAME call (its first few minutes). The reference is itself raw ASR and may \
contain its own errors. Use it ONLY as a hint for how company names, people, product and ticker \
names, and recurring terms appear in this call. Do NOT copy from the reference and do NOT insert \
its words where they were not spoken in the chunk.

Return the SAME chunk text with only genuine recognition errors fixed. This is a verbatim \
transcript, not an edit for readability.

Rules:
- Preserve wording exactly: same words, same order. Do NOT paraphrase, rephrase, summarize, \
translate, reorder, or "clean up" grammar.
- Keep disfluencies and false starts verbatim (um, uh, "we-we", repeated words, incomplete \
sentences). Do NOT remove filler words.
- Do NOT add or delete content. The output length must closely match the input.
- Only fix words that were clearly MIS-HEARD: misrecognized common words, company/product/person \
names, ticker symbols, financial terms and abbreviations, and numbers.
- Keep the same casing and punctuation style as the input; do not re-punctuate.
- If a passage is already correct, return it unchanged.
- Output ONLY the corrected transcript chunk, with no preamble, notes, or quotation marks."""

POSTHOC_CONTEXT_USER = """\
Reference transcript (first minutes of the same call, raw ASR — may contain errors, use as a hint only):

{reference}

Raw ASR transcript chunk to correct:

{chunk}"""

# Profile variant: the chunk is corrected using a CURATED entity profile rather than the raw
# 0:00-5:00 transcript. The profile is the LLM-built, spell-corrected transcript_metadata_knowledge
# list for the SAME call (company, ticker, executive, product names, financial abbreviations). Unlike
# the raw-transcript context reference, this profile is authoritative: it is the corrector's canonical
# spelling of exactly the entities that drive entity error rate, so — the opposite of the context
# prompt — the model is told to TRUST it and adopt its spellings for any entity it recognises as
# mis-heard. This is the post-hoc counterpart of injecting the same profile into Whisper's
# initial_prompt (RQ2: one profile, two injection points — decode-time vs post-decode correction).

POSTHOC_PROFILE_SYSTEM = """\
You correct automatic-speech-recognition (ASR) errors in earnings-call transcripts.

You are given a chunk of a raw ASR transcript to correct, together with a PROFILE of this call: a \
curated list of the company, ticker, executive, product, and domain-specific terms it contains, \
with their correct spelling. Treat the profile as authoritative: when a word in the chunk is a \
mis-heard version of a profile term, replace it with the profile's spelling.  \

Return the SAME chunk text with only genuine recognition errors fixed. This is a verbatim \
transcript, not an edit for readability.

Rules:
- Preserve wording exactly: same words, same order. Do NOT paraphrase, rephrase, summarize, \
translate, reorder, or "clean up" grammar.
- Keep disfluencies and false starts verbatim (um, uh, "we-we", repeated words, incomplete \
sentences). Do NOT remove filler words.
- Do NOT add or delete content. The output length must closely match the input.
- Only fix words that were clearly MIS-HEARD: misrecognized common words, and especially \
company/product/person names, ticker symbols, financial terms and abbreviations, and numbers — \
using the profile's spelling for any entity it lists.
- Keep the same casing and punctuation style as the input; do not re-punctuate.
- If a passage is already correct, return it unchanged.
- Output ONLY the corrected transcript chunk, with no preamble, notes, or quotation marks."""

POSTHOC_PROFILE_USER = """\
Profile of this call (authoritative spellings of the entities and terms it contains):

{profile}

Raw ASR transcript chunk to correct:

{chunk}"""

