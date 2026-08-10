# Part 2 — Linguistic Prompting (RQ2)

Can a call-level *textual* profile — domain vocabulary, named entities, speaking
style — improve ASR accuracy? An LLM builds the profile, and it is injected into
Whisper Medium at two different points: **prompt-time** (Whisper's
`initial_prompt`) and **post-hoc** (LLM correction of the finished hypothesis).

Dataset: *Earnings21*, 43 calls. All commands run from the **project root**.

---

## Experimental design

The central constraint is that a profile must never see the audio it is scored on.
Two disjoint fixed wall-clock windows enforce this:

| Window | Span | Role |
|---|---|---|
| Profile window | `[0:00, 5:00]` | Transcribed unprompted; the LLM reads this to build the profile |
| Evaluation window | `[5:00, 15:00]` | Transcribed with the profile injected; the only window scored |

References for the eval window come from `manual_annotation.csv` — for each call you
note the phrase heard at 5:00 and at 15:00, which locates exact word indices in the
ordered `.nlp` transcript. The reference is `tokens[i5:i15]`.

**Two orthogonal axes** define a prompting approach:

*Content strategy* — what the LLM sees when building the profile:

| Strategy | LLM input |
|---|---|
| `metadata_only` | Call metadata only (ticker, company, sector) |
| `transcript_only` | Noisy unprompted transcript of the profile window |
| `transcript_plus_knowledge` | Same, plus the LLM's own world knowledge |
| `transcript_metadata_knowledge` | All three combined |

*Prompt format* — how the profile is rendered: `list` (comma-separated keywords) or
`prose` (natural-language passage). Set with `--prompt-format`; capped by
`MAX_PROMPT_TERMS` / `MAX_PROMPT_WORDS` in `models/constants.py` so the result always
fits Whisper's ~224-token prompt window.

**Post-hoc correction** is the alternative injection point — decoding is left alone
and the LLM repairs the raw hypothesis instead:

| Mode | Flag | What the LLM sees alongside the chunk |
|---|---|---|
| `blind` | *(default)* | Nothing — the eval chunk only |
| `context` | `--use-context` | The call's own unprompted `[0:00, 5:00]` transcript, as a hint |
| `profile` | `--use-profile` | A built profile, treated as authoritative for spellings |

Correction is chunked at sentence boundaries (~200 words) to keep verbatim
discipline, and an edit-ratio guard reverts to the raw hypothesis if the LLM
rewrote too much.

---

## Setup

Profile building and post-hoc correction call the KISSKI/SAIA academic-cloud LLM
(`qwen3-30b-a3b-instruct-2507`) and run **locally**, not on the cluster. Put a key in
`.env`:

```
KISSKI_API_KEY=...        # or SAIA_API_KEY
```

The endpoint enforces ~10 requests/min; the client self-throttles to 9/min and
retries on 429 with header-driven back-off, so long batches pace themselves.

Whisper transcription is the expensive part and runs on the **cluster** via SLURM.

---

## Pipeline

### 1. Transcribe the profile window

Needed only for the `transcript_*` strategies — `metadata_only` skips this step.

```bash
sbatch --array=0-42 src/linguistic_prompting/slurm/run_earnings21_v2_profile_window.sh
```

Locally, for one call:
```bash
uv run python -m src.linguistic_prompting.pipeline.earnings21_window_profile \
    --data-dir   data/raw/earnings21 \
    --output-dir data/processed/linguistic_prompting/v2/profile_transcripts \
    --call-id    4320211 \
    --window-seconds 300
```

Output: `v2/profile_transcripts/<call_id>_300.json`

### 2. Build profiles

One invocation per (strategy, format) pair:

```bash
uv run python -m src.linguistic_prompting.pipeline.build_earnings21_profiles \
    --data-dir      data/raw/earnings21 \
    --profiles-dir  data/processed/linguistic_prompting/v2/profiles \
    --strategy      transcript_metadata_knowledge \
    --prompt-format list \
    --n-profile     300 \
    --skip-existing
```

Output: `v2/profiles/<strategy>/<call_id>_300.json`

> `--n-profile 300` is the window in **seconds** for the v2 methodology, and it
> becomes the `<tag>` in the filename. Downstream steps default to `--profile-tag 300`;
> the two must agree or the profile won't be found.

### 3. Transcribe the evaluation window

Baseline (no prompt) — run once:
```bash
sbatch --array=0-42 src/linguistic_prompting/slurm/run_earnings21_v2_baseline.sh
```

Prompted — one submission per strategy:
```bash
CALL_IDS=data/raw/earnings21/call_ids.txt
N=$(( $(wc -l < "$CALL_IDS") - 1 ))
for S in metadata_only transcript_only transcript_plus_knowledge transcript_metadata_knowledge; do
  sbatch --array=0-${N} \
         --export=ALL,CALL_IDS_FILE=${CALL_IDS},STRATEGY=${S},PROMPT_FORMAT=list \
         src/linguistic_prompting/slurm/run_earnings21_v2_prompted.sh
done
```

Set `PROMPT_FORMAT=prose` for the prose variants (profiles must have been built with
`--prompt-format prose`). Cluster jobs save **hypotheses only** — no scoring — so one
array task writes `prompted_<call_id>.csv`. Merge per strategy afterwards:

```bash
uv run python -c "
import glob, pandas as pd
d='data/processed/linguistic_prompting/v2/earnings21_window_transcript_only'
dfs=[pd.read_csv(f) for f in sorted(glob.glob(d+'/prompted_[0-9]*.csv'))]
pd.concat(dfs).to_csv(d+'/prompted_all.csv', index=False); print('Merged', len(dfs))
"
```

### 4. Post-hoc correction (alternative to step 3's prompting)

Runs locally against the LLM, reading the **baseline** hypotheses:

```bash
# blind
uv run python -m src.linguistic_prompting.pipeline.earnings21_posthoc_correct --skip-existing

# self-contained context
uv run python -m src.linguistic_prompting.pipeline.earnings21_posthoc_correct --use-context

# profile-conditioned (choose which built profile)
uv run python -m src.linguistic_prompting.pipeline.earnings21_posthoc_correct \
    --use-profile --profile-strategy transcript_metadata_knowledge
```

Writes to `v2/earnings21_window_posthoc_<mode>/` in the same shape scoring expects.
Profile mode appends the strategy name to the directory so runs never collide.

### 5. Score locally

Builds references from the annotations, then scores baseline plus every approach:

```bash
uv run python -m src.linguistic_prompting.pipeline.earnings21_window_score \
    --data-dir    data/raw/earnings21 \
    --annotations manual_annotation.csv
```

Output: `v2/scores/<approach>_scored.csv` (per call: `wer`, `entity_eer`,
`ref_words`, `n_entity_tokens`, `n_entity_errors`) and `cross_approach_summary.csv`.

Restrict to specific approaches with repeated `--approach` flags.

---

## Analysis

```bash
# Every approach vs baseline on one common call set, with paired Wilcoxon p-values
uv run python -m src.linguistic_prompting.comparison.compare_all

# One approach in detail: per-call table + micro/macro deltas
uv run python -m src.linguistic_prompting.comparison.compare_approach

# One call, word-level: which reference tokens the prompt fixed vs degraded
uv run python -m src.linguistic_prompting.comparison.compare_call
```

Two averaging conventions are reported throughout: **micro** (corpus-level,
word-weighted) and **macro** (mean of per-call values). They diverge when call
lengths vary, so the thesis reports both.

`pipeline/inspect_entities.py` is a diagnostic that prints, for a single call, which
entity tokens were counted as errors and why.

---

## Metrics

- **WER** — word error rate over the eval window, via `jiwer` with shared normalisation.
- **Entity-EER** — error rate restricted to reference tokens tagged as entities in the
  `.nlp` file. This is the metric prompting is *meant* to move: a profile supplies
  proper nouns and domain terms, so gains should concentrate on entity tokens even
  when overall WER barely shifts.

---

## Layout

```
src/linguistic_prompting/
├── models/
│   ├── constants.py            # paths, LLM endpoint, rate limits, prompt budgets
│   ├── prompts.py              # system/user templates per strategy + format rules
│   └── speaker_profile.py      # build/cache profiles, KISSKI client
├── pipeline/
│   ├── build_earnings21_profiles.py    # step 2
│   ├── earnings21_window_profile.py    # step 1 (v2 fixed window)
│   ├── earnings21_window_eval.py       # step 3 (v2 fixed window)
│   ├── earnings21_window_score.py      # step 5, local scoring
│   ├── earnings21_posthoc_correct.py   # step 4, three modes
│   ├── earnings21_profile_window.py    # v1: turn-based profile window
│   ├── earnings21_fullcall_eval.py     # v1: whole-call evaluation
│   ├── compare_fullcall.py             # v1 comparison
│   ├── inspect_entities.py             # entity-EER diagnostic
│   └── smoke_test_fullcall.py          # local pre-flight before a SLURM job
├── comparison/                 # loaders, metrics, word-level diffs
└── slurm/
    ├── run_earnings21_v2_profile_window.sh
    ├── run_earnings21_v2_baseline.sh
    ├── run_earnings21_v2_prompted.sh    # parametrised by STRATEGY, PROMPT_FORMAT
    └── run_earnings21_fullcall_*.sh     # v1 methodology
```

> **v1 vs v2.** The `fullcall` / `profile_window` modules are the earlier methodology:
> profiles built from the first *n turns* and the whole call evaluated. It carried the
> profile through the entire decode, which triggered hallucination loops. v2 replaced
> it with the fixed wall-clock windows described above. v1 code is kept because the
> thesis reports the comparison; new work should use the `window` modules.
