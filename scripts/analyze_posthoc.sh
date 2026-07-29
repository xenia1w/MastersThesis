#!/usr/bin/env bash
# Re-score and summarise the post-hoc LLM-correction experiment (RQ2).
# Safe to run live: always re-scores against the freshly merged prompted_all.csv,
# so the numbers can never be stale. Usage: bash scripts/analyze_posthoc.sh
set -euo pipefail
cd "$(dirname "$0")/.."

V2=data/processed/lexical_stylistic_prompting/v2
POSTHOC="$V2/earnings21_window_posthoc_blind"

echo "==> 1/3  re-merging per-call post-hoc files (guards against a partial merge)"
uv run python - <<'PY'
import glob, pandas as pd
d = "data/processed/lexical_stylistic_prompting/v2/earnings21_window_posthoc_blind"
files = sorted(glob.glob(f"{d}/posthoc_[0-9]*.csv"))
pd.concat([pd.read_csv(f) for f in files]).to_csv(f"{d}/prompted_all.csv", index=False)
print(f"    merged {len(files)} calls -> prompted_all.csv")
PY

echo "==> 2/3  re-scoring baseline vs post-hoc against the golden references"
uv run -m src.lexical_stylistic_prompting.pipeline.earnings21_window_score \
    --annotations manual_annotation.csv \
    --approach posthoc_blind

echo "==> 3/3  summary"
uv run python - <<'PY'
import pandas as pd
V2 = "data/processed/lexical_stylistic_prompting/v2"
P  = f"{V2}/earnings21_window_posthoc_blind/prompted_all.csv"
S  = f"{V2}/scores/cross_approach_summary.csv"

g = pd.read_csv(P)
print("\n--- how much did the LLM actually change? ---")
print(f"  calls              : {len(g)}")
print(f"  guard reverted     : {int(g.guard_applied.sum())} ({g.guard_applied.mean()*100:.0f}%)")
print(f"  edit_ratio med/max : {g.edit_ratio.median():.3f} / {g.edit_ratio.max():.3f}")

s = pd.read_csv(S)
print("\n--- WER / Entity Error Rate: baseline -> post-hoc ---")
for _, r in s.iterrows():
    print(f"  [{r.approach}]  n={int(r.n)}")
    print(f"    WER  {r.wer_base:.4f} -> {r.wer_prom:.4f}  (dmicro {r.wer_d_micro:+.4f})  p={r.p_wer:.3f}")
    print(f"    EER  {r.eer_base:.4f} -> {r.eer_prom:.4f}  (dmicro {r.eer_d_micro:+.4f})  p={r.p_eer:.3f}")
print("\n  (negative delta = improvement; p<0.05 = statistically significant)")
PY
