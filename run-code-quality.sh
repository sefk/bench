#!/bin/bash
# Grade every variant on the LiveCodeBench item set, one variant at a time.
#
# Slow: Qwen writes ~1,500 tokens per problem even with reasoning off, so 175
# problems take ~1.5 h on a MoE build, ~5 h on a 4-bit dense build, and ~7 h on
# an 8-bit dense one. Variants are ordered by what they tell you per hour, and
# the output file resumes, so stopping and re-running loses at most one item.
#
#   ./run-code-quality.sh                                   # default queue
#   ./run-code-quality.sh qwen/qwen3.6-27b@8bit             # just these
#   OUTDIR=results/2026-09-17 ./run-code-quality.sh ...     # resume into a dir
#   BUDGET=16384 OUTDIR=results/2026-09-17 ./run-code-quality.sh qwen/qwen3.6-35b-a3b@4bit
#
# BUDGET raises the token cap. At the default 4096 about half the hard
# problems are cut off before any code is written. A larger budget writes to
# its own file (quality-code-16k.jsonl), seeded with each variant's answers
# from the 4k file, and only the answers truncated there are regenerated: at
# temperature 0 an answer that finished under 4k would not change. The 4k
# results are left as they are, so the two budgets can be compared.
#
# Needs quality/code-tests/, built by quality/make-code-items.py.

set -u
cd "$(dirname "$0")"

OUTDIR=${OUTDIR:-results/$(date +%F)}
BUDGET=${BUDGET:-4096}
BASE="$OUTDIR/quality-code.jsonl"
if [ "$BUDGET" = 4096 ]; then
  OUT="$BASE"
  BUDGET_ARGS=()
else
  OUT="$OUTDIR/quality-code-$((BUDGET / 1024))k.jsonl"
  BUDGET_ARGS=(--max-tokens "$BUDGET")
fi
SUMMARY="${OUT%.jsonl}.json"

if [ $# -gt 0 ]; then
  VARIANTS=("$@")
else
  # MoE pair first (the quantization question, fast), then the 4-bit dense
  # pair (MoE vs dense, and 3.6 vs 3.8), then Apple's model. The 8-bit dense
  # builds are ~7 h each and are left to be asked for by name.
  VARIANTS=(
    "qwen/qwen3.6-35b-a3b@4bit"
    "qwen/qwen3.6-35b-a3b@8bit"
    "qwen/qwen3.6-27b@4bit"
    "qwen/qwen3.8-27b@4bit"
    "fm:system"
  )
fi

mkdir -p "$OUTDIR"
echo "code quality: ${#VARIANTS[@]} variants, budget $BUDGET -> $OUT"

# Copy a variant's 4k answers into the larger-budget file the first time it is
# run there, so only the truncated ones are regenerated.
seed_from_base() {
  [ "$OUT" = "$BASE" ] && return
  [ -f "$BASE" ] || return
  python3 - "$BASE" "$OUT" "$1" <<'PY'
import json, sys
from pathlib import Path
base, out, model = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
existing = out.read_text().splitlines() if out.exists() else []
if any(json.loads(l).get("model") == model for l in existing if l.strip()):
    sys.exit(0)
rows = [l for l in base.read_text().splitlines()
        if l.strip() and json.loads(l).get("model") == model]
with out.open("a") as fh:
    fh.writelines(l + "\n" for l in rows)
print(f"  seeded {len(rows)} answers for {model} from {base.name}")
PY
}

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    fm:*) target="$variant" ;;
    *) target="lmstudio:$variant" ;;
  esac
  seed_from_base "${target#*:}"
  echo ""
  echo "=== $target $(date +%T) ==="
  ./quality-bench --task code "${BUDGET_ARGS[@]+"${BUDGET_ARGS[@]}"}" "$target" -o "$OUT" \
    || echo "  FAILED: $target"
  # Refresh the summary after every variant, so the dashboard shows results
  # as they land rather than only at the end.
  ./quality-bench --score "$OUT" --json > "$SUMMARY"
done

lms unload --all >/dev/null 2>&1
echo ""
echo "code quality complete $(date +%T)"
./quality-bench --score "$OUT"
