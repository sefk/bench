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
#
# Needs quality/code-tests/, built by quality/make-code-items.py.

set -u
cd "$(dirname "$0")"

OUTDIR=${OUTDIR:-results/$(date +%F)}
OUT="$OUTDIR/quality-code.jsonl"

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
echo "code quality: ${#VARIANTS[@]} variants -> $OUT"

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    fm:*) target="$variant" ;;
    *) target="lmstudio:$variant" ;;
  esac
  echo ""
  echo "=== $target $(date +%T) ==="
  ./quality-bench --task code "$target" -o "$OUT" || echo "  FAILED: $target"
  # Refresh the summary after every variant, so the dashboard shows results
  # as they land rather than only at the end.
  ./quality-bench --score "$OUT" --json > "$OUTDIR/quality-code.json"
done

lms unload --all >/dev/null 2>&1
echo ""
echo "code quality complete $(date +%T)"
./quality-bench --score "$OUT"
