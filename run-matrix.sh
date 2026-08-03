#!/bin/bash
# Benchmark every quantization variant of every model, one at a time.
#
# `lms load` cannot address a variant -- it only matches base model keys and
# silently loads whichever variant is "selected". The REST API can: passing
# "qwen/qwen3.6-27b@8bit" as the model id JIT-loads exactly that build. So the
# variant id goes straight to lmstudio-bench, whose own unload_all() drains
# resident models first. That matters because only one large model fits on
# 64 GB; loading a second trips LM Studio's guardrail with an HTTP 400.
#
# One invocation per variant, so a failure late in the matrix does not cost the
# results already collected.

set -u
cd "$(dirname "$0")"

SIZES=${SIZES:-0,1000,4000,16000}
RUNS=${RUNS:-3}
BENCH=./lmstudio-bench

# Fastest first, so results land early if the run is interrupted.
VARIANTS=(
  "qwen/qwen3.6-27b@4bit"
  "qwen/qwen3.6-35b-a3b@4bit"
  "qwen/qwen3.6-35b-a3b@8bit"
  "qwen/qwen3.6-27b@8bit"
)

mkdir -p results
echo "matrix: ${#VARIANTS[@]} variants, sizes=$SIZES runs=$RUNS"

for variant in "${VARIANTS[@]}"; do
  slug="$(echo "$variant" | sed 's|.*/||; s|@|-|')"
  echo ""
  echo "=== $variant -> results/$slug.json ==="
  start=$(date +%s)

  if $BENCH "$variant" -s "$SIZES" -n "$RUNS" --json \
      > "results/$slug.json" 2> "results/$slug.log"; then
    rows=$(python3 -c "import json;print(len(json.load(open('results/$slug.json'))))" 2>/dev/null || echo 0)
    echo "  done in $(( ($(date +%s) - start) / 60 ))m, $rows rows"
    grep -E "failed|SKIPPED" "results/$slug.log" | sed 's/^/  /'
  else
    echo "  FAILED -- see results/$slug.log"
    tail -3 "results/$slug.log" | sed 's/^/  /'
  fi
done

lms unload --all >/dev/null 2>&1
echo ""
echo "matrix complete"
