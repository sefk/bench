#!/bin/bash
# Benchmark every quantization variant of every model, one at a time.
#
# JIT loading cannot select a variant -- the API exposes only base model ids --
# so each variant is loaded explicitly here and lmstudio-bench is told not to
# unload it. Only one large model fits on 64 GB, so nothing may be resident
# when the next one loads or LM Studio's guardrail rejects it with an HTTP 400.

set -u
cd "$(dirname "$0")"

SIZES=${SIZES:-0,1000,4000,16000}
RUNS=${RUNS:-3}
CTX=${CTX:-32768}
BENCH=./lmstudio-bench

# Fastest first, so results land early if the run is interrupted.
VARIANTS=(
  "qwen/qwen3.6-27b@4bit"
  "qwen/qwen3.6-35b-a3b@4bit"
  "qwen/qwen3.6-35b-a3b@8bit"
  "qwen/qwen3.6-27b@8bit"
)

resident() { lms ps --json 2>/dev/null | python3 -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?"; }

drain() {
  lms unload --all >/dev/null 2>&1
  for _ in $(seq 30); do
    [ "$(resident)" = "0" ] && return 0
    sleep 2
  done
  echo "  WARN: models still resident after unload: $(resident)"
}

mkdir -p results
echo "matrix: ${#VARIANTS[@]} variants, sizes=$SIZES runs=$RUNS ctx=$CTX"

for variant in "${VARIANTS[@]}"; do
  base="${variant%@*}"
  quant="${variant#*@}"
  slug="$(echo "$base" | sed 's|.*/||')-$quant"
  echo ""
  echo "=== $variant ==="

  drain
  if ! lms load "$variant" --context-length "$CTX" --yes >/dev/null 2>&1; then
    echo "  SKIP: failed to load $variant"
    continue
  fi
  echo "  loaded ($(resident) resident)"

  if $BENCH "$base" --no-unload --label "$variant" \
      -s "$SIZES" -n "$RUNS" --json > "results/$slug.json" 2>"results/$slug.log"; then
    echo "  done -> results/$slug.json"
    grep -E "failed|SKIPPED" "results/$slug.log" | sed 's/^/  /'
  else
    echo "  FAILED -- see results/$slug.log"
  fi
done

drain
echo ""
echo "matrix complete"
