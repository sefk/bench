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
#
# Power: this script never runs sudo. To measure energy, start the sampler
# yourself first and point POWER_LOG at it:
#
#   sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -o /tmp/power.log &
#   POWER_LOG=/tmp/power.log ./run-matrix.sh
#   sudo pkill -INT powermetrics        # when done
#
# Leaving the sampler running before and after the benchmark is what makes the
# idle baseline available, so the reported energy can be given as an increment
# over doing nothing.

set -u
cd "$(dirname "$0")"

SIZES=${SIZES:-0,1000,4000,16000}
RUNS=${RUNS:-3}
POWER_LOG=${POWER_LOG:-}
OUTDIR=${OUTDIR:-results/$(date +%F)}
BENCH=./lmstudio-bench

BENCH_ARGS=()
if [ -n "$POWER_LOG" ]; then
  if [ ! -e "$POWER_LOG" ]; then
    echo "power log $POWER_LOG does not exist -- start the sampler first:" >&2
    echo "  sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -o $POWER_LOG &" >&2
    exit 1
  fi
  BENCH_ARGS+=(--power-log "$POWER_LOG")
else
  echo "NOT measuring power (set POWER_LOG=/path/to/log to measure)" >&2
fi

# Fastest first, so results land early if the run is interrupted.
VARIANTS=(
  "qwen/qwen3.6-27b@4bit"
  "qwen/qwen3.6-35b-a3b@4bit"
  "qwen/qwen3.6-35b-a3b@8bit"
  "qwen/qwen3.6-27b@8bit"
)

mkdir -p "$OUTDIR"
echo "matrix: ${#VARIANTS[@]} variants, sizes=$SIZES runs=$RUNS -> $OUTDIR"

for variant in "${VARIANTS[@]}"; do
  slug="$(echo "$variant" | sed 's|.*/||; s|@|-|')"
  echo ""
  echo "=== $variant -> $OUTDIR/$slug.json ==="
  start=$(date +%s)

  if $BENCH "$variant" "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}" \
      -s "$SIZES" -n "$RUNS" --json \
      > "$OUTDIR/$slug.json" 2> "$OUTDIR/$slug.log"; then
    rows=$(python3 -c "import json;print(len(json.load(open('$OUTDIR/$slug.json'))))" 2>/dev/null || echo 0)
    echo "  done in $(( ($(date +%s) - start) / 60 ))m, $rows rows"
    grep -E "failed|SKIPPED|WARNING" "$OUTDIR/$slug.log" | sed 's/^/  /'
  else
    echo "  FAILED -- see $OUTDIR/$slug.log"
    tail -3 "$OUTDIR/$slug.log" | sed 's/^/  /'
  fi
done

lms unload --all >/dev/null 2>&1
echo ""
echo "matrix complete -> $OUTDIR"
