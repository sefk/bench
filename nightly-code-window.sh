#!/bin/bash
# Run the LiveCodeBench queue inside a nightly window, then stop.
#
# A 16k-budget pass over the dense builds is tens of hours of decoding, which
# is fine as long as it only ever happens while the machine is otherwise idle.
# launchd starts this at midnight (com.sefk.bench-code-16k); it runs until
# WINDOW_END and stops, and the next night picks up where it left off. Being
# stopped mid-item costs at most that one answer, since the output file
# resumes.
#
#   ./nightly-code-window.sh                       # the default dense queue
#   WINDOW_END=06:00 ./nightly-code-window.sh qwen/qwen3.6-27b@4bit
#   WINDOW_END=+2 ./nightly-code-window.sh ...     # +N minutes, for testing
#
# Progress and per-night start/stop lines go to $OUTDIR/quality-code-16k.log.

set -u
cd "$(dirname "$0")"
export PATH="/Users/sefk/.lmstudio/bin:/opt/homebrew/bin:$PATH"

BUDGET=${BUDGET:-16384}
OUTDIR=${OUTDIR:-results/2026-09-17}
WINDOW_END=${WINDOW_END:-06:00}
LOG="$OUTDIR/quality-code-16k.log"

# The dense 4-bit pair: both already have a 4k pass, so only their truncated
# answers are regenerated. The 8-bit dense builds have no 4k pass at all and
# would each be a full 175-problem run -- name them explicitly to take that on.
if [ $# -gt 0 ]; then
  VARIANTS=("$@")
else
  VARIANTS=("qwen/qwen3.6-27b@4bit" "qwen/qwen3.8-27b@4bit")
fi

mkdir -p "$OUTDIR"

if pgrep -qf "run-code-quality.sh" || pgrep -qf "quality-bench --task code"; then
  echo "$(date '+%F %T') window skipped: a run is already going" >> "$LOG"
  exit 0
fi

case "$WINDOW_END" in
  +*) deadline=$(( $(date +%s) + ${WINDOW_END#+} * 60 )) ;;
  *)
    deadline=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $WINDOW_END" +%s)
    # Started after the window's end time (a late launch, or a hand run in the
    # evening): aim at tomorrow's, so the window is never zero-length.
    [ "$deadline" -le "$(date +%s)" ] && deadline=$((deadline + 86400))
    ;;
esac

echo "" >> "$LOG"
echo "=== window $(date '+%F %T') -> $(date -r "$deadline" '+%F %T') ===" >> "$LOG"

BUDGET=$BUDGET OUTDIR=$OUTDIR ./run-code-quality.sh "${VARIANTS[@]}" >> "$LOG" 2>&1 &
runner=$!

while kill -0 "$runner" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date '+%F %T') window closed -- stopping; resumes next window" >> "$LOG"
    # The runner first, so it does not start the next variant, then the
    # in-flight item. Its answer is lost and regenerated next time.
    kill "$runner" 2>/dev/null
    pkill -f "quality-bench --task code"
    wait "$runner" 2>/dev/null
    lms unload --all >/dev/null 2>&1
    exit 0
  fi
  sleep 30
done

echo "$(date '+%F %T') queue finished inside the window" >> "$LOG"
