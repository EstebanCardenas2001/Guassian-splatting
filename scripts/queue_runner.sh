#!/usr/bin/env bash
# Run training jobs from a queue file, one at a time; the queue can be edited while it runs.
#   bash scripts/queue_runner.sh /mnt/logs/queue.txt /mnt/logs/queue_runs.log
# Each line: "<config.yaml> [key=value ...]". Lines are popped from the top; '#' lines skipped.
# Waits (polling every 60 s) when the queue is empty; stops when a line reads STOP.
set -uo pipefail
Q=$1; LOG=$2
cd "$(dirname "$0")/.."
source /mnt/miniforge3/etc/profile.d/conda.sh; set +u; conda activate gsplat; set -u
while true; do
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | head -1)
  if [ -z "$line" ]; then sleep 60; continue; fi
  # pop exactly that line
  python3 - "$Q" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
rows = open(q).read().splitlines()
rows.remove(line)
open(q, "w").write("\n".join(rows) + ("\n" if rows else ""))
PY
  [ "$line" = "STOP" ] && { echo "=== STOP" >> "$LOG"; break; }
  echo "=== START $(date -u +%FT%T) $line" >> "$LOG"
  python -m src.train --config $line 2>&1 | grep --line-buffered -viE "warn|imgs.append" >> "$LOG"
  echo "=== DONE $(date -u +%FT%T) exit=${PIPESTATUS[0]} $line" >> "$LOG"
done
