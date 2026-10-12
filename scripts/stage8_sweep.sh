#!/usr/bin/env bash
# Stage 8 runs: each argument is "config [overrides...]" (quoted).
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/miniforge3/etc/profile.d/conda.sh; set +u; conda activate gsplat; set -u
for spec in "$@"; do
  echo "=== $spec"; python -m src.train --config $spec 2>&1 | grep --line-buffered -viE "warn|imgs.append"
done
