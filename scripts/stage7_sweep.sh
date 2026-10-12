#!/usr/bin/env bash
# Stage 7 ablation: A baseline -> B +mask loss -> C +densification -> D +SH -> E random init.
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/miniforge3/etc/profile.d/conda.sh; set +u; conda activate gsplat; set -u
for X in ${@:-A B C D E}; do
  echo "=== run $X"; python -m src.train --config configs/stage7_$X.yaml 2>&1 | grep --line-buffered -viE "warn|imgs.append"
done
