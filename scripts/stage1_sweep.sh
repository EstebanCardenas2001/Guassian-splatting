#!/usr/bin/env bash
# Stage 1 sweep: N in {1k, 5k, 20k} x blending in {additive, alpha}. Extra args are passed
# to every run as config overrides (e.g. image.path=data/plant/crop.png).
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/miniforge3/etc/profile.d/conda.sh; set +u; conda activate gsplat; set -u
for mode in additive alpha; do
  for n in 1000 5000 20000; do
    python -m src.fit2d model.mode=$mode model.n=$n "$@"
  done
done
