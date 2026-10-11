#!/usr/bin/env bash
# Stage 3 experiments: A = GT + noise (local convergence), B = random init N=200 (3 seeds),
# C = random init N=400 (over-parameterized).
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/miniforge3/etc/profile.d/conda.sh; set +u; conda activate gsplat; set -u
python -m src.fit_synthetic init.mode=gt_perturbed "$@"
for s in 0 1 2; do python -m src.fit_synthetic init.mode=random init.n=200 seed=$s "$@"; done
python -m src.fit_synthetic init.mode=random init.n=400 "$@"
