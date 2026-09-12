#!/bin/sh
# Sequential coordinate-wise sweeps; each development trial keeps its own guard.
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
BASE=artifacts/v11_development/base_stageA.json
for axis in entropy_coefficient exploration_floor trace_decay discount critic_lr; do
  .venv/bin/python scripts/v11_develop.py --axis "$axis" --task catch \
    --episodes 600 --base "$BASE"
done
