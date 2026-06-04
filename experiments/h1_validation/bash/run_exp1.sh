#!/bin/bash
# Exp 1 driver — loops (dataset, condition) and appends to a single CSV.
# usage:
#   bash experiments/h1_validation/bash/run_exp1.sh
# optional env vars:
#   N=100              number of images per (dataset, condition) — or "all"
#   GPU=0              CUDA_VISIBLE_DEVICES
#   CSV=...            output CSV path
#   DATASETS="..."     subset of registry keys (default: all 5)
set -e

N=${N:-100}
GPU=3
CSV=${CSV:-experiments/h1_validation/results/exp1/all.csv}
DATASETS=${DATASETS:-"ACDC DarkZurich NighttimeDriving VOC20_matched Cityscapes"}

export PYTHONPATH=${PYTHONPATH:-.}
mkdir -p "$(dirname $CSV)"
echo "[exp1] CSV=$CSV  N=$N  GPU=$GPU"
echo "[exp1] DATASETS=$DATASETS"
echo ""

for ds in $DATASETS; do
  conds=$(python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['$ds']['conditions']))")
  for cond in $conds; do
    echo "===== [exp1] $ds / $cond (n=$N) ====="
    CUDA_VISIBLE_DEVICES=$GPU python -m experiments.h1_validation.exp1_gradient_cosine.run \
        --dataset $ds --condition $cond --n $N --out $CSV --skip_done
    echo ""
  done
done

echo "[exp1] All done. CSV at $CSV"
