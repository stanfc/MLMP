#!/bin/bash
# Exp 3 driver — loops (synthetic dataset, corruption, severity).
# usage:
#   bash experiments/h1_validation/bash/run_exp3.sh
# optional env vars:
#   GPU=0
#   CSV=...
#   DATASETS="VOC20_matched Cityscapes"   subset of synthetic registry keys
#   SEVERITIES="1 2 3 4 5"
set -e

GPU=${GPU:-0}
CSV=${CSV:-experiments/h1_validation/results/exp3/all.csv}
DATASETS=${DATASETS:-"VOC20_matched Cityscapes"}
SEVERITIES=${SEVERITIES:-"1 2 3 4 5"}

export PYTHONPATH=${PYTHONPATH:-.}
mkdir -p "$(dirname $CSV)"
echo "[exp3] CSV=$CSV  GPU=$GPU"
echo "[exp3] DATASETS=$DATASETS  SEVERITIES=$SEVERITIES"
echo ""

for ds in $DATASETS; do
  corrs=$(python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['$ds']['conditions']))")
  for corr in $corrs; do
    for sev in $SEVERITIES; do
      echo "===== [exp3] $ds / $corr / sev=$sev ====="
      CUDA_VISIBLE_DEVICES=$GPU python -m experiments.h1_validation.exp3_severity_sweep.run \
          --dataset $ds --corruption $corr --severity $sev --out $CSV --skip_done
      echo ""
    done
  done
done

echo "[exp3] All done. CSV at $CSV"
