#!/bin/bash
# Generates all 5 figures from the two experiments.
# usage: bash experiments/h1_validation/bash/plot_all.sh
set -e

EXP1_CSV=${EXP1_CSV:-experiments/h1_validation/results/exp1/all.csv}
EXP3_CSV=${EXP3_CSV:-experiments/h1_validation/results/exp3/all.csv}
EXP1_DIR=${EXP1_DIR:-experiments/h1_validation/results/exp1/figs/}
EXP3_DIR=${EXP3_DIR:-experiments/h1_validation/results/exp3/figs/}

export PYTHONPATH=${PYTHONPATH:-.}

if [ -f "$EXP1_CSV" ]; then
  echo "[plot] exp1 ← $EXP1_CSV → $EXP1_DIR"
  python -m experiments.h1_validation.exp1_gradient_cosine.plot --csv $EXP1_CSV --out_dir $EXP1_DIR
else
  echo "[plot] skip exp1 (no $EXP1_CSV)"
fi

if [ -f "$EXP3_CSV" ]; then
  echo "[plot] exp3 ← $EXP3_CSV → $EXP3_DIR"
  python -m experiments.h1_validation.exp3_severity_sweep.plot --csv $EXP3_CSV --out_dir $EXP3_DIR
else
  echo "[plot] skip exp3 (no $EXP3_CSV)"
fi
