#!/bin/bash
# Hyperparameter sweep for tent_divgate_smooth_anchor on ACDC (150R).
# Goal: can we recover smooth-anchor's deficit vs source-reset (31.59) by
#   (A) raising h_ceil      -> restore engages earlier / more often
#   (B) raising lag_scale   -> anchor reaches DEEPER (cleaner) snapshots
#   (C) combining A+B
#   (D) raising h_floor      -> fall back to SOURCE sooner
# Reference points: matched (ceil1.6/floor1.4/lag90)=29.53, source-reset=31.59.
#
# Runs all configs in the BACKGROUND on one GPU (default GPU 0). Each writes
# its own SAVE_DIR + log. Override GPU via:  SWEEP_GPU=1 bash <this>
#
# Compare afterwards:
#   python scripts/analyze_gate.py save/ACDCDataset/smooth_sweep_*  \
#                                  save/ACDCDataset/tent_divgate_continual_cau_rst_0.01

set -u
# GPUs to spread the sweep across (round-robin). Override: SWEEP_GPUS="0 1 2"
SWEEP_GPUS=(${SWEEP_GPUS:-0 1})
LOGDIR="logs/smooth_sweep"
mkdir -p "$LOGDIR"

# name | h_ceil | h_floor | lag_scale
CONFIGS=(
  "A_ceil1.8|1.8|1.4|90"
  "A_ceil2.0|2.0|1.4|90"
  "B_lag300|1.6|1.4|300"
  "B_lag1000|1.6|1.4|1000"
  "C_ceil2.0_lag1000|2.0|1.4|1000"
  "D_ceil1.8_floor1.55|1.8|1.55|90"
)

echo "Launching ${#CONFIGS[@]} smooth-anchor configs across GPUs: ${SWEEP_GPUS[*]} (background)..."
gi=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME CEIL FLOOR LAG <<< "$cfg"
  gpu="${SWEEP_GPUS[$((gi % ${#SWEEP_GPUS[@]}))]}"; gi=$((gi+1))
  save="save/ACDCDataset/smooth_sweep_${NAME}/"
  log="${LOGDIR}/${NAME}.log"
  echo "  -> ${NAME}: ceil=${CEIL} floor=${FLOOR} lag_scale=${LAG}  GPU=${gpu}  (log: ${log})"
  GPU_ID="$gpu" H_CEIL="$CEIL" H_FLOOR="$FLOOR" LAG_SCALE="$LAG" RST=0.01 \
    SAVE_DIR="$save" \
    bash bash/ACDC_10_round/smooth_anchor_default.sh > "$log" 2>&1 &
done

echo "PIDs: $(jobs -p | tr '\n' ' ')"
echo "Waiting for all ${#CONFIGS[@]} runs to finish..."
wait
echo ">>> SMOOTH-ANCHOR SWEEP COMPLETE <<<"
