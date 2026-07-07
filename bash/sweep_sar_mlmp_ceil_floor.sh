#!/bin/bash
# Parameter sweep for sar_mlmp_smooth_anchor — goal: beat MLMP-episodic on
# ACDC (>30.6) and V20-weather-subset (>76.2). Cityscapes already clears it.
#
# Analysis basis (gate_log H_margin + lag state of the baseline ceil2.9/floor2.2):
#   ACDC: H 2.23-2.50, gate active 89% restoring ~1730 back -> PINNED, can't climb.
#         => loosen (lower ceil into its band so healthy windows go OFF; or
#            shallower anchor via lower floor; or smaller rst).
#   V20:  H 2.14-3.11, gate OFF 34% (H>=ceil) -> drifts off the R36 peak.
#         => raise ceil so the gate stays engaged; raise floor for source pulls.
#
# Each job overrides the existing sar_mlmp runner via env vars. Packed across
# GPUs 0-3 (2 ACDC + 2 V20 per GPU), cv2 thread-capped, staggered.
#
# Usage:  bash bash/sweep_sar_mlmp_ceil_floor.sh

cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MLMP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

ACDC=bash/ACDC_10_round/sar_mlmp_smooth_anchor_continual.sh
V20=bash/v20_acdc_matched/sar_mlmp_smooth_anchor_continual.sh
LOG=save/_sweep_logs; mkdir -p "$LOG"

# fields: dataset script  gpu ceil floor lag rst
JOBS=(
  # --- ACDC: loosen to climb (+ user's floor-up hypothesis a7/a8) ---
  "ACDC $ACDC 0 2.4 2.2 150 0.005"
  "ACDC $ACDC 1 2.5 2.2 150 0.005"
  "ACDC $ACDC 2 2.9 1.9 150 0.005"
  "ACDC $ACDC 3 2.9 2.0 100 0.005"
  "ACDC $ACDC 0 2.9 2.2 150 0.002"
  "ACDC $ACDC 1 2.4 1.9 100 0.003"
  "ACDC $ACDC 2 2.9 2.4 150 0.005"
  "ACDC $ACDC 3 2.6 2.4 150 0.008"
  # --- V20 weather subset: engage gate to hold peak (raise ceil / floor) ---
  "V20  $V20  0 3.2 2.2 150 0.005"
  "V20  $V20  1 3.4 2.2 150 0.005"
  "V20  $V20  2 3.2 2.4 150 0.005"
  "V20  $V20  3 3.2 2.2 150 0.010"
  "V20  $V20  0 3.4 2.6 150 0.010"
  "V20  $V20  1 3.2 2.5 100 0.005"
  "V20  $V20  2 2.9 2.5 150 0.005"
  "V20  $V20  3 3.4 2.2 300 0.005"
)

echo "[$(date '+%F %T')] sweep start: ${#JOBS[@]} jobs" | tee -a "$LOG/_sweep.log"
for j in "${JOBS[@]}"; do
  read ds script gpu ceil floor lag rst <<< "$j"
  if [ "$ds" = "ACDC" ]; then
    base=save/ACDCDataset
  else
    base=save/PascalVOC20Dataset/v20_acdc_matched
  fi
  tag="smlp_c${ceil}_f${floor}_l${lag}_r${rst}"
  sd="$base/$tag/"
  echo "[$(date '+%F %T')] $ds GPU$gpu $tag" | tee -a "$LOG/_sweep.log"
  GPU_ID=$gpu H_CEIL=$ceil H_FLOOR=$floor LAG_SCALE=$lag RST=$rst SAVE_DIR="$sd" \
    nohup bash "$script" > "$LOG/${ds}_${tag}.log" 2>&1 &
  sleep 10
done
echo "[$(date '+%F %T')] all ${#JOBS[@]} sweep jobs launched" | tee -a "$LOG/_sweep.log"
