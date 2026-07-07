#!/bin/bash
# Round-2 sweep for sar_mlmp_smooth_anchor.
#
# Group A (V20 weather, 8) — "flatten the parabola tail".
#   Round-1 finding: V20 H stays 2.7-3.1, well above floor 2.2-2.6, so the
#   H<=floor "restore-to-clean-source" hard clamp NEVER fires; the only active
#   restore trails a recent (already-drifted) snapshot -> tail keeps sliding
#   down (parabola). c3.4/f2.6/r0.010 already nearly flat (meanLag 369).
#   => push floor UP into the H band (2.6-2.8) + stronger rst (0.010-0.020) +
#      sometimes larger lag_scale, to deepen the backward pull and start
#      touching the source clamp. Goal: last ~= peak (convergent plateau).
#
# Group B (ACDC, 8) — "different lever, gate is not the bottleneck".
#   Round-1 finding: all 8 gate variants capped ~30.05 (== baseline) < episodic
#   30.6. Gate is PINNED (H~2.28 always < ceil, deep-restoring 88-100%) AND the
#   SAR reliable filter is INACTIVE (mean entropy 1.25 << e_margin 1.8 -> 0%
#   filtered). So the ceiling is in the base SAR+UAML loss, not the gate.
#   Hold the best gate (c2.9/f2.2/l150/r0.002) fixed; vary base-loss levers:
#   lr, uaml_in_adapt (18-layer vs single-layer adapt), alpha_cls, e_margin-down.
#
# 16 jobs on the two IDLE GPUs (1 & 3); GPUs 0/2 left alone (other-session jobs).
# Usage:  bash bash/sweep_sar_mlmp_v20tail_acdc_levers.sh

cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MLMP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

ACDC=bash/ACDC_10_round/sar_mlmp_smooth_anchor_continual.sh
V20=bash/v20_acdc_matched/sar_mlmp_smooth_anchor_continual.sh
LOG=save/_sweep_logs; mkdir -p "$LOG"

# Each job: "TAG GPU SCRIPT ENV..."  (ENV = space-separated VAR=VAL overrides)
JOBS=(
  # ---- Group A: V20 tail-flatten (ceil above H band; raise floor + rst) ----
  "v20tail_c3.4_f2.6_l150_r0.015 1 $V20 H_CEIL=3.4 H_FLOOR=2.6 LAG_SCALE=150 RST=0.015"
  "v20tail_c3.4_f2.7_l150_r0.010 1 $V20 H_CEIL=3.4 H_FLOOR=2.7 LAG_SCALE=150 RST=0.010"
  "v20tail_c3.4_f2.7_l150_r0.015 1 $V20 H_CEIL=3.4 H_FLOOR=2.7 LAG_SCALE=150 RST=0.015"
  "v20tail_c3.4_f2.8_l150_r0.010 1 $V20 H_CEIL=3.4 H_FLOOR=2.8 LAG_SCALE=150 RST=0.010"
  "v20tail_c3.4_f2.8_l150_r0.020 3 $V20 H_CEIL=3.4 H_FLOOR=2.8 LAG_SCALE=150 RST=0.020"
  "v20tail_c3.2_f2.6_l150_r0.015 3 $V20 H_CEIL=3.2 H_FLOOR=2.6 LAG_SCALE=150 RST=0.015"
  "v20tail_c3.4_f2.7_l300_r0.010 3 $V20 H_CEIL=3.4 H_FLOOR=2.7 LAG_SCALE=300 RST=0.010"
  "v20tail_c3.4_f2.8_l300_r0.015 3 $V20 H_CEIL=3.4 H_FLOOR=2.8 LAG_SCALE=300 RST=0.015"

  # ---- Group B: ACDC base-loss levers (gate fixed c2.9/f2.2/l150/r0.002) ----
  "acdc_lvr_lr1e5            1 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 LR=0.00001"
  "acdc_lvr_uaml0           1 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 UAML_IN_ADAPT=0"
  "acdc_lvr_acls0.5         1 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 ALPHA_CLS=0.5"
  "acdc_lvr_acls1.0         1 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 ALPHA_CLS=1.0"
  "acdc_lvr_emargin1.2      3 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 E_MARGIN=1.2"
  "acdc_lvr_lr1e5_uaml0     3 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 LR=0.00001 UAML_IN_ADAPT=0"
  "acdc_lvr_lr1e5_acls0.5   3 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 LR=0.00001 ALPHA_CLS=0.5"
  "acdc_lvr_emargin1.2_acls0.5 3 $ACDC H_CEIL=2.9 H_FLOOR=2.2 LAG_SCALE=150 RST=0.002 E_MARGIN=1.2 ALPHA_CLS=0.5"
)

echo "[$(date '+%F %T')] round-2 sweep start: ${#JOBS[@]} jobs (GPU 1 & 3 only)" | tee -a "$LOG/_sweep2.log"
for j in "${JOBS[@]}"; do
  read -r tag gpu script env <<< "$j"
  if [[ "$tag" == v20* ]]; then
    sd="save/PascalVOC20Dataset/v20_acdc_matched/smlp_${tag}/"
  else
    sd="save/ACDCDataset/smlp_${tag}/"
  fi
  echo "[$(date '+%F %T')] GPU$gpu $tag :: $env" | tee -a "$LOG/_sweep2.log"
  env GPU_ID=$gpu SAVE_DIR="$sd" $env \
    nohup bash "$script" > "$LOG/r2_${tag}.log" 2>&1 &
  sleep 10
done
echo "[$(date '+%F %T')] all ${#JOBS[@]} round-2 jobs launched" | tee -a "$LOG/_sweep2.log"
