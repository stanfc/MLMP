#!/bin/bash
# Phase-switch stress test: does the gate cope with a MID-STREAM difficulty jump,
# not just a fixed corruption set for the whole 150 rounds? Rounds 1-75 use
# easy5corr (snow/frost/fog/brightness/contrast -- the ACDC-weather-mapped set
# already used for the normal-protocol V20/Cityscapes figures), rounds 76-150
# switch to hard5corr (defocus_blur/glass_blur/gaussian_noise/zoom_blur/
# elastic_transform -- same hard5corr set used on the clsTTA side).
# ACDC excluded (real weather, no corruption concept -- can't do this switch).
# Uses the 4 "trusted" arms from the dropped-hmargin figures: ctrl (GDG-PA),
# ABmad05_ecdf (flagship), ABmad05_ecdf_grow (gradnorm_uncapped),
# ABmad05_ecdf_grow_scaled (gradnorm_scaled).
#
# Usage:
#   GPU=3 ROUNDS=150 bash bash/sweep_adagate_easy2hard.sh v20 "ctrl ABmad05_ecdf ABmad05_ecdf_grow ABmad05_ecdf_grow_scaled"
#   GPU=4 ROUNDS=150 bash bash/sweep_adagate_easy2hard.sh cityscapes "ctrl ABmad05_ecdf ABmad05_ecdf_grow ABmad05_ecdf_grow_scaled"
set -u
source ~/miniconda3/etc/profile.d/conda.sh
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-20}"
TAG="${TAG:-}"

DS="${1:?dataset key: v20|cityscapes}"
ARMS="${2:?space-sep arm IDs, e.g. \"ctrl ABmad05_ecdf\"}"
GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-150}"
PHASE1_ROUNDS="${PHASE1_ROUNDS:-75}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

EASY5CORR="snow frost fog brightness contrast"
HARD5CORR="defocus_blur glass_blur gaussian_noise zoom_blur elastic_transform"

case "$DS" in
  v20)
     DATASET=PascalVOC20Dataset; DATA_DIR=".data/VOC2012/"; RESIZE="224 224"
     SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="v20_easy2hard/" ;;
  cityscapes)
     DATASET=CityscapesDataset;  DATA_DIR=".data/cityscapes/";   RESIZE="1120 560"
     SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="cityscapes_easy2hard/" ;;
  *) echo "bad dataset key: $DS (only v20|cityscapes -- ACDC has no corruption concept)"; exit 1 ;;
esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
SLOPE_WINDOW=10; SLOPE_DEADZONE=0.002; LAG_GAIN=1500; BASE_RST=0.01
H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50; TREND_HIST=50
mkdir -p save/_sweep_logs

arm_cfg() {
  case "$1" in
    ctrl)                        echo "abs   0.002 gain 1.5 fixed" ;;
    ABmad05_ecdf)                echo "mad   0.5   ecdf 1.5 fixed" ;;
    ABmad05_ecdf_grow)           echo "mad   0.5   ecdf 1.5 growing" ;;
    ABmad05_ecdf_grow_scaled)    echo "mad   0.5   ecdf 1.5 growing_scaled" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local arm="$1"
  local cfg; cfg="$(arm_cfg "$arm")" || { echo "unknown arm: $arm"; return 1; }
  read -r TSTAT TTHR LMODE LSAT SCAP <<< "$cfg"
  local SAVE="save/${DATASET}/${SUBDIR}adagate_${arm}${TAG}/"
  local LOG="save/_sweep_logs/adagate_easy2hard_${DS}_${arm}${TAG}.log"
  echo "[$(date +%H:%M)] START $DS arm=$arm easy2hard@$PHASE1_ROUNDS gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n mlmp python main_continual.py \
     --adapt --method deyo_mlmp_adagate_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 \
     --corruptions_list $EASY5CORR --corruptions_list2 $HARD5CORR --phase1_rounds "$PHASE1_ROUNDS" \
     $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --slope_window $SLOPE_WINDOW --slope_deadzone $SLOPE_DEADZONE --lag_gain $LAG_GAIN \
     --base_rst $BASE_RST --h_drop_ratio $H_DROP_RATIO --maxlag_shallow $MAXLAG_SHALLOW \
     --shallow_cap_mode "$SCAP" \
     --monitor_interval $MONITOR_INTERVAL \
     --trend_stat "$TSTAT" --trend_thr "$TTHR" --trend_hist $TREND_HIST \
     --lag_mode "$LMODE" --lag_sat "$LSAT" \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS arm=$arm (exit $?)"
}

for arm in $ARMS; do
  if [ "$SEQ" = "1" ]; then run_one "$arm"; else run_one "$arm" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS adagate easy2hard sweep COMPLETE."
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "$DS easy5corr->hard5corr sweep (4 arms) 完成" "EXP-EASY2HARD" 2>/dev/null || true
