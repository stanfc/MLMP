#!/bin/bash
# PROMPT-SET sweep on GDG-PA (deyo_mlmp_hmgate2_continual), ONE dataset, ONE GPU.
# The ONLY variable across runs is --prompt_dir (prompts_sweep/<ID>.yaml). The
# method and ALL gate hyperparameters are pinned to GDG-PA's best config, so any
# mIoU delta is attributable to the text prompts alone.
#
# Prompt sets live in prompts_sweep/ (S0_baseline .. S8_artistic); each yaml has a
# header comment stating its design rationale.
#
# Usage (one dataset, space-sep list of prompt IDs, runs CONCURRENT unless SEQ=1):
#   GPU=0 ROUNDS=150 bash bash/sweep_prompt.sh acdc "S0_baseline S2_photographic"
#   GPU=1 ROUNDS=150 bash bash/sweep_prompt.sh v20  "S0_baseline S3_street"
set -u
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-20}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
PROMPT_IDS="${2:?space-sep prompt-set IDs, e.g. \"S0_baseline S2_photographic\"}"
GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-150}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

case "$DS" in
  acdc)
     DATASET=ACDCDataset;        DATA_DIR="data/ACDC/";        RESIZE="1120 560"
     CONDS="fog night rain snow";                SUBSET="--subset_size 50 --subset_seed 0";  SUBDIR="" ;;
  v20)
     DATASET=PascalVOC20Dataset; DATA_DIR="data/VOC/VOC2012/"; RESIZE="224 224"
     CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="v20_acdc_matched/" ;;
  cityscapes)
     DATASET=CityscapesDataset;  DATA_DIR="data/Cityscape/";   RESIZE="1120 560"
     CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="" ;;
  *) echo "bad dataset key: $DS"; exit 1 ;;
esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
# GDG-PA best gate config (from bash/ACDC_10_round/deyo_mlmp_hmgate2_continual.sh)
SLOPE_WINDOW=10; SLOPE_DEADZONE=0.002; LAG_GAIN=1500; BASE_RST=0.01
H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50
mkdir -p save/_sweep_logs

run_one() {
  local pid="$1"
  local PROMPT_YAML="prompts_sweep/${pid}.yaml"
  local SAVE="save/${DATASET}/${SUBDIR}hmgate2_prompt_${pid}/"
  local LOG="save/_sweep_logs/prompt_${DS}_${pid}.log"
  if [ ! -f "$PROMPT_YAML" ]; then echo "MISSING $PROMPT_YAML"; return 1; fi
  echo "[$(date +%H:%M)] START $DS prompt=$pid gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_hmgate2_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir "$PROMPT_YAML" \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --slope_window $SLOPE_WINDOW --slope_deadzone $SLOPE_DEADZONE --lag_gain $LAG_GAIN \
     --base_rst $BASE_RST --h_drop_ratio $H_DROP_RATIO --maxlag_shallow $MAXLAG_SHALLOW \
     --monitor_interval $MONITOR_INTERVAL \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS prompt=$pid (exit $?)"
}

for pid in $PROMPT_IDS; do
  if [ "$SEQ" = "1" ]; then run_one "$pid"; else run_one "$pid" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS prompt sweep COMPLETE."
