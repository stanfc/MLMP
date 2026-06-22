#!/bin/bash
# Sweep deyo_mlmp_composite_gate_continual on ONE dataset, ONE GPU, configs CONCURRENT.
#
# Composite gate (docs/2026-06-18-contribution.md §7.3 design implication):
#   * mean_conf = mean max-softmax, monotone-in-time on ALL datasets (incl VOC20) ->
#     used as the TRIGGER (timing): restore turns on once windowed mean_conf rises
#     past conf_ceil (i.e. past the mIoU peak).
#   * grad_norm (anti-correlated with mIoU) -> used as restore DEPTH (intensity):
#     rst scales from base_rst up to base_rst*grad_mult_max as grad_norm rises.
# Motivation: H_margin gate (divgate) never fires on VOC20 (H stays ~3.0); mean_conf
# DOES cross its peak value on VOC20 (~R45-59), so this gate can hold the VOC20 peak
# that divgate cannot.
#
# Usage (conf_ceil is dataset-specific = its peak mean_conf):
#   GPU=0 ROUNDS=150 bash bash/sweep_composite.sh v20  "0.68:0.005:4 0.71:0.005:4 0.71:0.01:4"
#   GPU=1 ROUNDS=150 bash bash/sweep_composite.sh acdc "0.78:0.005:4 0.83:0.005:4 0.83:0.01:4"
set -u
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-20}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
CONFIGS="${2:?space-sep list of conf_ceil:base_rst:grad_mult_max}"
GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-150}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

case "$DS" in
  acdc)
     DATASET=ACDCDataset;        DATA_DIR="data/ACDC/";        RESIZE="1120 560"
     CONDS="fog night rain snow";                SUBSET="";                                  SUBDIR="" ;;
  v20)
     DATASET=PascalVOC20Dataset; DATA_DIR="data/VOC/VOC2012/"; RESIZE="224 224"
     CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="v20_acdc_matched/" ;;
  cityscapes)
     DATASET=CityscapesDataset;  DATA_DIR="data/Cityscape/";   RESIZE="1120 560"
     CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="" ;;
  *) echo "bad dataset key: $DS"; exit 1 ;;
esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
mkdir -p save/_sweep_logs

run_one() {
  local cc="$1" rst="$2" gm="$3"
  local SAVE="save/${DATASET}/${SUBDIR}deyo_mlmp_composite_cc${cc}_rst${rst}_gm${gm}/"
  local LOG="save/_sweep_logs/composite_${DS}_cc${cc}_rst${rst}_gm${gm}.log"
  echo "[$(date +%H:%M)] START $DS cc=$cc rst=$rst gm=$gm gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_composite_gate_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --conf_ceil "$cc" --base_rst "$rst" --grad_mult_max "$gm" --monitor_interval 50 \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS cc=$cc rst=$rst gm=$gm (exit $?)"
}

for cfg in $CONFIGS; do
  cc="${cfg%%:*}"; rest="${cfg#*:}"; rst="${rest%%:*}"; gm="${rest##*:}"
  if [ "$SEQ" = "1" ]; then run_one "$cc" "$rst" "$gm"; else run_one "$cc" "$rst" "$gm" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS composite sweep COMPLETE."
