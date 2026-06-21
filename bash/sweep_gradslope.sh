#!/bin/bash
# Sweep deyo_mlmp_gradslope_continual (grad_norm-slope restoration gate) params on
# ONE dataset, on ONE GPU. By default all configs run CONCURRENTLY (pack memory);
# set SEQ=1 to run them one at a time instead.
#
# grad_norm rises as mIoU degrades (docs/2026-06-18-contribution.md §7); this gate
# restores LN toward a lagged anchor when the smoothed grad_norm SLOPE exceeds
# slope_deadzone, with restore depth = lag_gain*slope (capped at max_lag), prob base_rst.
#
# Usage:
#   GPU=0 ROUNDS=150 bash bash/sweep_gradslope.sh v20 "0.001:0.01 0.002:0.01 0.004:0.01 0.002:0.02"
#   GPU=2 ROUNDS=1   bash bash/sweep_gradslope.sh acdc "0.002:0.01"   # smoke
#   GPU=0 SEQ=1      bash bash/sweep_gradslope.sh v20 "..."           # sequential
set -u
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-15}"   # seconds between concurrent launches (avoid model-load collision)

DS="${1:?dataset key: acdc|v20|cityscapes}"
CONFIGS="${2:?space-separated list of deadzone:base_rst, e.g. 0.002:0.01 0.004:0.02}"
GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-150}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

case "$DS" in
  acdc)
     DATASET=ACDCDataset;        DATA_DIR="data/ACDC/";        RESIZE="1120 560"
     CONDS="fog night rain snow";                    SUBSET="";                            SUBDIR="" ;;
  v20)
     DATASET=PascalVOC20Dataset; DATA_DIR="data/VOC/VOC2012/"; RESIZE="224 224"
     CONDS="snow frost fog brightness contrast";     SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="v20_acdc_matched/" ;;
  cityscapes)
     DATASET=CityscapesDataset;  DATA_DIR="data/Cityscape/";   RESIZE="1120 560"
     CONDS="snow frost fog brightness contrast";     SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="" ;;
  *) echo "bad dataset key: $DS (use acdc|v20|cityscapes)"; exit 1 ;;
esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
mkdir -p save/_sweep_logs

run_one() {
  local dz="$1" rst="$2"
  local SAVE="save/${DATASET}/${SUBDIR}deyo_mlmp_gradslope_dz${dz}_rst${rst}/"
  local LOG="save/_sweep_logs/gradslope_${DS}_dz${dz}_rst${rst}.log"
  echo "[$(date +%H:%M)] START $DS dz=$dz rst=$rst gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_gradslope_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --slope_window 10 --slope_deadzone "$dz" --lag_gain 100000 --max_lag 3000 \
     --base_rst "$rst" --monitor_interval 50 \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS dz=$dz rst=$rst (exit $?)"
}

for cfg in $CONFIGS; do
  dz="${cfg%%:*}"; rst="${cfg##*:}"
  if [ "$SEQ" = "1" ]; then
    run_one "$dz" "$rst"
  else
    run_one "$dz" "$rst" &
    sleep "$STAGGER"
  fi
done
wait
echo "[$(date +%H:%M)] $DS sweep COMPLETE."
