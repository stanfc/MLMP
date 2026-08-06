#!/bin/bash
# Learnable TEXT-EMBEDDING RESIDUAL + ORTHOGONALITY sweep on GDG-PA
# (deyo_mlmp_textres_hmgate2_continual).
#
# Arms (the LAMBDAS list is interpreted as follows):
#   "ctrl"  -> --text_res_lr 0            : no residual is created at all.
#                                           BIT-IDENTICAL to GDG-PA. The control.
#   "0.0"   -> residual ON, --lambda_orth 0 : residual WITHOUT the regularizer --
#                                           tests whether text-side degradation
#                                           happens on its own (the "does the
#                                           regularizer even have a job" arm).
#   <x>     -> residual ON, --lambda_orth x : residual + orthogonality penalty.
#
# Everything else (gate, DeYO, MLMP, LR, seed, subset) is pinned, so any mIoU delta
# is attributable to the text side alone.
#
# Usage (runs the lambda list CONCURRENT unless SEQ=1):
#   GPU=3 ROUNDS=150 bash bash/sweep_textres.sh acdc "ctrl 0.0 0.1 1.0 10.0"
set -u
SEQ="${SEQ:-0}"; STAGGER="${STAGGER:-20}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
LAMBDAS="${2:?space-sep arm list, e.g. \"ctrl 0.0 0.1 1.0 10.0\"}"
GPU="${GPU:-0}"; ROUNDS="${ROUNDS:-150}"
SEED="${SEED:-0}"
if [ "$SEED" = "0" ]; then SEED_SUF=""; else SEED_SUF="_s${SEED}"; fi
# residual LR and the per-class L2 cap are PINNED across the sweep (the cap, not the
# LR, is the real magnitude lever: at lr 5e-6 the cap binds after ~6k steps).
TEXT_RES_LR="${TEXT_RES_LR:-0.000005}"
TEXT_RES_MAX_NORM="${TEXT_RES_MAX_NORM:-0.5}"
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
SLOPE_WINDOW=10; SLOPE_DEADZONE=0.002; LAG_GAIN=1500; BASE_RST=0.01
H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50
mkdir -p save/_sweep_logs

run_one() {
  local arm="$1"
  local RLR LORTH
  if [ "$arm" = "ctrl" ]; then RLR="0.0"; LORTH="0.0";
  else                         RLR="$TEXT_RES_LR"; LORTH="$arm"; fi
  local SAVE="save/${DATASET}/${SUBDIR}textres_${arm}${SEED_SUF}/"
  local LOG="save/_sweep_logs/textres_${DS}_${arm}${SEED_SUF}.log"
  echo "[$(date +%H:%M)] START $DS arm=$arm seed=$SEED (res_lr=$RLR orth=$LORTH) gpu=$GPU -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_textres_hmgate2_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed "$SEED" \
     --vision_outputs $OUT_VISION \
     --slope_window $SLOPE_WINDOW --slope_deadzone $SLOPE_DEADZONE --lag_gain $LAG_GAIN \
     --base_rst $BASE_RST --h_drop_ratio $H_DROP_RATIO --maxlag_shallow $MAXLAG_SHALLOW \
     --monitor_interval $MONITOR_INTERVAL \
     --text_res_lr "$RLR" --lambda_orth "$LORTH" --text_res_max_norm "$TEXT_RES_MAX_NORM" \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS arm=$arm (exit $?)"
}

for a in $LAMBDAS; do
  if [ "$SEQ" = "1" ]; then run_one "$a"; else run_one "$a" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS textres sweep COMPLETE."
