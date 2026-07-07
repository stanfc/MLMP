#!/bin/bash
# Run the COMBINED method deyo_mlmp_divreg_composite_continual on ONE dataset,
# ONE GPU. Diversity loss (lambda_div) + composite gate (conf_ceil:base_rst:gm).
#
#   L = L_DeYO - lambda_div*H_margin   + composite-gate restore
#
# lambda_div fixed via env LAMBDA_DIV (default 0.3 = divreg sweep sweet spot).
# CALIBRATION: diversity LOWERS mean_conf, so conf_ceil is swept below the plain
# DeYO-MLMP gate-internal peaks (ACDC 0.696, VOC20 0.584, Cityscapes 0.510).
# gate_log.csv logs mean_conf -> read its peak to refine.
#
# Usage (configs = "conf_ceil:base_rst:grad_mult_max"):
#   GPU=0 ROUNDS=150 LAMBDA_DIV=0.3 bash bash/sweep_divreg_composite.sh acdc "0.62:0.02:4 0.66:0.02:4"
set -u
SEQ="${SEQ:-1}"
STAGGER="${STAGGER:-20}"
LAMBDA_DIV="${LAMBDA_DIV:-0.3}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
CONFIGS="${2:?space-sep conf_ceil:base_rst:grad_mult_max}"
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
  local SAVE="save/${DATASET}/${SUBDIR}deyo_mlmp_divreg${LAMBDA_DIV}_composite_cc${cc}_rst${rst}_gm${gm}/"
  local LOG="save/_sweep_logs/divregcomp_${DS}_lam${LAMBDA_DIV}_cc${cc}_rst${rst}_gm${gm}.log"
  echo "[$(date +%H:%M)] START $DS lam=$LAMBDA_DIV cc=$cc rst=$rst gm=$gm gpu=$GPU -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_divreg_composite_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --lambda_div "$LAMBDA_DIV" --conf_ceil "$cc" --base_rst "$rst" --grad_mult_max "$gm" \
     --monitor_interval 50 \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS lam=$LAMBDA_DIV cc=$cc rst=$rst gm=$gm (exit $?)"
}

for cfg in $CONFIGS; do
  cc="${cfg%%:*}"; rest="${cfg#*:}"; rst="${rest%%:*}"; gm="${rest##*:}"
  if [ "$SEQ" = "1" ]; then run_one "$cc" "$rst" "$gm"; else run_one "$cc" "$rst" "$gm" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS divreg+composite sweep COMPLETE."
