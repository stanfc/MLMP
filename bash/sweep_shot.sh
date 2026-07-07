#!/bin/bash
# Run shot_continual (SHOT Information-Maximization loss, continual/online) on ONE
# dataset, ONE GPU. Optional list of lambda_div values (SHOT default = 1.0).
#
# SHOT IM loss: L = L_ent - lambda_div * L_div  (entropy-min-all + diversity-max).
# This is the SHOT baseline for deyo_mlmp_divreg (same backbone/LN/MLMP/UAML/
# continual protocol; SHOT drops DeYO's reliability filter + PLPD + reweight and
# its offline centroid pseudo-labeling). See adapt/shot_continual.py docstring.
#
# Usage:
#   GPU=0 ROUNDS=150 bash bash/sweep_shot.sh acdc        "1.0"
#   GPU=1 ROUNDS=150 bash bash/sweep_shot.sh v20         "1.0"
#   GPU=3 ROUNDS=150 bash bash/sweep_shot.sh cityscapes  "1.0"
set -u
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-20}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
LAMBDAS="${2:-1.0}"
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
  local lam="$1"
  local tag="deyo_shot_lam${lam}"; [ "$lam" = "1.0" ] && tag="shot_continual"
  local SAVE="save/${DATASET}/${SUBDIR}${tag}/"
  local LOG="save/_sweep_logs/shot_${DS}_lam${lam}.log"
  echo "[$(date +%H:%M)] START SHOT $DS lambda_div=$lam gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method shot_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --lambda_div "$lam" \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  SHOT $DS lambda_div=$lam (exit $?)"
}

for lam in $LAMBDAS; do
  if [ "$SEQ" = "1" ]; then run_one "$lam"; else run_one "$lam" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS SHOT sweep COMPLETE."
