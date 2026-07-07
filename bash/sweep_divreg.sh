#!/bin/bash
# Sweep deyo_mlmp_divreg_continual on ONE dataset, ONE GPU, configs CONCURRENT.
#
# Diversity regularizer (loss-side, NOT a gate):
#   L = L_DeYO  -  lambda_div * H_margin
# where H_margin = entropy of the batch marginal class distribution (mean softmax
# over prompts/batch/space). Maximizing it (SHOT information-maximization) directly
# opposes the collapse mode (1-2 classes dominate -> low H_margin -> mIoU crash).
#
# Why H_margin: it is the strongest DIFFERENTIABLE signal on the collapse-regime
# datasets (ACDC rho=+0.94, Cityscapes +0.98). grad_norm correlates better across
# all 3 incl VOC20, but Phase O-1 proved grad_norm is a bad LOSS target. On VOC20
# (uniform-drift) H_margin is near-blind (rho~+0.14) -> expect gains on ACDC/
# Cityscapes, ~neutral VOC20 at small lambda_div. This is the honest tradeoff.
#
# lambda_div=0 -> bit-identical to deyo_mlmp_continual (its own control).
#
# Usage:
#   GPU=0 ROUNDS=150 bash bash/sweep_divreg.sh acdc       "0.0 0.1 0.3 1.0 3.0"
#   GPU=1 ROUNDS=150 bash bash/sweep_divreg.sh v20        "0.0 0.1 0.3 1.0 3.0"
#   GPU=2 ROUNDS=150 bash bash/sweep_divreg.sh cityscapes "0.0 0.1 0.3 1.0 3.0"
set -u
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-20}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
LAMBDAS="${2:?space-sep list of lambda_div values, e.g. \"0.0 0.1 0.3 1.0 3.0\"}"
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
  local SAVE="save/${DATASET}/${SUBDIR}deyo_mlmp_divreg_lam${lam}/"
  local LOG="save/_sweep_logs/divreg_${DS}_lam${lam}.log"
  echo "[$(date +%H:%M)] START $DS lambda_div=$lam gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_divreg_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --lambda_div "$lam" \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS lambda_div=$lam (exit $?)"
}

for lam in $LAMBDAS; do
  if [ "$SEQ" = "1" ]; then run_one "$lam"; else run_one "$lam" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS divreg sweep COMPLETE."
