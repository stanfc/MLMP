#!/bin/bash
# Entropy-weighted PROMPT-aggregation sweep on GDG-PA
# (deyo_mlmp_promptw_hmgate2_continual). Two INDEPENDENT sides:
#   side=adapt : vary --prompt_weight_beta, eval stays GDG-PA original (avg_embed).
#                beta 0.0 == bit-identical GDG-PA control.
#   side=eval  : --prompt_weight_beta 0 (adapt unchanged), eval_prompt_mode=ent_weight,
#                vary --eval_prompt_beta. beta 0.0 == uniform logit ensemble.
# Gate + DeYO hyperparameters pinned to GDG-PA best; only the prompt-weighting varies.
#
# Usage (runs the beta list CONCURRENT unless SEQ=1):
#   GPU=0 ROUNDS=150 bash bash/sweep_promptw.sh acdc adapt "0.0 0.5 1.0 2.0 4.0"
#   GPU=1 ROUNDS=150 bash bash/sweep_promptw.sh v20  eval  "0.0 1.0 2.0"
set -u
SEQ="${SEQ:-0}"; STAGGER="${STAGGER:-20}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
SIDE="${2:?side: adapt|eval}"
BETAS="${3:?space-sep beta list, e.g. \"0.0 0.5 1.0 2.0 4.0\"}"
GPU="${GPU:-0}"; ROUNDS="${ROUNDS:-150}"
# SEED varies algorithmic stochasticity (restore masks, DeYO augmentation). The image
# subset is pinned by --subset_seed 0 regardless, so seeds give a PAIRED comparison.
# seed 0 keeps the original dir names; other seeds get an _s<SEED> suffix.
SEED="${SEED:-0}"
if [ "$SEED" = "0" ]; then SEED_SUF=""; else SEED_SUF="_s${SEED}"; fi
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
case "$SIDE" in adapt|eval) ;; *) echo "bad side: $SIDE"; exit 1 ;; esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
SLOPE_WINDOW=10; SLOPE_DEADZONE=0.002; LAG_GAIN=1500; BASE_RST=0.01
H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50
mkdir -p save/_sweep_logs

run_one() {
  local beta="$1"
  local PWB EMODE EPB
  if [ "$SIDE" = "adapt" ]; then PWB="$beta"; EMODE="avg_embed";  EPB="1.0";
  else                           PWB="0.0";   EMODE="ent_weight"; EPB="$beta"; fi
  local SAVE="save/${DATASET}/${SUBDIR}promptw_${SIDE}_b${beta}${SEED_SUF}/"
  local LOG="save/_sweep_logs/promptw_${DS}_${SIDE}_b${beta}${SEED_SUF}.log"
  echo "[$(date +%H:%M)] START $DS $SIDE beta=$beta seed=$SEED (PWB=$PWB $EMODE EPB=$EPB) gpu=$GPU -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_promptw_hmgate2_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed "$SEED" \
     --vision_outputs $OUT_VISION \
     --slope_window $SLOPE_WINDOW --slope_deadzone $SLOPE_DEADZONE --lag_gain $LAG_GAIN \
     --base_rst $BASE_RST --h_drop_ratio $H_DROP_RATIO --maxlag_shallow $MAXLAG_SHALLOW \
     --monitor_interval $MONITOR_INTERVAL \
     --prompt_weight_beta $PWB --eval_prompt_mode $EMODE --eval_prompt_beta $EPB \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS $SIDE beta=$beta (exit $?)"
}

for b in $BETAS; do
  if [ "$SEQ" = "1" ]; then run_one "$b"; else run_one "$b" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS $SIDE promptw sweep COMPLETE."
