#!/bin/bash
# Batch-size ablation: does our method hold up at batch = 1 / 8 / 64?
#
# Produces, per dataset, one figure per batch size with FOUR lines:
#   no_adapt        - tent_continual with --adapt omitted (model never updates)
#   mlmp_episodic   - main.py MLMP, reset per sample (episodic upper bound)
#   gradnorm_scaled - deyo_mlmp_adagate_continual --shallow_cap_mode growing_scaled
#                     (學長 2026.08.14 flagship; config from adagate_ABmad05_ecdf_grow_scaled)
#   deyo_mlmp       - IDENTICAL to gradnorm_scaled but --base_rst 0.0, i.e. the gate
#                     never restores (學長's own no-gate convention) => DeYO+MLMP alone.
#                     One-flag ablation: gate+restoration removed, nothing else changes.
#
# Protocol (per user, 2026-08-21): ACDC = FULL 406 img/round; v20 and cityscapes =
# subset 100/corruption (500 img/round) because full would take far too long.
# LR is fixed at 5e-6 for every batch size -- this codebase's OVSS convention;
# no batch-scaled-LR rule exists here (學長 states this explicitly in their ablation).
#
# usage: bash bash/batch_ablation.sh <acdc|v20|cityscapes> <method> <batch> <gpu> [rounds]
set -u
DS="${1:?dataset: acdc|v20|cityscapes}"
MK="${2:?method: no_adapt|mlmp_episodic|gradnorm_scaled|deyo_mlmp}"
BS="${3:?batch size}"
GPU="${4:-0}"
ROUNDS="${5:-150}"
PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

case "$DS" in
  acdc)       DATASET=ACDCDataset;        DATA_DIR="data/ACDC/";        RESIZE="1120 560"
              CONDS="fog night rain snow";                SUBSET="" ;;
  v20)        DATASET=PascalVOC20Dataset; DATA_DIR="data/VOC/VOC2012/"; RESIZE="224 224"
              CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0" ;;
  cityscapes) DATASET=CityscapesDataset;  DATA_DIR="data/Cityscape/";   RESIZE="1120 560"
              CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0" ;;
  *) echo "bad dataset key: $DS"; exit 1 ;;
esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
SAVE="save/${DATASET}/batch_ablation/${MK}_b${BS}/"
LOG="save/_batch_ablation_logs/${DS}_${MK}_b${BS}.log"
mkdir -p "$SAVE" save/_batch_ablation_logs

# shared flags for the two adagate-based arms (identical except --base_rst)
ADAGATE_COMMON="--adapt --method deyo_mlmp_adagate_continual
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled
  --lr 0.000005 --steps 1"

echo "[$(date '+%m-%d %H:%M')] START $DS/$MK/b$BS gpu=$GPU rounds=$ROUNDS -> $SAVE"

if [ "$MK" = "mlmp_episodic" ]; then
  # episodic: main.py, resets per sample -> a flat reference line, no rounds needed
  CUDA_VISIBLE_DEVICES=$GPU $PY main.py \
    --adapt --method mlmp --ovss_type naclip --ovss_backbone ViT-L/14 \
    --prompt_dir prompts.yaml --alpha_cls 1.0 \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
    --workers 1 --lr 0.001 --steps 1 --trials 1 --batch_size "$BS" --seed 0 \
    --vision_outputs $OUT_VISION \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
elif [ "$MK" = "no_adapt" ]; then
  # --adapt omitted => model never updates => identical every round (flat line)
  CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
    --method tent_continual --ovss_type naclip --ovss_backbone ViT-L/14 \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
    --workers 1 --batch_size "$BS" --continual_rounds "$ROUNDS" --seed 0 \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
else
  case "$MK" in
    gradnorm_scaled) RST=0.01 ;;
    deyo_mlmp)       RST=0.0  ;;   # gate never restores => DeYO+MLMP only
    *) echo "bad method key: $MK"; exit 1 ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
    $ADAGATE_COMMON --base_rst $RST \
    --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
    --workers 1 --batch_size "$BS" --continual_rounds "$ROUNDS" --seed 0 \
    --vision_outputs $OUT_VISION \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
fi
echo "[$(date '+%m-%d %H:%M')] DONE  $DS/$MK/b$BS (exit $?)"
