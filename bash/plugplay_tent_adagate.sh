#!/bin/bash
# Plug-and-play evidence: the SAME AdaGate on a DIFFERENT base objective.
#
# The flagship (`deyo_mlmp_adagate_continual`) demonstrates the gate on one base
# objective. This runs it on plain TENT, with the identical one-flag control the
# batch ablation uses:
#     gate  : --base_rst 0.01
#     nogate: --base_rst 0.00   (gate computed and logged, never restores)
#
# Why an internal control rather than the published `tent_continual` number
# (ACDC R150 = 7.90): tent_continual trains ALL LayerNorm params, while the
# AdaGate code path excludes ln_post and the top `top_block_exclude=6` blocks.
# Comparing across that boundary would confound the gate with the trainable
# parameter set; --base_rst is the only difference between these two arms.
#
# LR 1e-5 follows tent_continual's own convention (save/ACDCDataset/
# tent_continual_Round150_lr_0.00001), not the flagship's 5e-6.
#
# usage: bash bash/plugplay_tent_adagate.sh <gate|nogate> <gpu> [rounds]
set -u
MK="${1:?arm: gate|nogate}"
GPU="${2:?gpu id}"
ROUNDS="${3:-150}"
PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

case "$MK" in
  gate)   RST=0.01 ;;
  nogate) RST=0.0  ;;
  *) echo "bad arm: $MK"; exit 1 ;;
esac

SAVE="save/ACDCDataset/plugplay/tent_adagate_${MK}_b1/"
LOG="save/_plugplay_logs/acdc_tent_adagate_${MK}.log"
mkdir -p "$SAVE" save/_plugplay_logs

echo "[$(date '+%m-%d %H:%M')] START tent_adagate/$MK base_rst=$RST gpu=$GPU rounds=$ROUNDS -> $SAVE"
CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
  --adapt --method tent_adagate_continual \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst $RST \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled \
  --lr 0.00001 --steps 1 \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset ACDCDataset --data_dir data/ACDC/ --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list fog night rain snow \
  --workers 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 --vision_outputs -1 \
  --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE  tent_adagate/$MK (exit $?)"
