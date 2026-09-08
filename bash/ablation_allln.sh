#!/bin/bash
# Ablation of ONE flagship design choice: which LayerNorm params may adapt.
#
# Byte-for-byte the flagship command (save/ACDCDataset/batch_ablation/
# gradnorm_scaled_b1/cmd.sh) with exactly two changes:
#   --method  deyo_mlmp_adagate_continual -> deyo_mlmp_adagate_allln_continual
#             (identical class; only `_is_excluded` differs, so tbe=0 means
#              nothing is frozen and all 100 params adapt instead of 98)
#   --top_block_exclude 6 -> 0
# LR stays at the flagship's 5e-6 (NOT the 1e-5 the plug-and-play arms use).
#
# Control = the existing flagship run: save/ACDCDataset/batch_ablation/
# gradnorm_scaled_b1/  (74 params, R150 = 31.92, mean 31.74)
#
# usage: bash bash/ablation_allln.sh <gpu> [rounds]
set -u
GPU="${1:?gpu id}"
ROUNDS="${2:-150}"
PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

SAVE="save/ACDCDataset/ablation/adagate_allln_b1/"
LOG="save/_ablation_logs/acdc_adagate_allln.log"
mkdir -p "$SAVE" save/_ablation_logs

echo "[$(date '+%m-%d %H:%M')] START adagate_allln (100 params) gpu=$GPU rounds=$ROUNDS -> $SAVE"
CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
  --save_dir "$SAVE" --data_dir data/ACDC/ --prompt_dir prompts.yaml \
  --dataset ACDCDataset --workers 1 --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list fog night rain snow \
  --phase1_rounds 0 --class_extensions --split val --subset_seed 0 \
  --corruption_severity 5 --ovss_type naclip --ovss_backbone ViT-L/14 \
  --adapt --method deyo_mlmp_adagate_allln_continual \
  --batch_size 1 --lr 5e-06 --steps 1 --continual_rounds "$ROUNDS" --seed 0 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 \
  --top_block_exclude 0 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
  --monitor_interval 50 --ln_ckpt_every 0 \
  --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled > "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE  adagate_allln (exit $?)"
