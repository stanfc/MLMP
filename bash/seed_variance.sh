#!/bin/bash
# Seed variance on the FLAGSHIP configuration.
#
# Every result in this project is seed=0 -- it is the standing caveat in every
# memo and the first thing a reviewer asks about a 150-round stochastic method
# (the gate draws a stochastic restore mask every window, and DeYO shuffles
# patches). This runs the flagship at additional seeds so the headline number
# can be reported as mean +- std instead of a single draw.
#
# Byte-identical to save/ACDCDataset/batch_ablation/gradnorm_scaled_b1/cmd.sh
# except --seed and --save_dir.  Seed 0 is that existing run (R150 = 31.92);
# do not re-run it.
#
# usage: bash bash/seed_variance.sh <gpu> <seed> [rounds] [base_rst]
#   base_rst 0.01 = gated (default, the flagship);  0.0 = the no-gate control
set -u
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROUNDS="${3:-150}"
BASE_RST="${4:-0.01}"

[ "$SEED" = "0" ] && { echo "seed 0 already exists as batch_ablation/gradnorm_scaled_b1 -- refusing to re-run" >&2; exit 1; }

TAG="gradnorm_scaled_b1_seed${SEED}"
[ "$BASE_RST" = "0.0" ] && TAG="deyo_mlmp_b1_seed${SEED}"

PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

SAVE="save/ACDCDataset/seed_variance/${TAG}/"
LOG="save/_backbone_logs/${TAG}.log"
mkdir -p "$SAVE" save/_backbone_logs

echo "[$(date '+%m-%d %H:%M')] START ${TAG} (base_rst=${BASE_RST}) gpu=${GPU} rounds=${ROUNDS}"
CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
  --save_dir "$SAVE" --data_dir data/ACDC/ --prompt_dir prompts.yaml \
  --dataset ACDCDataset --workers 1 --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list fog night rain snow \
  --phase1_rounds 0 --class_extensions --split val --subset_seed 0 \
  --corruption_severity 5 --ovss_type naclip --ovss_backbone ViT-L/14 \
  --adapt --method deyo_mlmp_adagate_continual \
  --batch_size 1 --lr 5e-06 --steps 1 --continual_rounds "$ROUNDS" --seed "$SEED" \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 \
  --top_block_exclude 6 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst "$BASE_RST" \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
  --monitor_interval 50 --ln_ckpt_every 0 \
  --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled \
  >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE  ${TAG} (exit $?)"
tail -1 "$SAVE/results_all_rounds.txt" 2>/dev/null
