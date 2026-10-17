#!/bin/bash
# CAGA (hgate, locked config), sub100 5corr, 150 rounds. Also frozen-weight eval on held-out corruptions every round.
# Stream: sub100, 5 weather corruptions (snow frost fog brightness contrast), 150 rounds.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-0}

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=0

METHOD="deyo_mlmp_adagate_hgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150
SUBSET_SIZE=100
SUBSET_SEED=0

SAVE_DIR="save/${DATASET}/adagate_hgate_5corr_sub100_oodprobe/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        --dataset $DATASET --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 --patch_stride 112 \
                        --corruptions_list $CONDITIONS --workers $WORKERS \
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \
                        --subset_size $SUBSET_SIZE --subset_seed $SUBSET_SEED \
                        --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
                        --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
                        --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
                        --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
                        --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 --monitor_interval 50 \
                        --trend_stat mad --trend_thr 0.5 --trend_hist 50 --lag_mode ecdf --lag_sat 1.5 \
                        --shallow_cap_mode growing_scaled --anchor_h_min_ratio 0.95 \
                        --ood_corruptions gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
                        --save_dir $SAVE_DIR \
                        --class_extensions

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP 5corr_oodprobe ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP 5corr_oodprobe ❌"
fi
