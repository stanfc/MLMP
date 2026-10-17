#!/bin/bash
# CAGA-hgate: deyo_mlmp_adagate_hgate_continual, backbone ablation arm 'clearclip_kk_B16'
# (ovss_type=clearclip_kk, ovss_backbone=ViT-B/16). Same locked CAGA config otherwise.
# (trend_stat=mad trend_thr=0.5 lag_mode=ecdf shallow_cap_mode=growing_scaled)
# plus anchor_h_min_ratio=0.95: a new grad_norm-minimum anchor is only accepted
# if h_margin is not already degraded relative to its own running peak.
# Separate method name/save_dir -- does not touch CAGA's own results.
# ACDCDataset, 4 conditions (fog/night/rain/snow), 150 rounds.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-0}

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="deyo_mlmp_adagate_hgate_continual"
OVSS_TYPE="clearclip_kk"
OVSS_BACKBONE="ViT-B/16"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/adagate_hgate_backbone_clearclip_kk_B16/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        --dataset $DATASET --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 --patch_stride 112 \
                        --corruptions_list $CONDITIONS --workers $WORKERS \
                        --class_extensions --split val --subset_seed 0 --corruption_severity 5 \
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \
                        --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
                        --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
                        --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
                        --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
                        --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 --monitor_interval 50 \
                        --trend_stat mad --trend_thr 0.5 --trend_hist 50 --lag_mode ecdf --lag_sat 1.5 \
                        --shallow_cap_mode growing_scaled --anchor_h_min_ratio 0.95 \
                        --save_dir $SAVE_DIR

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP backbone_clearclip_kk_B16 ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP backbone_clearclip_kk_B16 ❌"
fi
