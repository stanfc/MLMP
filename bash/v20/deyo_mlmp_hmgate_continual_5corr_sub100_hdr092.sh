#!/bin/bash
# deyo_mlmp_hmgate_continual on PascalVOC20Dataset (5corr sub100).
# Composite gate: mean_conf TRIGGER + grad_norm DEPTH. This is the KEY test —
# H_margin is blind to VOC20's uniform degradation, but grad_norm tracks it and
# mean_conf is monotone, so the composite should brake here too.
# conf_ceil from observed mean_conf at the mIoU peak (VOC20 peak conf 0.732 @R59).
# See docs/2026-06-18-contribution.md §7.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=2

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=0

METHOD="deyo_mlmp_hmgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

# slope_window 10 (~500 batch) was too noisy on V20 -> false early triggering
# capped the climb (peak only 75.0). 50 windows ~= 2500 batch (5 rounds) smooths
# the medium-term trend so restore only fires on a real sustained grad rise.
SLOPE_WINDOW=10
SLOPE_DEADZONE=0.002
LAG_GAIN=1500
BASE_RST=0.01
H_DROP_RATIO=0.92
MAXLAG_SHALLOW=6
MONITOR_INTERVAL=50

SAVE_DIR="save/${DATASET}/${METHOD}_5corr_sub100_hdr092/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        --subset_size 100 \
                        --subset_seed 0 \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --slope_window $SLOPE_WINDOW \
                        --slope_deadzone $SLOPE_DEADZONE \
                        --lag_gain $LAG_GAIN \
                        --base_rst $BASE_RST \
                        --h_drop_ratio $H_DROP_RATIO \
                        --maxlag_shallow $MAXLAG_SHALLOW \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
