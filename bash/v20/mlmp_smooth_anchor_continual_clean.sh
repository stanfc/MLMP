#!/bin/bash
# MLMP-SmoothAnchor-Continual on PascalVOC20 CLEAN (no corruption, 150 rounds).
# Pair with bash/v20/tent_continual_clean.sh and mlmp_episodic_clean.sh
# to see whether MLMP+gate stays stable on clean V20 where TENT collapses.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=1

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="original"
WORKERS=1
SUBSET_SIZE=100
SUBSET_SEED=0

METHOD="mlmp_smooth_anchor_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP base
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
PROMPT_INTEGRATION="loss"

# Optimization
BATCH_SIZE=1
LR=0.000005     # 5e-6 (matches ACDC mlmp_continual.sh)
STEPS=1

# SmoothAnchor gate
H_CEIL=1.8
H_FLOOR=1.5
LAG_SCALE=90.0
MAX_LAG=3000
RST=0.005
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_clean_sub${SUBSET_SIZE}/"

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
                        --subset_size $SUBSET_SIZE \
                        --subset_seed $SUBSET_SEED \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_integration $PROMPT_INTEGRATION \
                        --alpha_cls $ALPHA_CLS \
                        --h_ceil $H_CEIL \
                        --h_floor $H_FLOOR \
                        --lag_scale $LAG_SCALE \
                        --max_lag $MAX_LAG \
                        --rst $RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
