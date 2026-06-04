#!/bin/bash
# MLMP (episodic) on ACDC, fog-only (train+val).
# Standard episodic TTA: reset → adapt → evaluate per sample.
# No state carries between samples, so rounds are meaningless here — single pass.
# Establishes the episodic upper bound on the same fog-train+val stream the
# continual methods see.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog"
SPLIT="train+val"        # 400 train + 100 val = 500 fog images
WORKERS=4

# Method Configuration
METHOD="mlmp"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP multi-level: use last 18 layers of ViT-L/14
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# Hyperparameters (match mlmp.sh — episodic upper-bound config)
BATCH_SIZE=1
LR=0.001
STEPS=1
TRIALS=1

# Output
SAVE_DIR="save/${DATASET}/mlmp_episodic_fog_only_train+val/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --split $SPLIT \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
