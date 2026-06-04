#!/bin/bash
# MLMP (episodic) on DZ+ND combined stream (2-sub-dataset night CTTA).
# Standard episodic TTA: reset → adapt → evaluate per sample.
# Episodic upper-bound baseline vs continual methods.

# GPU Configuration
GPU_ID=2

# Dataset Configuration
DATASET=DZ_ND_Combined
DATA_DIR="data/"  # ignored in combined-stream mode
INIT_RESIZE="1120 560"
CONDITIONS="dark_zurich nighttime_driving"
WORKERS=1

# Method Configuration
METHOD="mlmp"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP multi-level: last 18 layers of ViT-L/14
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# Hyperparameters
BATCH_SIZE=1
LR=0.001
STEPS=1
TRIALS=1

# Output
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/mlmp/}"

# Run (episodic: uses main.py)
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
