#!/bin/bash
# No-Adaptation baseline on ACDC (10-round continual evaluation).
# Runs the off-the-shelf NA-CLIP model without any weight updates.
# Results are constant across all rounds (no learning occurs).
# Use this to establish the zero-shot source model performance.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Model Configuration
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
METHOD="tent_continual"   # lightest method; --adapt is omitted so no updates occur

# Experiment
CONTINUAL_ROUNDS=10
BATCH_SIZE=1

# Output
SAVE_DIR="save/${DATASET}/No_Adaptation/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
