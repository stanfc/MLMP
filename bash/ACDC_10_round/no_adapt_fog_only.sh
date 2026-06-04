#!/bin/bash
# No-Adaptation baseline on ACDC, fog-only stream (CTTA evaluation, 150 rounds).
# Off-the-shelf NA-CLIP, no weight updates. Results should be ~constant across
# rounds (zero-shot source model). Companion to tent / tent_divgate fog-only runs.

# GPU Configuration
GPU_ID=2

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog"
SPLIT="train+val"        # 400 train + 100 val = 500 fog images per round
WORKERS=4

# Model Configuration
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
METHOD="tent_continual"   # lightest method; --adapt is omitted so no updates occur

# Experiment
# No --adapt: weights are frozen, so every round produces identical results.
# 1 round is sufficient — pad with the constant in plotting if needed.
CONTINUAL_ROUNDS=1
BATCH_SIZE=1

# Output
SAVE_DIR="save/${DATASET}/No_Adaptation_fog_only_train+val/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --split $SPLIT \
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
