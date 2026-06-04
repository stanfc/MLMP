#!/bin/bash
# Pure TENT-Continual on ACDC, fog-only stream (CTTA, 150 rounds).
# No diversity gate — naive entropy minimization without per-sample reset.
# Companion to tent_divgate_continual_fog_only.sh: same dataset/split/rounds,
# matching STEPS=1 so the gated and ungated runs are directly comparable.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog"
SPLIT="train+val"        # 400 train + 100 val = 500 fog images per round
WORKERS=4

# Method Configuration
METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_divgate_continual_fog_only.sh for fair comparison)
BATCH_SIZE=1
LR=0.00001
STEPS=1

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_fog_only_train+val/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
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
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
