#!/bin/bash
# TENT-Continual baseline on ACDC (10-round CTTA).
# Standard entropy minimization WITHOUT per-sample reset — naive continual TTA.
# Expected to degrade over time due to error accumulation and catastrophic forgetting
# (reproduces the "TENT-continual" row of CoTTA Table 5).

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
# Lower LR than episodic TTA — continual updates accumulate over 10 rounds
BATCH_SIZE=1
LR=0.00001
STEPS=10

# Experiment
CONTINUAL_ROUNDS=10

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

# Run
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
