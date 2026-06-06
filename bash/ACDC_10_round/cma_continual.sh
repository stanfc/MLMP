#!/bin/bash
# CMA-Continual on ACDC (10-round CTTA).
# Cross-Modal Alignment loss: pull confident pixels' visual features toward
# the frozen text embedding of their predicted class.
# Designed to avoid the trivial-solution collapse of entropy minimization.
# See docs/cma_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="cma_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
# Same LR/steps/batch as tent_continual for clean comparison; only the loss differs.
BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=1.0

# Experiment
CONTINUAL_ROUNDS=150

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_k_${TOP_K_PERCENT}/"

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
                        --top_k_percent $TOP_K_PERCENT \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
