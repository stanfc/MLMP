#!/bin/bash
# CLIPArTT-Continual on ACDC (CTTA, 150 rounds).
# Plain CLIPArTT self-distillation loss, no gate, no restoration.
# Continual analogue of episodic adapt/clipartt.py.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="clipartt_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
BATCH_SIZE=1
LR=0.00001
STEPS=1

CLIPARTT_K=3          # top-K classes per pixel for "A or B or C" prompts

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_K${CLIPARTT_K}/"

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
                        --clipartt_k $CLIPARTT_K \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
