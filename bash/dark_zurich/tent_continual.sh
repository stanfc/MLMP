#!/bin/bash
# TENT-Continual on Dark Zurich (CTTA, 150 rounds).
# Entropy minimization without per-sample reset — naive continual TTA.
# NOTE: STEPS=1 and CONTINUAL_ROUNDS=150 vs ACDC source's legacy
# STEPS=10 / ROUNDS=10 (matches the new 150R protocol).

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
INIT_RESIZE="1120 560"
CONDITIONS="night"
WORKERS=4

# Method Configuration
METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
BATCH_SIZE=1
LR=0.00001
STEPS=1                       # was 10 in ACDC source legacy script

# Experiment
CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}   # was 10 in ACDC source legacy script

# Output
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

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
