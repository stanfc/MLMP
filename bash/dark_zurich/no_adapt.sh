#!/bin/bash
# No-Adaptation baseline on Dark Zurich (CTTA setup, 150 rounds).
# Runs the off-the-shelf NA-CLIP model without any weight updates.
# Establishes the zero-shot source mIoU for this dataset.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
INIT_RESIZE="1120 560"
CONDITIONS="night"
WORKERS=1

# Model Configuration
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
METHOD="tent_continual"   # lightest method; --adapt omitted so no updates occur

# Experiment
CONTINUAL_ROUNDS=1
BATCH_SIZE=1

# Output
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/No_Adaptation/}"

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
