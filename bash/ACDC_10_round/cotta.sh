#!/bin/bash
# CoTTA on ACDC (10-round CTTA) with NA-CLIP backbone.
# Open-vocabulary segmentation: class names are text prompts, no closed-set head.
#
# CoTTA three mechanisms:
#   1. EMA teacher (mt=0.999) for stable pseudo-labels
#   2. Augmentation-averaged pseudo-labels when anchor confidence < ap
#   3. Stochastic restoration (rst=0.01) to prevent catastrophic forgetting

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# CoTTA hyperparameters (matching original CoTTA paper values)
MT=0.999        # EMA smoothing factor for teacher
RST=0.00        # stochastic restoration probability
AP=0.92         # anchor confidence threshold (augment when mean conf < AP)
AUG_N=32        # number of augmented teacher views

# Use last layer only (standard CoTTA spirit — no multi-level fusion)
OUT_VISION="-1"

# Hyperparameters
BATCH_SIZE=1
LR=0.00001     # lower LR for continual — updates accumulate over 10 rounds
STEPS=1        # online: 1 step per sample

# Experiment
CONTINUAL_ROUNDS=150

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}_no_restore/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --mt $MT \
                        --rst $RST \
                        --ap $AP \
                        --aug_n $AUG_N \
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
