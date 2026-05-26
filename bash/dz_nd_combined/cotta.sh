#!/bin/bash
# CoTTA on DZ+ND combined stream (2-sub-dataset night CTTA) (CTTA, 150 rounds) with NA-CLIP backbone.
# Three mechanisms: EMA teacher, augmentation-averaged pseudo-labels,
# stochastic restoration. NOTE: RST=0.01 is intentional (the matching
# ACDC source script had RST=0.00 which disables stochastic restoration —
# same bug observed on bash/v20_acdc_matched/cotta.sh).

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=DZ_ND_Combined
DATA_DIR="data/"  # ignored in combined-stream mode
INIT_RESIZE="1120 560"
CONDITIONS="dark_zurich nighttime_driving"
WORKERS=4

# Method Configuration
METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# CoTTA hyperparameters
MT=0.999        # EMA smoothing factor for teacher
RST=0.01        # stochastic restoration probability (BUG FIX vs ACDC source: was 0.00)
AP=0.92         # anchor confidence threshold
AUG_N=32        # number of augmented teacher views

OUT_VISION="-1"

# Hyperparameters
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Experiment
CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}

# Output
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

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
