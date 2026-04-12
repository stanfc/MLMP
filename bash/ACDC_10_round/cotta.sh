#!/bin/bash
# CoTTA on ACDC (10-round CTTA).
# Implements teacher-student EMA + augmentation-averaged pseudo-labels +
# stochastic restoration, applied to OVSS (NA-CLIP backbone).
# This is the main CoTTA baseline — compare with mlmp_cotta.sh.

# GPU Configuration
GPU_ID=0

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

# Hyperparameters
BATCH_SIZE=1
LR=0.0001      # lower than episodic TTA — updates accumulate over 10 rounds
STEPS=1        # online: 1 step per sample

# CoTTA-specific hyperparameters
EMA_ALPHA=0.999         # teacher EMA smoothing factor
RESTORATION_P=0.01      # fraction of LN weights stochastically restored to source
CONF_THRESHOLD=0.1      # source confidence below which augmentation is applied
N_AUGMENTATIONS=8       # number of augmented teacher views for pseudo-label

# Experiment
CONTINUAL_ROUNDS=10

# Output
SAVE_DIR="save/${DATASET}/${METHOD}/"

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
                        --ema_alpha $EMA_ALPHA \
                        --restoration_p $RESTORATION_P \
                        --conf_threshold $CONF_THRESHOLD \
                        --n_augmentations $N_AUGMENTATIONS \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
