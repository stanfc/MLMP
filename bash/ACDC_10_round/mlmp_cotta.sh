#!/bin/bash
# MLMP-CoTTA on ACDC (10-round CTTA).
# Combines MLMP's multi-level multi-prompt adaptation signal with CoTTA's
# anti-forgetting mechanisms for robust long-term continual adaptation.
#
# Student training: MLMP loss — multi-prompt CE against teacher pseudo-label
#                               + ILE (CLS entropy) term
# Teacher:          EMA of student, uses adaptive_weighted_mean UAML fusion
# Anti-forgetting:  stochastic restoration of LN weights to source values

# GPU Configuration
GPU_ID=0

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="mlmp_cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP multi-level: use last 18 layers of ViT-L/14 (same as episodic MLMP)
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# Hyperparameters
BATCH_SIZE=1
LR=0.0001      # lower than episodic TTA — updates accumulate over 10 rounds
STEPS=1        # online: 1 step per sample

# CoTTA-specific hyperparameters
EMA_ALPHA=0.999
RESTORATION_P=0.01
CONF_THRESHOLD=0.1
N_AUGMENTATIONS=8

# Experiment
CONTINUAL_ROUNDS=10

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_batch_${BATCH_SIZE}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
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
