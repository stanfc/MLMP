#!/bin/bash
# TENT-DivGate-Continual on DZ+ND combined stream (2-sub-dataset night CTTA) (CTTA, 150 rounds).
# Pure TENT pixel-wise entropy + diversity-gated stochastic restoration.
# NOTE: H_THRESHOLD=1.6 is the "best confirmed" value from CLAUDE.md
# (ACDC source has 1.8 which performed worse: mean=30.14 vs 31.59).

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=DZ_ND_Combined
DATA_DIR="data/"  # ignored in combined-stream mode
INIT_RESIZE="1120 560"
CONDITIONS="dark_zurich nighttime_driving"
WORKERS=1

# Method Configuration
METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (pure TENT loss, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Diversity gate (CLAUDE.md best-confirmed values)
H_THRESHOLD=1.6       # was 1.8 in ACDC source; 1.6 gives mean=31.59 vs 30.14
H_WARNING=1.4
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}
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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
