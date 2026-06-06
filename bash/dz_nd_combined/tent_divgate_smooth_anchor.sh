#!/bin/bash
# TENT-DivGate-SMOOTH-ANCHOR on DZ+ND combined stream (2-sub-dataset night CTTA).
# Smooth-anchor restoration: restore toward a snapshot lag(H) batches ago
# (depth continuous in H_margin) instead of resetting to frozen source.
#
# FAIR-COMPARISON SETUP: gate geometry matched to the source-reset baseline
# bash/dz_nd_combined/tent_divgate_continual.sh (H_THRESHOLD=1.6, H_WARNING=1.4,
# CAUTIOUS_RST=0.01). Only the restoration TARGET differs.

# GPU Configuration
GPU_ID=${GPU_ID:-0}

# Dataset Configuration
DATASET=DZ_ND_Combined
DATA_DIR="data/"  # ignored in combined-stream mode
INIT_RESIZE="1120 560"
CONDITIONS="dark_zurich nighttime_driving"
WORKERS=1

# Method Configuration
METHOD="tent_divgate_smooth_anchor"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (pure TENT loss, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Smooth anchor (matched to source-reset baseline gate geometry)
H_CEIL=${H_CEIL:-1.6}          # H_margin >= this -> no restore (was H_THRESHOLD)
H_FLOOR=${H_FLOOR:-1.4}        # H_margin <= this -> restore to SOURCE (was H_WARNING)
LAG_SCALE=${LAG_SCALE:-90.0}   # lag(H) = LAG_SCALE / (H - H_FLOOR)
MAX_LAG=${MAX_LAG:-3000}       # deepest non-source anchor / snapshot buffer size
RST=${RST:-0.01}               # fixed restore rate (was CAUTIOUS_RST)
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=600
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
                        --h_ceil $H_CEIL \
                        --h_floor $H_FLOOR \
                        --lag_scale $LAG_SCALE \
                        --max_lag $MAX_LAG \
                        --rst $RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
