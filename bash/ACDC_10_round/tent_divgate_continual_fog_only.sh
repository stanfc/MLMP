#!/bin/bash
# TENT-DivGate-Continual on ACDC, fog-only stream (CTTA, 150 rounds).
# Same gate / hyperparams as tent_divgate_continual.sh, but:
#   - only the fog condition is used (no night/rain/snow rotation)
#   - train + val splits are concatenated -> 500 images per round
# Goal: isolate behaviour on a single domain over a longer per-round stream.

# GPU Configuration
GPU_ID=2

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog"
SPLIT="train+val"        # 400 train + 100 val = 500 fog images per round
WORKERS=1

# Method Configuration
METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh: pure TENT, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Diversity gate (proposal_after_cma.md §2.3 defaults)
H_THRESHOLD=1.8       # H_margin >= this  -> aggressive (rst=0)
H_WARNING=1.5         # h_warning <= H < h_threshold -> cautious
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations
CAUTIOUS_RST=0.005
BRAKE_RST=0.02


CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_fog_only_train+val/"

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
                        --split $SPLIT \
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
