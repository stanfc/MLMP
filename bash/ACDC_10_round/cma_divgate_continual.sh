#!/bin/bash
# CMA-DivGate-Continual on ACDC (CTTA, 150 rounds).
# Cross-Modal Alignment loss + diversity-gated stochastic restoration:
# H_margin (marginal class entropy) is computed every MONITOR_INTERVAL
# batches; mode switches between aggressive (rst=0), cautious, brake.
# See docs/cma_divgate_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="cma_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (shared with CMA baseline for clean comparison)
BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=0.2

# Diversity gate (proposal_after_cma.md §2.3 defaults)
H_THRESHOLD=1.8       # H_margin >= this  -> aggressive (rst=0)
H_WARNING=1.2         # h_warning <= H < h_threshold -> cautious
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations
CAUTIOUS_RST=0.005
BRAKE_RST=0.005

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_brake_${BRAKE_RST}"

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
                        --top_k_percent $TOP_K_PERCENT \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
