#!/bin/bash
# CMA-Layered-Continual on ACDC (CTTA, 150 rounds).
# Cross-Modal Alignment loss + layer-stratified stochastic restoration:
# early ViT blocks adapt freely, late blocks are strongly anchored to source.
# See docs/cma_layered_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="cma_layered_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (shared with CMA baseline for clean comparison)
BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=0.2

# Layer-stratified restoration (proposal_after_cma.md §1.4 defaults)
# - all 0   -> equivalent to cma_continual (no restoration)
# - all 1   -> LN params fully frozen
# - any 0   -> that group is unrestored
EARLY_RST=0.0001
MID_RST=0.001
LATE_RST=0.01

# Group cutoffs for ViT-L/14 (24 blocks)
# blocks [0, EARLY_CUTOFF)            + ln_pre  -> early
# blocks [EARLY_CUTOFF, LATE_CUTOFF)            -> mid
# blocks [LATE_CUTOFF, 24)            + ln_post -> late
EARLY_CUTOFF=8
LATE_CUTOFF=16

# Experiment (match CMA baseline run length for fair comparison)
CONTINUAL_ROUNDS=150

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_rate_${EARLT_RST}_${MID_RST}_${LATE_RST}/"

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
                        --top_k_percent $TOP_K_PERCENT \
                        --early_rst $EARLY_RST \
                        --mid_rst $MID_RST \
                        --late_rst $LATE_RST \
                        --early_cutoff $EARLY_CUTOFF \
                        --late_cutoff $LATE_CUTOFF \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
