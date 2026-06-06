#!/bin/bash
# TENT-ContGate-Continual on ACDC (CTTA, 150 rounds).
# Pure TENT pixel-wise entropy + CONTINUOUS diversity-gated stochastic
# restoration. Same H_margin signal and Buffer-N machinery as DivGate,
# but rst is a linear function of H_margin instead of three discrete
# modes:
#
#   ratio = clamp((H_HIGH - H_margin) / (H_HIGH - H_LOW), 0, 1)
#   rst   = ratio * MAX_RST
#
# Goal: remove the discrete jumps in rst that DivGate produces every
# monitor window, which appear to drive the high-frequency saw-tooth
# in the late phase of tent_divgate_continual.

# GPU Configuration
GPU_ID=1

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="tent_contgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh: pure TENT, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Continuous diversity gate
H_HIGH=1.8            # H_margin >= this -> rst = 0 (no brake)
H_LOW=0.8             # H_margin <= this -> rst = MAX_RST (full brake)
MAX_RST=0.01          # upper bound on rst
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_max_${MAX_RST}/"

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
                        --h_high $H_HIGH \
                        --h_low $H_LOW \
                        --max_rst $MAX_RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
&& python plot_results.py \
        --runs $SAVE_DIR \
              "save/${DATASET}/tent_divgate_continual_step_1" \
        --labels "contgate_max_${MAX_RST}" "tent_divgate (discrete)" \
        --out_dir "figures/${METHOD}_max_${MAX_RST}"
