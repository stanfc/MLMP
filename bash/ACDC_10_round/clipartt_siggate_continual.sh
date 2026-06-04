#!/bin/bash
# CLIPArTT-SigGate-Continual on ACDC (CTTA, 150 rounds).
# CLIPArTT self-distillation base loss + sigmoid-shaped diversity-gated
# stochastic restoration. Identical gate + restore machinery to
# tent_siggate_continual; only the base loss differs.

# GPU Configuration
GPU_ID=1

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=0

# Method Configuration
METHOD="clipartt_siggate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
BATCH_SIZE=1
LR=0.00001
STEPS=1

# CLIPArTT base
CLIPARTT_K=3          # top-K classes per pixel for "A or B or C" prompts

# Sigmoid diversity gate (same defaults as tent_siggate_continual)
H_HIGH=1.8
H_LOW=0.8
MAX_RST=0.01
SIGMOID_K=-1
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_K${CLIPARTT_K}_max_${MAX_RST}/"
GATE_LOG="${SAVE_DIR}gate_log.csv"

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
                        --clipartt_k $CLIPARTT_K \
                        --h_high $H_HIGH \
                        --h_low $H_LOW \
                        --max_rst $MAX_RST \
                        --sigmoid_k $SIGMOID_K \
                        --monitor_interval $MONITOR_INTERVAL \
                        --gate_log_path $GATE_LOG \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
