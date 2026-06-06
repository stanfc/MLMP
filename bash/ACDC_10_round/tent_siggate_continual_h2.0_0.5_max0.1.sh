#!/bin/bash
# TENT-SigGate-Continual on ACDC (CTTA, 150 rounds) — h2.0_0.5_max0.1 variant.
# Hand-tuned so that H_margin=1.8 maps to rst~=0.01 (matches the
# "moderate brake at moderately healthy H" intuition).
#
# Reference points (auto-k = 6/(2.0-0.5) = 4.0, mid=1.25):
#   H=2.0  -> rst ~= 0.0047  (almost no brake)
#   H=1.8  -> rst ~= 0.0100  <-- target
#   H=1.55 -> rst ~= 0.0269
#   H=1.25 -> rst  = 0.0500  (mid)
#   H=1.0  -> rst ~= 0.0731
#   H=0.5  -> rst ~= 0.0953  (saturated near MAX_RST)

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="tent_siggate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh: pure TENT, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Sigmoid diversity gate
H_HIGH=2.0            # upper anchor; rst saturates near 0 here
H_LOW=0.5             # lower anchor; rst saturates near MAX_RST here
MAX_RST=0.1           # upper bound on rst
SIGMOID_K=-1          # <=0 -> auto = 6/(H_HIGH - H_LOW) = 4.0
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_h${H_HIGH}_${H_LOW}_max${MAX_RST}/"

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
                        --sigmoid_k $SIGMOID_K \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
&& python plot_results.py \
        --runs $SAVE_DIR \
              "save/${DATASET}/tent_siggate_continual_max_0.01" \
              "save/${DATASET}/tent_divgate_continual_step_1" \
        --labels "siggate_h${H_HIGH}_${H_LOW}_max${MAX_RST}" "siggate_max_0.01 (default)" "divgate (discrete)" \
        --out_dir "figures/${METHOD}_h${H_HIGH}_${H_LOW}_max${MAX_RST}"
