#!/bin/bash
# TENT-SigGate-Continual on ACDC (CTTA, 150 rounds) — wide-range variant.
# H_HIGH=3.0, H_LOW=0.1, MAX_RST=0.1.
# Sigmoid spans (almost) the full theoretical H_margin range
# (0..log(19)~2.94), and MAX_RST is 10x the default (0.01).
# With auto-k = 6/(H_HIGH-H_LOW) ~= 2.07 the curve is very gentle in the
# healthy region and only saturates near MAX_RST when H_margin -> 0.

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
H_HIGH=3.0            # upper anchor; rst saturates near 0 here
H_LOW=0.1             # lower anchor; rst saturates near MAX_RST here
MAX_RST=0.1           # upper bound on rst (10x larger than the default 0.01 run)
SIGMOID_K=-1          # <=0 -> auto = 6/(H_HIGH - H_LOW)
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
