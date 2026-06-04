#!/bin/bash
# TENT-SigGate-Continual on ACDC (CTTA, 150 rounds) — h1.8_1.0_max0.05 variant.
#
# Tuned from organiser's best DivGate sweep result on save_tekai/save:
#   tent_divgate_continual_cau_rst_0.008  (h_threshold=1.6, h_warning=1.4,
#                                          cautious_rst=0.008, brake_rst=0.05)
#   -> R150 mean = 31.49 (best across the whole DivGate sweep)
#
# DivGate's discrete behaviour translated to a continuous sigmoid:
#   - H_HIGH=1.8  : H above this -> rst ~ 0 (more plasticity than DivGate's 1.6)
#   - H_LOW=1.0   : H below this -> rst saturates near MAX_RST
#   - MAX_RST=0.05: same as DivGate's brake_rst
#   - auto k = 6/(1.8-1.0) = 7.5
#
# At H=1.6 the sigmoid gives rst ~= 0.0089, which matches DivGate's
# best cautious_rst=0.008. So the curve passes through DivGate's best
# (H_THRESHOLD, cautious_rst) point but smooths the two sides:
#   H=1.8  -> rst ~= 0.0024  (gentler than aggressive=0)
#   H=1.6  -> rst ~= 0.0089  <-- matches DivGate cautious=0.008
#   H=1.4  -> rst  = 0.0250  (sigmoid mid)
#   H=1.2  -> rst ~= 0.0411
#   H=1.0  -> rst ~= 0.0476  (saturated near MAX_RST=0.05)

# GPU Configuration
GPU_ID=2

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="tent_siggate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh: pure TENT, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Sigmoid diversity gate
H_HIGH=1.8            # upper anchor; rst saturates near 0 here
H_LOW=1.0             # lower anchor; rst saturates near MAX_RST here
MAX_RST=0.05          # upper bound on rst (matches DivGate brake_rst)
SIGMOID_K=-1          # <=0 -> auto = 6/(H_HIGH - H_LOW) = 7.5
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
              "save_tekai/save/${DATASET}/tent_divgate_continual_cau_rst_0.008" \
              "save/${DATASET}/tent_siggate_continual_max_0.01" \
        --labels "siggate_h${H_HIGH}_${H_LOW}_max${MAX_RST}" "divgate_cau_0.008 (best baseline)" "siggate_max_0.01 (default)" \
        --out_dir "figures/${METHOD}_h${H_HIGH}_${H_LOW}_max${MAX_RST}"
