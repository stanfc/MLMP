#!/bin/bash
# TENT-SigGate-Continual on ACDC (CTTA, 150 rounds).
# Pure TENT pixel-wise entropy + SIGMOID-shaped diversity-gated stochastic
# restoration. Same H_margin signal and Buffer-N machinery as ContGate,
# but the H_margin -> rst mapping is a centred sigmoid:
#
#   mid = (H_HIGH + H_LOW) / 2
#   rst = MAX_RST * sigmoid(-k * (H_margin - mid))
#
# Default k = 6 / (H_HIGH - H_LOW) so the mapping is ~95% saturated at
# H = H_HIGH (rst ~ 0) and at H = H_LOW (rst ~ MAX_RST). Pass any positive
# SIGMOID_K to override.

# GPU Configuration
GPU_ID=3

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
H_HIGH=1.8            # upper anchor; rst saturates near 0 here
H_LOW=0.8             # lower anchor; rst saturates near MAX_RST here
MAX_RST=0.01          # upper bound on rst
SIGMOID_K=-1          # <=0 -> auto = 6/(H_HIGH - H_LOW)
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_max_${MAX_RST}_v2/"
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
                        --h_high $H_HIGH \
                        --h_low $H_LOW \
                        --max_rst $MAX_RST \
                        --sigmoid_k $SIGMOID_K \
                        --monitor_interval $MONITOR_INTERVAL \
                        --gate_log_path $GATE_LOG \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
&& python plot_results.py \
        --runs $SAVE_DIR \
              "save/${DATASET}/tent_contgate_continual_max_0.01" \
              "save/${DATASET}/tent_divgate_continual_step_1" \
        --labels "siggate_max_${MAX_RST}_v2" "contgate (linear)" "divgate (discrete)" \
        --out_dir "figures/${METHOD}_max_${MAX_RST}_v2" \
&& python plot_gate_entropy.py \
        --run $SAVE_DIR \
        --out "figures/${METHOD}_max_${MAX_RST}_v2/gate_entropy.png"
