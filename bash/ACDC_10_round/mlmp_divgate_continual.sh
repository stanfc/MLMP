#!/bin/bash
# MLMP-DivGate-Continual on ACDC (CTTA, 150 rounds).
# MLMP multi-prompt multi-level entropy loss + diversity-gated stochastic restoration.
#
# Sanity-check counterpart to bash/cityscapes_continual/mlmp_divgate_continual.sh:
# MLMP-continual on ACDC peaks at ~28-29 mIoU (mean 28.9 step=1) and degrades
# slowly. The gate should prevent late-round drift; if MLMP-DivGate beats
# MLMP-continual here, the same recipe is worth pushing on Cityscapes weather.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="mlmp_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP multi-level: use last 18 layers of ViT-L/14
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
PROMPT_INTEGRATION="loss"
ALPHA_CLS=1.0

# Hyperparameters (match mlmp_continual.sh)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Diversity gate (best confirmed hyperparameters from TENT-DivGate ACDC sweep)
H_THRESHOLD=1.6       # H_margin >= this  -> aggressive (rst=0)
H_WARNING=1.4         # h_warning <= H < h_threshold -> cautious; < h_warning -> brake
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_threshold_${H_THRESHOLD}/}"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --prompt_integration $PROMPT_INTEGRATION \
                        --alpha_cls $ALPHA_CLS \
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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
