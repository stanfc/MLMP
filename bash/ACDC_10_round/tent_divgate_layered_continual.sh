#!/bin/bash
# TENT-DivGate-Layered-Continual on ACDC (CTTA, 150 rounds).
#
# Same as tent_divgate_continual, but with HARD FREEZE on the late
# group of visual-encoder LayerNorm parameters:
#   - blocks [0, EARLY_CUTOFF) + ln_pre        -> trainable (early)
#   - blocks [EARLY_CUTOFF, LATE_CUTOFF)        -> trainable (mid)
#   - blocks [LATE_CUTOFF, num_blocks) + ln_post-> FROZEN (late)
#
# DivGate hyper-parameters use the original defaults from
# proposal_after_cma.md §2.3 / docs/tent_divgate_continual_spec.md §7
# so that any difference vs tent_divgate_continual is attributable to
# the layer freeze alone.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="tent_divgate_layered_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh: pure TENT, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Diversity gate (ORIGINAL defaults from proposal_after_cma.md §2.3)
H_THRESHOLD=1.8       # H_margin >= this  -> aggressive (rst=0)
H_WARNING=1.2         # h_warning <= H < h_threshold -> cautious
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations
CAUTIOUS_RST=0.005
BRAKE_RST=0.05

# Layered freeze (24-block ViT-L/14)
EARLY_CUTOFF=8        # block < 8 + ln_pre        -> trainable
LATE_CUTOFF=16        # block >= 16 + ln_post     -> FROZEN

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_cutoff_${EARLY_CUTOFF}_${LATE_CUTOFF}/"

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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --early_cutoff $EARLY_CUTOFF \
                        --late_cutoff $LATE_CUTOFF \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
&& python plot_results.py \
        --runs $SAVE_DIR \
              "save/${DATASET}/tent_divgate_continual_step_1" \
        --labels "layered_freeze_${EARLY_CUTOFF}_${LATE_CUTOFF}" "tent_divgate (no freeze)" \
        --out_dir "figures/${METHOD}_cutoff_${EARLY_CUTOFF}_${LATE_CUTOFF}"
