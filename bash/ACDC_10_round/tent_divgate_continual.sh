#!/bin/bash
# TENT-DivGate-Continual on ACDC (CTTA, 150 rounds).
# Pure TENT pixel-wise entropy + diversity-gated stochastic restoration.
# Same gate mechanism as cma_divgate_continual, but the base loss is
# TENT entropy instead of CMA cosine alignment — TENT's natural peak
# (32.4) exceeds MLMP-episodic (30.6), so gating it is the highest-
# leverage anti-collapse experiment.
# See docs/tent_divgate_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="${DATA_DIR:-data/ACDC/}"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh: pure TENT, no top_k)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Diversity gate (proposal_after_cma.md §2.3 defaults; all env-overridable)
H_THRESHOLD=${H_THRESHOLD:-1.8}   # H_margin >= this  -> aggressive (rst=0)
H_WARNING=${H_WARNING:-1.5}       # h_warning <= H < h_threshold -> cautious
MONITOR_INTERVAL=${MONITOR_INTERVAL:-50}   # batches between H_margin re-evaluations
CAUTIOUS_RST=${CAUTIOUS_RST:-0.005}
BRAKE_RST=${BRAKE_RST:-0.02}


CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_caut_0.005_brake_0.02/}"

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
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
&& python plot_results.py \
        --runs $SAVE_DIR \
              save/${DATASET}/tent_divgate_continual_step_1 \
        --labels "caut_0.005_brake_0.02" "previous (caut_0.008_brake_0.01)" \
        --out_dir figures/caut_0.005_brake_0.02
