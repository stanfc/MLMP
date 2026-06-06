#!/bin/bash
# CMA-Proto-Continual on ACDC (CTTA).
# Cross-Modal Alignment + per-class prototype memory bank (source fixed + target EMA).
# Designed to break the confirmation-bias loop that caused CMA-continual to collapse.
# See docs/cma_proto_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

# Method Configuration
METHOD="cma_proto_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (shared with CMA baseline for clean comparison)
BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=0.2

# CMA-Proto specific loss weights
# λ_tgt = 0 degrades to source-only hard-anchor variant (Option A in design)
LAMBDA_CMA=1.0
LAMBDA_SRC=1.0
LAMBDA_TGT=0.5
EMA_ALPHA=0.999

# Source prototype initialization (uses fog as source proxy; no labels needed)
SRC_CORRUPTION=fog
SRC_CONF_THRESHOLD=0.5
SRC_MAX_SAMPLES=5000

# Experiment (match CMA baseline run length for fair comparison)
CONTINUAL_ROUNDS=150

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

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
                        --lambda_cma $LAMBDA_CMA \
                        --lambda_src $LAMBDA_SRC \
                        --lambda_tgt $LAMBDA_TGT \
                        --ema_alpha $EMA_ALPHA \
                        --src_corruption $SRC_CORRUPTION \
                        --src_conf_threshold $SRC_CONF_THRESHOLD \
                        --src_max_samples $SRC_MAX_SAMPLES \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
