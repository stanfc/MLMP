#!/bin/bash
# DELTA-Continual (DOT-only) on ACDC (CTTA, 150 rounds).
# TENT entropy minimization + class-aware Dynamic Online re-weighting (ICLR 2023).
# TBR is omitted (NA-CLIP uses LayerNorm, no BN). See adapt/delta_continual.py.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=2

# ── Dataset ────────────────────────────────────────────────────────
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# ── Method ─────────────────────────────────────────────────────────
METHOD="delta_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters (match tent_divgate_continual.sh on ACDC) ─
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── DOT hyperparameters (paper-style defaults) ─────────────────────
# dot_momentum = EMA decay for the class-frequency tracker.
# dot_alpha    = exponent on inverse frequency; 0=plain TENT, 1=full inverse-freq.
DOT_MOMENTUM=0.9
DOT_ALPHA=1.0

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_alpha_${DOT_ALPHA}_mom_${DOT_MOMENTUM}/}"

# ───────────────────────────────────────────────────────────────────
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
                        --dot_momentum $DOT_MOMENTUM \
                        --dot_alpha $DOT_ALPHA \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
