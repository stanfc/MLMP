#!/bin/bash
# No-Adaptation baseline on PascalVOC21Dataset (CTTA, N rounds).
# Runs off-the-shelf NA-CLIP without any weight updates.
# Results are constant across all rounds — zero-shot source model performance.
# Use this to establish the lower bound before comparing continual TTA methods.
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 224x224 + patch 224x224 stride 112 → 1 patch/image (matches MLMP paper).
# This is THE comparable v20/v21 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-3}

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC21Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
CORRUPTIONS_ARRAY=(
    # --- noise ---
    gaussian_noise
    shot_noise
    impulse_noise
    # --- blur ---
    defocus_blur
    glass_blur
    motion_blur
    zoom_blur
    # --- weather ---
    snow
    frost
    fog
    brightness
    contrast
    # --- digital ---
    elastic_transform
    pixelate
    jpeg_compression
)
# One-liner subset override: CORRUPTIONS_LIST="fog snow" bash script.sh
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_continual"   # lightest runner; --adapt is omitted so no updates occur
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
BATCH_SIZE=1
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/No_Adaptation/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
