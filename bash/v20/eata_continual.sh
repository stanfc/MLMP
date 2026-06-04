#!/bin/bash
# EATA-Continual on PascalVOC20Dataset (CTTA, N rounds).
# Pre-stream Fisher (clean VOC20) + reliable/non-redundant filter + EWC loss (ICML 2022).
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 224x224 + patch 224x224 stride 112 → 1 patch/image (matches MLMP paper).

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=3

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

# ── Corruption conditions (ImageNet-C standard order) ──────────────
CORRUPTIONS_ARRAY=(
    # --- noise ---
    # gaussian_noise
    # shot_noise
    # impulse_noise
    # --- blur ---
    # defocus_blur
    # glass_blur
    # motion_blur
    # zoom_blur
    # --- weather ---
    snow
    frost
    fog
    brightness
    contrast
    # --- digital ---
    # elastic_transform
    # pixelate
    # jpeg_compression
)
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="eata_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── EATA (paper defaults; e_margin = 0.4 * ln(num_classes)) ────────
E_MARGIN=1.198        # 0.4 * ln(20) for VOC20
D_MARGIN=0.2         # cosine-similarity gap for non-redundant filter
FISHER_ALPHA=2000     # EWC weight
FISHER_SIZE=2000      # source samples for Fisher estimation (VOC20 val has ~1449)

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_weather_dmargin_${D_MARGIN}/}"

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
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --e_margin $E_MARGIN \
                        --d_margin $D_MARGIN \
                        --fisher_alpha $FISHER_ALPHA \
                        --fisher_size $FISHER_SIZE \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
