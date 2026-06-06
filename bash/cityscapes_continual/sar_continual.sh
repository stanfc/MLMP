#!/bin/bash
# SAR-Continual on CityscapesDataset (CTTA, N rounds).
# SAM optimizer + reliable sample filtering + model recovery (ICLR 2023).
# 15 ImageNet-C corruptions applied on-the-fly.
# See docs/2026-05-17-sar-eata-v20-design.md for the full design.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=2

# ── Dataset ────────────────────────────────────────────────────────
DATASET=CityscapesDataset
DATA_DIR="data/Cityscape/"
INIT_RESIZE="1120 560"
WORKERS=1

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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
# One-liner subset override: CORRUPTIONS_LIST="fog snow frost brightness" bash script.sh
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="sar_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters (match tent_divgate_continual on Cityscapes) ─
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── SAR (paper defaults; e_margin = 0.4 * ln(num_classes)) ─────────
# Cityscapes uses 19 classes → e_margin = 0.4 * ln(19) ≈ 1.178
E_MARGIN=1.8          # sample-level mean-pixel entropy threshold
SAM_RHO=0.05          # SAM perturbation radius (paper default)
E_0=0.1               # recovery threshold: trigger reset when loss_ma < E_0
                      # (SAR paper uses 0.2 for ImageNet/1000-class; scaled by
                      #  ln(C) ratio: 0.2 * ln(19)/ln(1000) ≈ 0.085 → round to 0.1)
EMA_FACTOR=0.9        # loss MA decay
RECOVERY_WARMUP=50    # batches before recovery can fire

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_weather/}"

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
                        --sam_rho $SAM_RHO \
                        --e_0 $E_0 \
                        --ema_factor $EMA_FACTOR \
                        --recovery_warmup $RECOVERY_WARMUP \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
