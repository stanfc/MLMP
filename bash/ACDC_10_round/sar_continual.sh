#!/bin/bash
# SAR-Continual on ACDC (CTTA, 150 rounds).
# SAM optimizer + reliable sample filtering + model recovery (ICLR 2023).
# See docs/2026-05-17-sar-eata-v20-design.md for the full design.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=3

# ── Dataset ────────────────────────────────────────────────────────
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# ── Method ─────────────────────────────────────────────────────────
METHOD="sar_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters (match tent_divgate_continual.sh on ACDC) ─
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── SAR (paper defaults; e_margin = 0.4 * ln(num_classes)) ─────────
# ACDC uses 19 Cityscapes classes → e_margin = 0.4 * ln(19) ≈ 1.178
E_MARGIN=1.178        # sample-level mean-pixel entropy threshold
SAM_RHO=0.05          # SAM perturbation radius (paper default)
E_0=0.2               # recovery threshold for loss EMA
EMA_FACTOR=0.9        # loss MA decay
RECOVERY_WARMUP=50    # batches before recovery can fire

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

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
                        --e_margin $E_MARGIN \
                        --sam_rho $SAM_RHO \
                        --e_0 $E_0 \
                        --ema_factor $EMA_FACTOR \
                        --recovery_warmup $RECOVERY_WARMUP \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
