#!/bin/bash
# EATA-Continual on ACDC (CTTA, 150 rounds).
# Pre-stream Fisher (clean source proxy) + reliable/non-redundant filter
# + Fisher-weighted EWC loss (ICML 2022).
# See docs/2026-05-17-sar-eata-v20-design.md for the full design.
#
# NOTE on Fisher source for ACDC:
#   ACDC has no "clean" split. We use the first condition (fog) as a
#   source proxy — same convention as cma_proto_continual / dpcore on ACDC.
#   Pass --src_corruption fog explicitly; main_continual.py's dispatch
#   for eata_continual falls back to corruption="original" by default,
#   but ACDC's dataset config treats the condition as the corruption,
#   so we override here via --src_corruption.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=3

# ── Dataset ────────────────────────────────────────────────────────
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# ── Method ─────────────────────────────────────────────────────────
METHOD="eata_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── EATA (paper defaults; e_margin = 0.4 * ln(num_classes)) ────────
# ACDC uses 19 Cityscapes classes → e_margin = 0.4 * ln(19) ≈ 1.178
E_MARGIN=1.178        # sample-level mean-pixel entropy threshold
D_MARGIN=0.2         # cosine-similarity gap for non-redundant filter
FISHER_ALPHA=2000     # EWC weight
FISHER_SIZE=2000      # source samples for Fisher (fog has ~2000 ACDC val images)

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
                        --d_margin $D_MARGIN \
                        --fisher_alpha $FISHER_ALPHA \
                        --fisher_size $FISHER_SIZE \
                        --src_corruption fog \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
