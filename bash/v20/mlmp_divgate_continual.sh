#!/bin/bash
# MLMP-DivGate-Continual on PascalVOC20Dataset (CTTA, N rounds).
# MLMP multi-prompt multi-level entropy loss + diversity-gated stochastic restoration.
# Pairs MLMP's stronger base loss (vs pure TENT) with the same H_margin gate that
# stabilises TENT-DivGate. Useful when adaptation headroom is small — MLMP's
# multi-level signal extracts more from the same data than pure entropy.
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 224x224 + patch 224x224 stride 112 → 1 patch/image (matches MLMP paper).
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-3}

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
CORRUPTIONS_ARRAY=(
    # --- noise ---
    # gaussian_noise
    # shot_noise
    # impulse_noise
    # # --- blur ---
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
METHOD="mlmp_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── MLMP multi-level: last 18 layers of ViT-L/14 ──────────────────
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
PROMPT_INTEGRATION="loss"
ALPHA_CLS=1.0

# ── Training hyperparameters (match mlmp_continual) ────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Diversity gate (ACDC-best confirmed) ───────────────────────────
# Note: if H_margin distribution on v20 turns out higher than ACDC's
# ~1.79 (gate stays in aggressive mode the entire run), raise
# H_THRESHOLD toward 2.0 — same calibration logic as cityscape.
H_THRESHOLD=1.6
H_WARNING=1.4
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_threshold_${H_THRESHOLD}_weather/}"

# ───────────────────────────────────────────────────────────────────
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
                        --corruptions_list $CORRUPTIONS_LIST \
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
