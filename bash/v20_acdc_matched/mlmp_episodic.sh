#!/bin/bash
# MLMP (episodic) on PascalVOC20 with ACDC-matched subset.
# Per-sample reset upper bound. Uses main.py (not main_continual.py) — no
# round concept, but runs on the same 101-img subset per corruption so the
# mean mIoU is directly comparable to bash/v20_acdc_matched/ continual runs.
# See docs/v20_acdc_matched_spec.md.
#
# ─── Patch convention (DO NOT CHANGE) ───
# INIT_RESIZE 224x224 + patch 224x224 stride 112 → 1 patch/image.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=2

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

# ── ACDC-matched subset (101 × 4 = 404 ≈ ACDC's 406/round) ─────────
IMAGES_PER_CORRUPTION=101
SUBSET_SEED=0
ANN_FILE="${DATA_DIR}ImageSets/Segmentation/val_subset_${IMAGES_PER_CORRUPTION}_seed${SUBSET_SEED}.txt"
if [ ! -f "$ANN_FILE" ]; then
    echo "+++ Subset file missing, generating: $ANN_FILE"
    python scripts/make_voc_subset.py --n $IMAGES_PER_CORRUPTION --seed $SUBSET_SEED
fi

# ── Corruption conditions (4 closest to ACDC: snow/fog/frost/contrast) ──
# Mapping rationale:
#   snow     ↔ ACDC snow      (direct)
#   fog      ↔ ACDC fog       (direct)
#   frost    ↔ ACDC rain      (both wet/icy outdoor weather, surface coverage)
#   contrast ↔ ACDC night     (low contrast ≈ poor visibility)
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
    # brightness
    contrast
    # --- digital ---
    # elastic_transform
    # pixelate
    # jpeg_compression
)
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="mlmp"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── MLMP multi-level: last 18 layers of ViT-L/14 ──────────────────
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# ── Training hyperparameters (match bash/v20/mlmp_episodic.sh) ─────
# Higher LR — safe because state resets every sample.
BATCH_SIZE=1
LR=0.001
STEPS=1
TRIALS=1

# ── Experiment ─────────────────────────────────────────────────────
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/v20_acdc_matched/mlmp_episodic/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --ann_file $ANN_FILE \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
