#!/bin/bash
# CoTTA on PascalVOC20 with ACDC-matched round size (404 ≈ 406/round).
# See docs/v20_acdc_matched_spec.md.
#
# CoTTA three mechanisms:
#   1. EMA teacher (mt=0.999) for stable pseudo-labels
#   2. Augmentation-averaged pseudo-labels when anchor confidence < ap
#   3. Stochastic restoration (rst=0.00 in v20 baseline)
#
# ─── Patch convention (DO NOT CHANGE) ───
# INIT_RESIZE 224x224 + patch 224x224 stride 112 → 1 patch/image.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=3

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
METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── CoTTA hyperparameters (match bash/v20/cotta.sh) ────────────────
MT=0.999        # EMA smoothing factor for teacher
RST=0.01        # stochastic restoration probability
AP=0.92         # anchor confidence threshold (augment when mean conf < AP)
AUG_N=32        # number of augmented teacher views

# Use last layer only (standard CoTTA spirit — no multi-level fusion)
OUT_VISION="-1"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/v20_acdc_matched/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --mt $MT \
                        --rst $RST \
                        --ap $AP \
                        --aug_n $AUG_N \
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
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
