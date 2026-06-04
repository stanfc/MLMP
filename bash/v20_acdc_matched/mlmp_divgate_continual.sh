#!/bin/bash
# MLMP-DivGate-Continual on PascalVOC20 with ACDC-matched round size.
# See docs/v20_acdc_matched_spec.md.
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
METHOD="mlmp_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── MLMP multi-level: last 18 layers of ViT-L/14 ──────────────────
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
PROMPT_INTEGRATION="loss"
ALPHA_CLS=1.0

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Diversity gate (ACDC-best confirmed) ───────────────────────────
H_THRESHOLD=1.6
H_WARNING=1.4
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/v20_acdc_matched/${METHOD}_threshold_${H_THRESHOLD}/}"

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
                        --ann_file $ANN_FILE \
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
