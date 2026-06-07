#!/bin/bash
# No-Adaptation baseline on CityscapesDataset (CTTA, N rounds).
# Runs off-the-shelf NA-CLIP without any weight updates.
# Results are constant across all rounds — zero-shot source model performance.
# Use this to establish the lower bound before comparing continual TTA methods.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-2}

# ── Dataset ────────────────────────────────────────────────────────
DATASET=CityscapesDataset
DATA_DIR="data/Cityscape/"
INIT_RESIZE="1120 560"
WORKERS=1

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
CORRUPTIONS_ARRAY=(
    # ACDC-matched 4 corruptions: snow<->snow, fog<->fog, frost<->rain, contrast<->night
    snow
    frost
    fog
    contrast
)
# One-liner subset override: CORRUPTIONS_LIST="fog snow frost brightness" bash script.sh
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_continual"   # lightest runner; --adapt is omitted so no updates occur
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
BATCH_SIZE=1
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/No_Adaptation_weather/}"

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
                        --subset_size 101 --subset_seed 0 \
                        \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
