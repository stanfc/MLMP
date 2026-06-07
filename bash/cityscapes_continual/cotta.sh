#!/bin/bash
# CoTTA on CityscapesDataset (CTTA, N rounds) with NA-CLIP backbone.
# Open-vocabulary segmentation: class names are text prompts, no closed-set head.
# 15 ImageNet-C corruptions applied on-the-fly.
#
# CoTTA three mechanisms:
#   1. EMA teacher (mt=0.999) for stable pseudo-labels
#   2. Augmentation-averaged pseudo-labels when anchor confidence < ap
#   3. Stochastic restoration (rst=0.01) to prevent catastrophic forgetting

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
METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── CoTTA hyperparameters (matching original CoTTA paper values) ────
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
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

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
                        --workers $WORKERS \
                        --subset_size 101 --subset_seed 0 \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
