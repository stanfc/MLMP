#!/bin/bash
# DPCore on ACDC (10-round CTTA) with NA-CLIP backbone.
# Open-vocabulary segmentation: visual prompts adapt to each domain.
#
# DPCore three components:
#   1. Visual Prompt Adaptation (VPA): learns prompts to align test batch to source stats
#   2. Prompt Coreset (PC): stores (prompt, stats) pairs for seen domains
#   3. Dynamic Update (DU): reuses coreset prompts for ID domains, learns new ones for OOD

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="224 224"
CONDITIONS="fog night rain snow"
WORKERS=4

# DPCore source statistics: use clean Cityscapes, NOT fog-as-source.
# Fog-as-source makes loss_raw≈0 for fog test batches → ID condition impossible.
SRC_DATASET=CityscapesDataset
SRC_DATA_DIR="data/Cityscape/"

# Method Configuration
METHOD="dpcore"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1"
PROMPT_DIR="prompts.yaml"

# DPCore Hyperparameters (from paper defaults, tuned for ACDC)
BATCH_SIZE=32
LR=1e-5
STEPS=50            # OOD inner-loop steps (was 50, reduced: ViT-L/14 diverges with large steps)
TEMP_TAU=1.0       # temperature for coreset weight softmax (lower=nearest-neighbor, higher=average blend)
EMA_ALPHA=0.999    # EMA momentum for coreset statistics update
THR_RHO=0.5       # loss ratio threshold for ID/OOD decision (higher=easier to be ID, lower=harder)
PROMPT_NUM=8       # number of visual prompt tokens
VERBOSE=false       # enable to see loss_raw/loss_new/ratio per batch for debugging

# CTTA
CONTINUAL_ROUNDS=10

# Output
SAVE_DIR="save/${DATASET}/dpcore_batch_${BATCH_SIZE}_LR_${LR}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --batch_size $BATCH_SIZE \
                        --lr $LR \
                        --steps $STEPS \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --temp_tau $TEMP_TAU \
                        --ema_alpha $EMA_ALPHA \
                        --thr_rho $THR_RHO \
                        --prompt_num $PROMPT_NUM \
                        $( [ "$VERBOSE" = "true" ] && echo "--verbose_dpcore" ) \
                        \
                        --src_dataset $SRC_DATASET \
                        --src_data_dir $SRC_DATA_DIR \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
