#!/bin/bash
# KFF (Class-aware Domain Knowledge Fusion and Fission) on Cityscapes
# with NA-CLIP backbone. Continual across the 15 ImageNet-C corruption types.
# Reference: NeurIPS 2025, arXiv:2510.12150.
#
# Note: batch_size kept small (vs DPCore's 64 + micro_batch=4) because the
# current KFF port does not yet support micro-batching. Raise if GPU allows.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=1

# Method and OVSS Model Configuration
METHOD="kff"
OUT_VISION="-1"
PROMPT_DIR="prompts.yaml"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# KFF Hyperparameters (faithful to reference defaults)
BATCH_SIZE=8
LR=1e-4            # cls-prompt optimiser LR
LR_DOMAIN=1e-5     # domain-prompt optimiser LR
STEPS=50           # OOD inner-loop steps (E_OOD)
TRIALS=1
# Domain-prompt base
TAU=3.0
EMA_ALPHA=0.1
THR_D=25.0
N_D=20
LAMDA=1.0
# Class-prompt base
N_C=100
THR_C=0.005
THR_ENT=2.0
ALPHA_C=0.1
# Shared
PROMPT_NUM=8
VERBOSE=false

# Output
SAVE_DIR=".save/${DATASET}/${METHOD}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --continual \
                        --method $METHOD \
                        --prompt_dir $PROMPT_DIR \
                        --vision_outputs $OUT_VISION \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --save_dir $SAVE_DIR \
                        --data_dir $DATA_DIR \
                        --dataset $DATASET \
                        --workers $WORKERS \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $ALL_CORRUPTIONS \
                        \
                        --lr $LR \
                        --lr_domain $LR_DOMAIN \
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --tau $TAU \
                        --ema_alpha $EMA_ALPHA \
                        --thr_d $THR_D \
                        --n_d $N_D \
                        --lamda $LAMDA \
                        --n_c $N_C \
                        --thr_c $THR_C \
                        --thr_ent $THR_ENT \
                        --alpha_c $ALPHA_C \
                        --prompt_num $PROMPT_NUM \
                        $( [ "$VERBOSE" = "true" ] && echo "--verbose_kff" ) \
                        \
                        --class_extensions
