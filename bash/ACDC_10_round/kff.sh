#!/bin/bash
# KFF (Class-aware Domain Knowledge Fusion and Fission) on ACDC (10-round CTTA)
# with NA-CLIP backbone. Reference: NeurIPS 2025, arXiv:2510.12150.
#
# Faithful port of the ImageNet-C KFF paper's ours.py/vpt.py to OVSS:
#   - Domain-prompt base with L2-distance ID/OOD + softmax weighted sum + EMA
#   - Class-aware prompt with cosine-similarity matching + fission/fusion
#   - Loss = distribution_loss + 3 * per-pixel entropy
#   - Two separate optimizers (domain-prompt LR and cls-prompt LR)

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="224 224"
CONDITIONS="fog night rain snow"
WORKERS=1

# Source statistics: clean Cityscapes (matching DPCore convention).
SRC_DATASET=CityscapesDataset
SRC_DATA_DIR=".data/cityscapes/"

# Method Configuration
METHOD="kff"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1"
PROMPT_DIR="prompts.yaml"

# KFF Hyperparameters (reference defaults in cfgs/*.yaml)
BATCH_SIZE=16
LR=1e-4            # cls-prompt optimiser LR (KFF paper: OPTIM.LR)
LR_DOMAIN=1e-5     # domain-prompt optimiser LR (KFF paper: OPTIM.LR_DOMAIN)
STEPS=50           # OOD inner-loop steps (E_OOD)
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

# CTTA
CONTINUAL_ROUNDS=10

# Output
SAVE_DIR=".save/${DATASET}/kff_batch_${BATCH_SIZE}_LR_${LR}/"

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
                        --lr_domain $LR_DOMAIN \
                        --steps $STEPS \
                        --continual_rounds $CONTINUAL_ROUNDS \
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
                        --src_dataset $SRC_DATASET \
                        --src_data_dir $SRC_DATA_DIR \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
