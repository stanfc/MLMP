#!/bin/bash
# MLMP-Continual on ACDC (5 corruptions, sub=100, 150 rounds).
# Full MLMP loss WITH ILE (alpha_cls=1.0). No gate, no restoration.
# Companion to mlmp_continual_no_ile (alpha_cls=0.0)
# and mlmp_smooth_anchor_continual (with gate).

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0        # ILE enabled
PROMPT_INTEGRATION="loss"

BATCH_SIZE=1
LR=0.000005   # 5e-6
STEPS=1

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_integration $PROMPT_INTEGRATION \
                        --alpha_cls $ALPHA_CLS \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
