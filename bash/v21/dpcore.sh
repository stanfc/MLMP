#!/bin/bash
# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=PascalVOC21Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=4

# Method and OVSS Model Configuration
METHOD="dpcore"
OUT_VISION="-1"
PROMPT_DIR="prompts.yaml"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# DPCore Hyperparameters
BATCH_SIZE=128
LR=5e-4
STEPS=50
TRIALS=1
TEMP_TAU=3.0
EMA_ALPHA=0.999
THR_RHO=0.9
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
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --temp_tau $TEMP_TAU \
                        --ema_alpha $EMA_ALPHA \
                        --thr_rho $THR_RHO \
                        --prompt_num $PROMPT_NUM \
                        $( [ "$VERBOSE" = "true" ] && echo "--verbose_dpcore" ) \
                        \
                        --class_extensions
