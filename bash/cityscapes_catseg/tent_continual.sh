#!/bin/bash
# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1152 768"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=4

# Method and OVSS Model Configuration
METHOD="tent"
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/14"
CATSEG_CKPT=".weights/catseg/model_large.pth"

# Hyperparameters
BATCH_SIZE=64
MICRO_BATCH_SIZE=4
LR=1e-5
STEPS=1
TRIALS=1

# Output
SAVE_DIR=".save/${DATASET}_catseg/${METHOD}_continual/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --continual \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone "$OVSS_BACKBONE" \
                        --catseg_checkpoint $CATSEG_CKPT \
                        \
                        --save_dir $SAVE_DIR \
                        --data_dir $DATA_DIR \
                        --dataset $DATASET \
                        --workers $WORKERS \
                        --init_resize $INIT_RESIZE \
                        --patch_size 384 384 \
                        --patch_stride 192 \
                        --corruptions_list $ALL_CORRUPTIONS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --micro_batch_size $MICRO_BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --class_extensions
