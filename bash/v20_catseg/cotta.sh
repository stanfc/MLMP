#!/bin/bash
# GPU Configuration
GPU_ID=0

# Dataset Configuration
DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="384 384"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=4

# Method and OVSS Model Configuration
METHOD="cotta"
OUT_VISION="-1"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/14"
CATSEG_CKPT=".weights/catseg/model_large.pth"

# CoTTA Hyperparameters
BATCH_SIZE=32
MICRO_BATCH_SIZE=4
LR=1e-5
STEPS=1
TRIALS=1
MT=0.999
RST=0.01
AP=0.7
AUG_N=32
FINETUNE_MODE="ln"
PROMPT_INTEGRATION="text"

# Output
SAVE_DIR=".save/${DATASET}_catseg/${METHOD}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --continual \
                        --method $METHOD \
                        --prompt_dir $PROMPT_DIR \
                        --vision_outputs $OUT_VISION \
                        --alpha_cls $ALPHA_CLS \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone "$OVSS_BACKBONE" \
                        $( [ -n "$CATSEG_CKPT" ] && echo "--catseg_checkpoint $CATSEG_CKPT" ) \
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
                        --prompt_integration $PROMPT_INTEGRATION \
                        --mt $MT \
                        --rst $RST \
                        --ap $AP \
                        --aug_n $AUG_N \
                        --finetune_mode $FINETUNE_MODE \
                        \
                        --plot_loss \
                        --class_extensions
