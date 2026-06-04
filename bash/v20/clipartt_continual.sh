#!/bin/bash
# CLIPArTT-Continual on PascalVOC20-C, 150 rounds, 15 corruptions.
# Plain CLIPArTT self-distillation, no gate, no restoration.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=2

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=0
SUBSET_SIZE=100
SUBSET_SEED=0

METHOD="clipartt_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CLIPARTT_K=3

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_K${CLIPARTT_K}_sub100/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $ALL_CORRUPTIONS \
                        --workers $WORKERS \
                        --subset_size $SUBSET_SIZE \
                        --subset_seed $SUBSET_SEED \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --clipartt_k $CLIPARTT_K \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
