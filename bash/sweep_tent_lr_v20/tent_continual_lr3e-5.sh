#!/bin/bash
# TENT-Continual LR sweep on PascalVOC20-C (CTTA, 150 rounds, 5 corruptions).
# LR=3e-5

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=3

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=0
SUBSET_SIZE=100
SUBSET_SEED=0

METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=3e-5
STEPS=1

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_lr${LR}_5corr_sub100/"

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
                        --corruptions_list $CONDITIONS \
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
                        --save_dir $SAVE_DIR \
                        --class_extensions
