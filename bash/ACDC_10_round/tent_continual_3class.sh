#!/bin/bash
# TENT-Continual on ACDC with 19 classes merged into 3 super-classes (CTTA, 150 rounds).
# STEPS=1 + 150 rounds matches the 19-class baseline at
# save_tekai/save/ACDCDataset/tent_continual_Round150_lr_0.00001/

GPU_ID=${GPU_ID:-3}

DATASET=ACDCMerged3Dataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

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
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
