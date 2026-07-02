#!/bin/bash
# deyo_mlmp_hmgate2_continual on CityscapesDataset full 5corr (50R)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=3

DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=0

METHOD="deyo_mlmp_hmgate2_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=50

SAVE_DIR="save/${DATASET}/deyo_mlmp_hmgate2_continual_full_5corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        --dataset $DATASET --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 --patch_stride 112 \
                        --corruptions_list $CONDITIONS --workers $WORKERS \
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \
                        --vision_outputs $OUT_VISION \
                        --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 \
                        --base_rst 0.01 --h_drop_ratio 0.9 --maxlag_shallow 6 \
                        --monitor_interval 50 \
                        --save_dir $SAVE_DIR \
                        --class_extensions
