#!/bin/bash
# Auto-generated: decoupled_buf10_mon50
# Method: tent_divgate_decoupled
# Extras: marginal_buf_size=10, monitor_interval=50

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=1

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="tent_divgate_decoupled"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_THRESHOLD=1.8
H_WARNING=1.5
CAUTIOUS_RST=0.005
BRAKE_RST=0.02

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/decoupled_buf10_mon50/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --marginal_buf_size 10 \
                        --monitor_interval 50 \
                        --save_dir $SAVE_DIR \
                        --class_extensions
