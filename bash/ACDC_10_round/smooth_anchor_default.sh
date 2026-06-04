#!/bin/bash
# tent_divgate_smooth_anchor: continuous H_margin -> anchor_lag mapping
#
# lag(H) = 90 / (H - 1.5):
#   H=1.80 -> lag=300        (=recent_anchor default)
#   H=1.65 -> lag=600
#   H=1.55 -> lag=1800
#   H<=1.50 -> source        (also H giving lag > max_lag=3000 -> source)
#   H>=1.80 -> no restore
# Fixed rst=0.005 throughout the restore-active region.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=0

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=0

METHOD="tent_divgate_smooth_anchor"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_CEIL=1.8
H_FLOOR=1.5
LAG_SCALE=90.0
MAX_LAG=3000
RST=0.005
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/smooth_anchor_default/"

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
                        --h_ceil $H_CEIL \
                        --h_floor $H_FLOOR \
                        --lag_scale $LAG_SCALE \
                        --max_lag $MAX_LAG \
                        --rst $RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        --save_dir $SAVE_DIR \
                        --class_extensions
