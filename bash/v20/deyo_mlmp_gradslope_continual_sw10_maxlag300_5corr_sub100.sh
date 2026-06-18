#!/bin/bash
# deyo_mlmp_gradslope_continual on VOC20 (5corr sub100) — sw10 + SHALLOW lag.
# sw10 gives a stable LINEAR climb because it brakes ~52% of windows, but each
# brake used lag~2889 (restore ~6 rounds back) which flattens the climb.
# Here max_lag=300 keeps the same high-frequency braking (stability) but each
# brake only undoes ~0.6 round -> gentler -> STEEPER climb. Goal: keep the
# no-collapse linear behaviour but raise its slope.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=3

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=0

METHOD="deyo_mlmp_gradslope_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

SLOPE_WINDOW=10
SLOPE_DEADZONE=0.002
LAG_GAIN=100000
MAX_LAG=300
BASE_RST=0.01
MONITOR_INTERVAL=50

SAVE_DIR="save/${DATASET}/${METHOD}_sw10_maxlag300_5corr_sub100/"

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
                        --subset_size 100 \
                        --subset_seed 0 \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --slope_window $SLOPE_WINDOW \
                        --slope_deadzone $SLOPE_DEADZONE \
                        --lag_gain $LAG_GAIN \
                        --max_lag $MAX_LAG \
                        --base_rst $BASE_RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
