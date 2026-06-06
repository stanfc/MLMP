#!/bin/bash
# deyo_mlmp_smooth_anchor_continual on ACDCDataset: DeYO + MLMP(multi-prompt/layer+UAML) + SmoothAnchor.
# lag(H)=150/(H-2.2), h_ceil=2.9 (no restore when healthy ~3.0),
# h_floor=2.2 (deep collapse -> source). gate H logged to gate_log.csv.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=1

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="deyo_mlmp_smooth_anchor_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

H_CEIL=2.9
H_FLOOR=2.2
LAG_SCALE=150
MAX_LAG=3000
RST=0.005
MONITOR_INTERVAL=50

SAVE_DIR="save/${DATASET}/${METHOD}/"

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
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --h_ceil $H_CEIL \
                        --h_floor $H_FLOOR \
                        --lag_scale $LAG_SCALE \
                        --max_lag $MAX_LAG \
                        --rst $RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
