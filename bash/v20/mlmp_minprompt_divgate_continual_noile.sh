#!/bin/bash
# mlmp_minprompt_divgate_continual on PascalVOC20Dataset 5corr sub100

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=1
SUBSET_SIZE=100
SUBSET_SEED=0

METHOD="mlmp_minprompt_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=0.0

BATCH_SIZE=1
LR=0.000005
STEPS=1

H_THRESHOLD=1.6
H_WARNING=0.8
CAUTIOUS_RST=0.005
BRAKE_RST=0.02
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_noile_5corr_sub100/"

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
                        --subset_size $SUBSET_SIZE \
                        --subset_seed $SUBSET_SEED \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --alpha_cls $ALPHA_CLS \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
