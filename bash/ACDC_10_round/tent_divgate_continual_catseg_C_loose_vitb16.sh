#!/bin/bash
# CatSeg DivGate sweep — C_loose on GPU 2.
# H_THR=1.6, H_WARN=1.4, CAU=0.003, BRAKE=0.02.

GPU_ID=2

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1152 768"
CONDITIONS="fog night rain snow"
WORKERS=4

METHOD="tent_divgate_continual_catseg"
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-B/16"
CATSEG_CKPT=".weights/catseg/model_base.pth"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_THRESHOLD=1.6
H_WARNING=1.4
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.003
BRAKE_RST=0.02

CONTINUAL_ROUNDS=150
TAG="C_loose_h${H_THRESHOLD}_${H_WARNING}_cau${CAUTIOUS_RST}_brake${BRAKE_RST}"
SAVE_DIR="save/${DATASET}/${METHOD}_${TAG}_vitb16/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
    --adapt \
    --method $METHOD \
    --ovss_type $OVSS_TYPE \
    --ovss_backbone "$OVSS_BACKBONE" \
    --catseg_checkpoint $CATSEG_CKPT \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --init_resize $INIT_RESIZE \
    --patch_size 384 384 \
    --patch_stride 192 \
    --corruptions_list $CONDITIONS \
    --workers $WORKERS \
    --lr $LR \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --continual_rounds $CONTINUAL_ROUNDS \
    --seed 0 \
    --h_threshold $H_THRESHOLD \
    --h_warning $H_WARNING \
    --monitor_interval $MONITOR_INTERVAL \
    --cautious_rst $CAUTIOUS_RST \
    --brake_rst $BRAKE_RST \
    --save_dir $SAVE_DIR \
    --class_extensions
