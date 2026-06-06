#!/bin/bash
# TENT-DivGate-Continual on ACDC with CAT-Seg backbone (CTTA, 150 rounds).
#
# Same gate hyperparameters as the NA-CLIP version on this machine
# (h_thr=1.8, h_warn=1.5, cau=0.005, brake=0.02), but the underlying
# OVSS model is CAT-Seg ViT-L/14 (patch_size 384, stride 192).
#
# Both visual encoder LayerNorms AND sem_seg_head LayerNorms are trained.
# Text encoder + token_embedding stay frozen.

GPU_ID=0

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1152 768"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="tent_divgate_continual_catseg"
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/14"
CATSEG_CKPT=".weights/catseg/model_large.pth"

BATCH_SIZE=1
LR=0.00001
STEPS=1

# Diversity gate (default: same as NA-CLIP DivGate baseline on this machine)
H_THRESHOLD=1.8
H_WARNING=1.5
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.005
BRAKE_RST=0.02

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_caut_${CAUTIOUS_RST}_brake_${BRAKE_RST}/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone "$OVSS_BACKBONE" \
                        --catseg_checkpoint $CATSEG_CKPT \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 384 384 \
                        --patch_stride 192 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
