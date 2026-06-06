#!/bin/bash
# CLIPArTT-DivGate-Continual on Cityscapes-C, 150 rounds, 15 corruptions.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=1

METHOD="clipartt_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CLIPARTT_K=3

H_THRESHOLD=1.8
H_WARNING=1.5
CAUTIOUS_RST=0.005
BRAKE_RST=0.02
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_K${CLIPARTT_K}/"
GATE_LOG="${SAVE_DIR}gate_log.csv"

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
                        --corruptions_list $ALL_CORRUPTIONS \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --clipartt_k $CLIPARTT_K \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        --gate_log_path $GATE_LOG \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
