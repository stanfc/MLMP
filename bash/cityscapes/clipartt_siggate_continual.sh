#!/bin/bash
# CLIPArTT-SigGate-Continual on Cityscapes-C, 150 rounds, 15 corruptions.
# CLIPArTT self-distillation + sigmoid-shaped diversity-gated stochastic
# restoration.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=1

METHOD="clipartt_siggate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CLIPARTT_K=3

# Sigmoid diversity gate (same defaults as tent_siggate_continual)
H_HIGH=1.8
H_LOW=0.8
MAX_RST=0.01
SIGMOID_K=-1
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_K${CLIPARTT_K}_max_${MAX_RST}/"
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
                        --h_high $H_HIGH \
                        --h_low $H_LOW \
                        --max_rst $MAX_RST \
                        --sigmoid_k $SIGMOID_K \
                        --monitor_interval $MONITOR_INTERVAL \
                        --gate_log_path $GATE_LOG \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
