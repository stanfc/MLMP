#!/bin/bash
# Sweep run D1 (GPU 3): H_HIGH=1.6, H_LOW=1.0, MAX_RST=0.05.
# Auto-generated. Identical structure to tent_siggate_continual.sh except
# for the gate hyperparameters and SAVE_DIR.

GPU_ID=3

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

METHOD="tent_siggate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_HIGH=1.6
H_LOW=1.0
MAX_RST=0.05
SIGMOID_K=-1
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
TAG="D1_h${H_HIGH}_${H_LOW}_max${MAX_RST}"
SAVE_DIR="save/${DATASET}/sweep_sig_${TAG}/"
GATE_LOG="${SAVE_DIR}gate_log.csv"

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
    --h_high $H_HIGH \
    --h_low $H_LOW \
    --max_rst $MAX_RST \
    --sigmoid_k $SIGMOID_K \
    --monitor_interval $MONITOR_INTERVAL \
    --gate_log_path $GATE_LOG \
    --save_dir $SAVE_DIR \
    --class_extensions \
&& python plot_results.py \
    --runs $SAVE_DIR \
          "save/${DATASET}/tent_divgate_continual_step_1" \
    --labels "sweep_${TAG}" "tent_divgate (discrete winner)" \
    --out_dir "figures/sweep_sig_${TAG}" \
&& python plot_gate_entropy.py \
    --run $SAVE_DIR \
    --out "figures/sweep_sig_${TAG}/gate_entropy.png"
