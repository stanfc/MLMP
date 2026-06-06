#!/bin/bash
# P2: same shape as A1 (DivGate winner), but MAX_RST pushed to 0.10.
# Question: with the winner's mid (1.2) and width (0.8), is doubling the
# brake cap from 0.05 -> 0.10 still beneficial, or does it over-restore?

GPU_ID=1

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="tent_siggate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_HIGH=1.6
H_LOW=0.8
MAX_RST=0.10
SIGMOID_K=-1
MONITOR_INTERVAL=50

CONTINUAL_ROUNDS=150
TAG="h${H_HIGH}_${H_LOW}_max${MAX_RST}"
SAVE_DIR="save/${DATASET}/tent_siggate_${TAG}/"
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
          "save/${DATASET}/sweep_sig_A1_h1.6_0.8_max0.05" \
          "save/${DATASET}/tent_divgate_continual_step_1" \
    --labels "siggate_${TAG}" "A1 (max=0.05)" "tent_divgate (discrete)" \
    --out_dir "figures/tent_siggate_${TAG}" \
&& python plot_gate_entropy.py \
    --run $SAVE_DIR \
    --out "figures/tent_siggate_${TAG}/gate_entropy.png"
