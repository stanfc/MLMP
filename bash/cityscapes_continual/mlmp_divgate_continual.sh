#!/bin/bash
# MLMP-DivGate-Continual on CityscapesDataset (CTTA, N rounds).
# MLMP multi-prompt multi-level entropy loss + diversity-gated stochastic restoration.
# Designed for the Cityscapes weather-subset where TENT-DivGate caps below source
# (no headroom): MLMP-episodic peaks at 22.60 mIoU, so its base loss has more
# signal to exploit than pure TENT — combined with the gate, may exceed 22.60
# under continual conditions.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-2}

# ── Dataset ────────────────────────────────────────────────────────
DATASET=CityscapesDataset
DATA_DIR="data/Cityscape/"
INIT_RESIZE="1120 560"
WORKERS=1

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
CORRUPTIONS_ARRAY=(
    # ACDC-matched 4 corruptions: snow<->snow, fog<->fog, frost<->rain, contrast<->night
    snow
    frost
    fog
    contrast
)
# One-liner subset override: CORRUPTIONS_LIST="fog snow frost brightness" bash script.sh
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="mlmp_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── MLMP multi-level: last 18 layers of ViT-L/14 ──────────────────
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
PROMPT_INTEGRATION="loss"
ALPHA_CLS=1.0

# ── Training hyperparameters (match mlmp_continual) ────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Diversity gate (best confirmed hyperparameters from ACDC) ──────
# Note: Cityscapes H_margin runs higher than ACDC (~2.0 vs ~1.79), so the
# gate may fire less often at h_thr=1.6. If you find the gate stays in
# aggressive mode the entire run, raise H_THRESHOLD to ~2.0.
H_THRESHOLD=1.6
H_WARNING=1.4
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_weather_threshold_${H_THRESHOLD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --prompt_integration $PROMPT_INTEGRATION \
                        --alpha_cls $ALPHA_CLS \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        --subset_size 101 --subset_seed 0 \
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
