#!/bin/bash
# SAR-DivGate-Continual on CityscapesDataset (CTTA, 150 rounds).
# SAR (SAM + reliable filter) base + DivGate's 3-tier stochastic restore
# replaces SAR's hard model recovery.
# 15 ImageNet-C corruptions applied on-the-fly.
# See docs/sar_divgate_continual_spec.md for the full design.

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
METHOD="sar_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── SAR (matches sar_continual.sh on Cityscapes / ACDC) ────────────
E_MARGIN=1.8          # sample-level mean-pixel entropy threshold
SAM_RHO=0.05          # SAM perturbation radius

# ── DivGate (TENT-DivGate ACDC best — see CLAUDE.md) ───────────────
H_THRESHOLD=1.6       # aggressive cutoff
H_WARNING=1.4         # cautious / brake cutoff
MONITOR_INTERVAL=50   # batches between H_margin checks
CAUTIOUS_RST=0.01     # mid restoration rate
BRAKE_RST=0.05        # strong restoration rate

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_weather/}"

# ───────────────────────────────────────────────────────────────────
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
                        --e_margin $E_MARGIN \
                        --sam_rho $SAM_RHO \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
