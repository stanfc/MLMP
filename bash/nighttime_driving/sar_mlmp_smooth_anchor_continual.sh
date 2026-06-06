#!/bin/bash
# SAR-MLMP-SmoothAnchor on Nighttime Driving (CTTA, 1200 rounds, native night shift).
# SAR (SAM + reliable filter) + full MLMP (multi-prompt/layer + UAML eval)
# + smooth-anchor restoration. See docs/sar_mlmp_smooth_anchor_continual_spec.md.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=${GPU_ID:-3}

# ── Dataset ────────────────────────────────────────────────────────
DATASET=NighttimeDrivingDataset
DATA_DIR="data/NighttimeDrivingTest/"
INIT_RESIZE="1120 560"
CONDITIONS="night"
WORKERS=1

# ── Method ─────────────────────────────────────────────────────────
METHOD="sar_mlmp_smooth_anchor_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

# ── Training hyperparameters (MLMP family LR) ──────────────────────
BATCH_SIZE=1
LR=${LR:-0.000005}
STEPS=1

# ── MLMP / UAML ────────────────────────────────────────────────────
ALPHA_CLS=${ALPHA_CLS:-0.0}
UAML_IN_ADAPT=${UAML_IN_ADAPT:-1}

# ── SAR ────────────────────────────────────────────────────────────
E_MARGIN=${E_MARGIN:-1.8}
SAM_RHO=${SAM_RHO:-0.05}

# ── SmoothAnchor (adapt-time ensemble H ~3.0 healthy) ──────────────
H_CEIL=${H_CEIL:-2.9}
H_FLOOR=${H_FLOOR:-2.2}
LAG_SCALE=${LAG_SCALE:-150.0}
MAX_LAG=${MAX_LAG:-3000}
RST=${RST:-0.005}
MONITOR_INTERVAL=50

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=1200
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
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
                        --alpha_cls $ALPHA_CLS \
                        --uaml_in_adapt $UAML_IN_ADAPT \
                        --e_margin $E_MARGIN \
                        --sam_rho $SAM_RHO \
                        --h_ceil $H_CEIL \
                        --h_floor $H_FLOOR \
                        --lag_scale $LAG_SCALE \
                        --max_lag $MAX_LAG \
                        --rst $RST \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
