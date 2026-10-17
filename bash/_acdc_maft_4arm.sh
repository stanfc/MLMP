#!/bin/bash
# MAFT+ on ACDC: no_adapt / no-gate / +AdaGate / MLMP-episodic, matching the
# 4-column table (no-adapt, no gate, +AdaGate, MLMP-ep) already built for the
# other 16 backbones. UAML bypassed (vision_outputs=-1 only, decision 1A).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
CONDITIONS="fog night rain snow"
CKPT=".weights/maft/maftp_l.pth"

run() {
  local NAME=$1 GPU=$2; shift 2
  local SAVE="save/${DATASET}/maft_${NAME}/"
  local LOG="save/_sweep_logs/maft_acdc_${NAME}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  echo "[$(date +%H:%M)] START $NAME gpu=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU python "$@" --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $NAME (exit $?)"
}

# 1. no_adapt (1 round, batch-invariant)
run no_adapt 2 main_continual.py \
  --method tent_continual --ovss_type maft --ovss_backbone ViT-L/14 --maft_checkpoint "$CKPT" \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --batch_size 1 --continual_rounds 1 --seed 0 &
sleep 20

# 2. MLMP episodic
run mlmp_episodic 2 main.py \
  --adapt --method mlmp --ovss_type maft --ovss_backbone ViT-L/14 --maft_checkpoint "$CKPT" \
  --vision_outputs -1 --prompt_dir prompts.yaml --alpha_cls 1.0 \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.001 --steps 1 --batch-size 1 --trials 1 --seed 0 &
sleep 20

# 3. no-gate (deyo_mlmp_adagate_continual, base_rst=0.0)
run nogate 0 main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual --ovss_type maft --ovss_backbone ViT-L/14 --maft_checkpoint "$CKPT" \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs -1 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.0 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode growing_scaled \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 &
sleep 20

# 4. +AdaGate (base_rst=0.01, full gate)
run adagate 1 main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual --ovss_type maft --ovss_backbone ViT-L/14 --maft_checkpoint "$CKPT" \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs -1 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.01 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode growing_scaled \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 &
wait
echo "[$(date +%H:%M)] MAFT+ ACDC 4-arm ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "MAFT+ ACDC 4組(no_adapt/no-gate/AdaGate/MLMP-ep)全部完成" "EXP-MAFT-ACDC-4ARM" 2>/dev/null || true
