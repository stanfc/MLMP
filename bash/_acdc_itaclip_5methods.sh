#!/bin/bash
# ACDC, ovss_type=itaclip (ported ITACLIP attention: SCLIP-style last layer,
# blended 50/50 with intermediate-layer average -- see ovss/clip/model.py).
# 5 methods, per the modern 150R/LR=5e-6/STEPS=1 flagship convention (except
# no_adapt=continual_rounds irrelevant, MLMP=episodic default LR=0.001).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
OVSS_TYPE="itaclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

run() {
  local NAME=$1 GPU=$2; shift 2
  local SAVE="save/${DATASET}/itaclip_${NAME}/"
  local LOG="save/_sweep_logs/itaclip_acdc_${NAME}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  echo "[$(date +%H:%M)] START $NAME gpu=$GPU -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU python "$@" --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $NAME (exit $?)"
}

# 1. no_adapt (episodic-equivalent: tent_continual without --adapt, 1 round)
run no_adapt 3 main_continual.py \
  --method tent_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize $INIT_RESIZE \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --batch_size 1 --continual_rounds 1 --seed 0 &
sleep 15

# 2. MLMP (episodic)
run mlmp_episodic 3 main.py \
  --adapt --method mlmp --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --vision_outputs $OUT_VISION --prompt_dir prompts.yaml --alpha_cls 1.0 \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize $INIT_RESIZE \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.001 --steps 1 --batch-size 1 --trials 1 --seed 0 &
sleep 15

# 3. tent (continual)
run tent_continual 3 main_continual.py \
  --adapt --method tent_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize $INIT_RESIZE \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.00001 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 &
sleep 15

# 4. DeYO+MLMP (no gate)
run deyo_mlmp 4 main_continual.py \
  --adapt --method deyo_mlmp_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize $INIT_RESIZE \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION &
sleep 15

# 5. DeYO+MLMP+AdaGate (current confirmed method, ABmad05_ecdf_grow_scaled)
run deyo_mlmp_adagate 4 main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize $INIT_RESIZE \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.01 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode growing_scaled \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 &

wait
echo "[$(date +%H:%M)] ACDC itaclip 5-method run COMPLETE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "ACDC itaclip backbone 5個方法跑完了" "EXP-ACDC-ITACLIP" 2>/dev/null || true
