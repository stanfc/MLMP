#!/bin/bash
# naclip counterpart to bash/_acdc_itaclip_restart_after_reset.sh -- identical
# 5-method roster/hyperparams/100-param-LN setup, only ovss_type differs, for
# a clean naclip-vs-itaclip backbone comparison.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
CONDITIONS="fog night rain snow"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

run() {
  local NAME=$1 GPU=$2; shift 2
  local SAVE="save/${DATASET}/naclip_${NAME}/"
  local LOG="save/_sweep_logs/naclip_acdc_${NAME}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  CUDA_VISIBLE_DEVICES=$GPU python "$@" --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1 &
}

run no_adapt 3 main_continual.py \
  --method tent_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --batch_size 1 --continual_rounds 1 --seed 0
sleep 15

run mlmp_episodic 3 main.py \
  --adapt --method mlmp --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --vision_outputs $OUT_VISION --prompt_dir prompts.yaml --alpha_cls 1.0 \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.001 --steps 1 --batch-size 1 --trials 1 --seed 0
sleep 15

run tent_continual 3 main_continual.py \
  --adapt --method tent_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.00001 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0
sleep 15

run deyo_mlmp 4 main_continual.py \
  --adapt --method deyo_mlmp_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION
sleep 15

run deyo_mlmp_adagate 4 main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.01 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode growing_scaled \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5
wait
echo "[$(date +%H:%M)] naclip ACDC 5-method (backbone-compare) DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "ACDC naclip 5個方法(backbone比較用)完成" "EXP-ACDC-NACLIP-COMPARE" 2>/dev/null || true
