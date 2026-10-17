#!/bin/bash
# Machine-wide reset wiped all processes (all 6 GPUs went to 0MiB, unrelated
# other users' jobs vanished too) -- restarting the 3 ACDC/itaclip continual
# jobs that had no usable checkpoint / had a corrupted results file.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
CONDITIONS="fog night rain snow"
OVSS_TYPE="itaclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

run() {
  local NAME=$1 GPU=$2; shift 2
  local SAVE="save/${DATASET}/itaclip_${NAME}/"
  local LOG="save/_sweep_logs/itaclip_acdc_${NAME}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  CUDA_VISIBLE_DEVICES=$GPU python "$@" --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1 &
}

run tent_continual 0 main_continual.py \
  --adapt --method tent_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.00001 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0
sleep 15

run deyo_mlmp 1 main_continual.py \
  --adapt --method deyo_mlmp_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION
sleep 15

run deyo_mlmp_adagate 2 main_continual.py \
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
echo "[$(date +%H:%M)] ACDC itaclip restart-after-reset DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "ACDC itaclip 3個method(斷線重跑)完成" "EXP-ACDC-ITACLIP-RESTART" 2>/dev/null || true
