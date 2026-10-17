#!/bin/bash
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

SAVE1="save/${DATASET}/itaclip_deyo_mlmp/"
LOG1="save/_sweep_logs/itaclip_acdc_deyo_mlmp.log"
mkdir -p "$SAVE1"
CUDA_VISIBLE_DEVICES=4 python main_continual.py \
  --adapt --method deyo_mlmp_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION \
  --save_dir "$SAVE1" --class_extensions > "$LOG1" 2>&1 &
sleep 15

SAVE2="save/${DATASET}/itaclip_deyo_mlmp_adagate/"
LOG2="save/_sweep_logs/itaclip_acdc_deyo_mlmp_adagate.log"
mkdir -p "$SAVE2"
CUDA_VISIBLE_DEVICES=4 python main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
  --prompt_dir prompts.yaml \
  --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
  --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.01 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode growing_scaled \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 \
  --save_dir "$SAVE2" --class_extensions > "$LOG2" 2>&1 &
wait
echo "[$(date +%H:%M)] deyo_mlmp + deyo_mlmp_adagate (100-param retry) DONE"
