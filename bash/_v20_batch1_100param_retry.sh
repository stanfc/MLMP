#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

# gated (red line, "ours")
SAVE1="save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch1_clean/"
LOG1="save/_sweep_logs/v20_batch1_gated_100param.log"
mkdir -p "$SAVE1"
CUDA_VISIBLE_DEVICES=3 python main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
  --ann_file "$ANN_FILE" --subset_seed 0 \
  --workers 1 --lr 5e-06 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 --max_windows 2000 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --monitor_interval 50 \
  --trend_stat mad --trend_thr 0.5 --trend_hist 50 --lag_mode ecdf --lag_sat 1.5 \
  --shallow_cap_mode growing_scaled --h_max_mode lifetime --h_max_decay 0.98 \
  --save_dir "$SAVE1" --class_extensions > "$LOG1" 2>&1 &
sleep 15

# no-gate (blue line, base_rst=0.0)
SAVE2="save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch1_fulldata/"
LOG2="save/_sweep_logs/v20_batch1_nogate_100param.log"
mkdir -p "$SAVE2"
CUDA_VISIBLE_DEVICES=3 python main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
  --ann_file "$ANN_FILE" --subset_seed 0 \
  --workers 1 --lr 5e-06 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs $OUT_VISION \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.0 --max_windows 2000 \
  --h_drop_ratio 0.9 --maxlag_shallow 6 --monitor_interval 5 \
  --trend_stat mad --trend_thr 0.5 --trend_hist 50 --lag_mode ecdf --lag_sat 1.5 \
  --shallow_cap_mode fixed \
  --save_dir "$SAVE2" --class_extensions > "$LOG2" 2>&1 &
wait
echo "[$(date +%H:%M)] V20 batch=1 gated+no-gate (100-param retry) DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "V20 batch=1 gated+no-gate 100參數版重跑完成" "EXP-V20-B1-100P" 2>/dev/null || true
