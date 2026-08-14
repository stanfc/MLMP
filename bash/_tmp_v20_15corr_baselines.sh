#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

CUDA_VISIBLE_DEVICES=5 conda run -n mlmp python main_continual.py \
  --save_dir save/PascalVOC20Dataset/v20_15corr/no_adapt/ --data_dir .data/VOC2012/ \
  --prompt_dir prompts.yaml --dataset PascalVOC20Dataset --workers 1 \
  --init_resize 224 224 --patch_size 224 224 --patch_stride 112 \
  --corruptions_list $CONDS --class_extensions --split val \
  --subset_size 100 --subset_seed 0 --corruption_severity 5 \
  --ovss_type naclip --ovss_backbone ViT-L/14 --method deyo_mlmp_adagate_continual \
  --batch_size 1 --lr 5e-06 --steps 1 --continual_rounds 1 --seed 0 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 --monitor_interval 50 \
  --trend_stat mad --trend_thr 0.5 --trend_hist 50 --lag_mode ecdf --lag_sat 1.5 \
  --shallow_cap_mode fixed \
  > save/_sweep_logs/v20_15corr_no_adapt.log 2>&1

echo "[$(date +%H:%M)] no_adapt DONE, starting mlmp_episodic"

CUDA_VISIBLE_DEVICES=5 conda run -n mlmp python main.py \
  --save_dir save/PascalVOC20Dataset/v20_15corr/mlmp_episodic/ --data_dir .data/VOC2012/ \
  --prompt_dir prompts.yaml --dataset PascalVOC20Dataset --workers 1 \
  --subset_size 100 --subset_seed 0 \
  --init_resize 224 224 --patch_size 224 224 --patch_stride 112 \
  --corruptions_list $CONDS --ovss_type naclip --ovss_backbone ViT-L/14 \
  --class_extensions --split val --adapt --method mlmp \
  --batch_size 1 --lr 0.001 --steps 1 --trials 1 --seed 0 \
  --vision_outputs -1 --prompt_integration loss --alpha_cls 1.0 \
  > save/_sweep_logs/v20_15corr_mlmp_episodic.log 2>&1

echo "[$(date +%H:%M)] mlmp_episodic DONE"
