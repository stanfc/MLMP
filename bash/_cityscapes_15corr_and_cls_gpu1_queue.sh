#!/bin/bash
# GPU1 queue: cityscapes baselines (quick) -> hard5corr_sub7000_batch8
# nogate+gdgpa+adagate (concurrent, ~6GB each, NOT fullsize) -> last 2 Cityscapes
# arms (concurrent, ~9GB each).
set -e
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

echo "[$(date +%H:%M)] cityscapes_15corr: no_adapt"
CUDA_VISIBLE_DEVICES=1 conda run -n mlmp python main_continual.py \
  --save_dir save/CityscapesDataset/cityscapes_15corr/no_adapt/ --data_dir .data/cityscapes/ \
  --prompt_dir prompts.yaml --dataset CityscapesDataset --workers 1 \
  --init_resize 1120 560 --patch_size 224 224 --patch_stride 112 \
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
  --shallow_cap_mode fixed --class_extensions \
  > save/_sweep_logs/cityscapes_15corr_no_adapt.log 2>&1

echo "[$(date +%H:%M)] cityscapes_15corr: mlmp_episodic"
CUDA_VISIBLE_DEVICES=1 conda run -n mlmp python main.py \
  --save_dir save/CityscapesDataset/cityscapes_15corr/mlmp_episodic/ --data_dir .data/cityscapes/ \
  --prompt_dir prompts.yaml --dataset CityscapesDataset --workers 1 \
  --subset_size 100 --subset_seed 0 \
  --init_resize 1120 560 --patch_size 224 224 --patch_stride 112 \
  --corruptions_list $CONDS --ovss_type naclip --ovss_backbone ViT-L/14 \
  --class_extensions --split val --adapt --method mlmp \
  --batch_size 1 --lr 0.001 --steps 1 --trials 1 --seed 0 \
  --vision_outputs -1 --prompt_integration loss --alpha_cls 1.0 \
  > save/_sweep_logs/cityscapes_15corr_mlmp_episodic.log 2>&1

echo "[$(date +%H:%M)] hard5corr_sub7000_batch8: nogate+gdgpa+adagate (block=875 steps/corr, same recipe as K3/K4/K7 -- NOT fullsize, no reason to pay for 50000 images when subset_size/batch_size already gives the same long block cheaply)"
cd /home/stanfc/TTA-on-OVSS/clsTTA
CORR="defocus_blur glass_blur gaussian_noise zoom_blur elastic_transform"

CUDA_VISIBLE_DEVICES=1 python main_cls_continual.py \
  --arch clip:ViT-L/14 --method gdgpa \
  --data_root /home/stanfc/TTA-on-OVSS/MLMP/.data/ImageNet-C \
  --severity 5 --batch_size 8 --rounds 15 --corruptions $CORR \
  --subset_size 7000 --subset_seed 0 \
  --lr 0.0001 --base_rst 0.0 --monitor_interval 50 --shuffle --seed 0 \
  --save_dir save/clip_vitL14_hard5corr_sub7000_batch8_nogate_15R \
  > save/_launch_hard5corr_sub7000_batch8_nogate.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_cls_continual.py \
  --arch clip:ViT-L/14 --method gdgpa \
  --data_root /home/stanfc/TTA-on-OVSS/MLMP/.data/ImageNet-C \
  --severity 5 --batch_size 8 --rounds 15 --corruptions $CORR \
  --subset_size 7000 --subset_seed 0 \
  --lr 0.0001 --base_rst 0.01 --monitor_interval 50 --shuffle --seed 0 \
  --save_dir save/clip_vitL14_hard5corr_sub7000_batch8_gdgpa_15R \
  > save/_launch_hard5corr_sub7000_batch8_gdgpa.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_cls_continual.py \
  --arch clip:ViT-L/14 --method adagate \
  --data_root /home/stanfc/TTA-on-OVSS/MLMP/.data/ImageNet-C \
  --severity 5 --batch_size 8 --rounds 15 --corruptions $CORR \
  --subset_size 7000 --subset_seed 0 \
  --lr 0.0001 --base_rst 0.01 --monitor_interval 50 --shuffle --seed 0 \
  --trend_stat mad --trend_thr 0.5 --lag_mode ecdf --shallow_cap_mode growing_hmargin_scaled \
  --save_dir save/clip_vitL14_hard5corr_sub7000_batch8_adagate_15R \
  > save/_launch_hard5corr_sub7000_batch8_adagate.log 2>&1 &
wait

echo "[$(date +%H:%M)] cityscapes_15corr: last 2 arms (growing_hmargin, growing_hmscaled) -- GPU1 now has room"
cd /home/stanfc/TTA-on-OVSS/MLMP
GPU=1 ROUNDS=150 bash bash/sweep_adagate.sh cityscapes_15corr "ABmad05_ecdf_grow_hmargin ABmad05_ecdf_grow_hmscaled"

echo "[$(date +%H:%M)] GPU1 queue ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "GPU1 queue(cityscapes baseline+2arm, hard5corr gdgpa+adagate)全部完成" "EXP-Q1" 2>/dev/null || true
