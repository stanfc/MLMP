#!/bin/bash
# no-adapt + MLMP-episodic reference lines for the V20 batch-size ablation
# (trainval.txt, no subset cap, corruptions=snow/frost/fog/brightness/contrast
# -- same config as bash/_v20_batch_ablation_nogate.sh and
# bash/_v20_deyo_adagate_batch8_64.sh) so the batch=1/8/64 comparison figure
# has properly-matched flat reference lines instead of borrowed values from a
# different subset config. Both are batch-size-invariant (no_adapt: frozen
# model, no cross-sample state; episodic: reset() every sample), so one run
# each covers all three batch-size panels.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"

echo "[$(date +%H:%M)] START no_adapt"
CUDA_VISIBLE_DEVICES=3 python main_continual.py \
  --method tent_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 \
  --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
  --ann_file "$ANN_FILE" --workers 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --save_dir save/PascalVOC20Dataset/v20_batch_ablation/No_Adaptation/ --class_extensions \
  > save/_sweep_logs/v20_batch_ablation_no_adapt.log 2>&1 &
disown
sleep 10

echo "[$(date +%H:%M)] START mlmp_episodic"
CUDA_VISIBLE_DEVICES=3 python main.py \
  --adapt --method mlmp \
  --ovss_type naclip --ovss_backbone ViT-L/14 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --prompt_dir prompts.yaml --alpha_cls 1.0 \
  --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
  --ann_file "$ANN_FILE" --workers 1 --lr 0.001 --steps 1 --batch_size 1 --trials 1 --seed 0 \
  --save_dir save/PascalVOC20Dataset/v20_batch_ablation/mlmp_episodic/ --class_extensions \
  > save/_sweep_logs/v20_batch_ablation_mlmp_episodic.log 2>&1 &
disown
wait
echo "[$(date +%H:%M)] v20_batch_ablation_references ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "V20 batch ablation 參考線(no_adapt+episodic)跑完了" "EXP-V20-REF" 2>/dev/null || true
