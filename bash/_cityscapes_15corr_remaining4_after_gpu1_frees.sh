#!/bin/bash
# Waits for hard5corr_fullsize adagate (last heavy job in the GPU1 queue before it
# starts on cityscapes arms) to finish, i.e. its results file to appear, THEN
# launches the remaining 4 Cityscapes arms (ctrl, flagship, growing,
# growing_scaled) concurrently -- by then GPU1 has freed the ~53GB used by
# gdgpa+adagate and there is room for all 4 alongside the queue's own 2 arms.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
echo "[$(date +%H:%M)] waiting for hard5corr_sub7000_batch8 adagate (last queue step before cityscapes arms) to exit..."
# nogate/gdgpa/adagate for hard5corr_sub7000_batch8 are ~5-6GB each (not 26GB like the
# old fullsize version), so they don't need to be waited on individually -- just wait
# for the last one (adagate) to appear and finish, which means the whole chain is done.
until pgrep -f "main_cls_continual.py.*hard5corr_sub7000_batch8_adagate" > /dev/null; do sleep 15; done
until ! pgrep -f "main_cls_continual.py.*hard5corr_sub7000_batch8_adagate" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] GPU1 should have room now -- launching remaining 4 Cityscapes arms"
GPU=1 ROUNDS=150 bash bash/sweep_adagate.sh cityscapes_15corr "ctrl ABmad05_ecdf ABmad05_ecdf_grow ABmad05_ecdf_grow_scaled"
echo "[$(date +%H:%M)] remaining 4 Cityscapes arms DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "Cityscapes 15corr 剩下 4 個 arm 也跑完了" "EXP-CS15b" 2>/dev/null || true
