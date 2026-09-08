#!/bin/bash
# Launch the six 150-round backbone-ablation arms.
# GPU 1 and 3 are deliberately avoided: 3 runs the adagate_allln ablation,
# 1 holds another user's 50 GB allocation.
# Estimated per-round cost measured on a 2-round smoke:
#   ViT-L/14 108 s   ViT-B/16 40 s   ViT-B/32 35 s   -> longest arm ~4.5 h.
cd /home/tekai324/MLMP
R=${1:-150}
bash bash/backbone_ablation.sh clearclip_L14_gate   4 $R &
bash bash/backbone_ablation.sh clearclip_L14_nogate 5 $R &
( bash bash/backbone_ablation.sh naclip_B16_gate    0 $R
  bash bash/backbone_ablation.sh naclip_B32_gate    0 $R ) &
( bash bash/backbone_ablation.sh naclip_B16_nogate  2 $R
  bash bash/backbone_ablation.sh naclip_B32_nogate  2 $R ) &
wait
echo "[$(date '+%m-%d %H:%M')] ALL BACKBONE ARMS DONE"
