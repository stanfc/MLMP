#!/bin/bash
# One-command morning check for the backbone campaign.
#   bash bash/backbone_status.sh
cd /home/tekai324/MLMP
echo "=============================================================="
echo " backbone campaign  --  $(date '+%Y-%m-%d %H:%M')"
echo "=============================================================="
echo
echo "-- queue log ------------------------------------------------"
tail -14 save/_backbone_logs/overnight_status.txt 2>/dev/null || echo "  (queue has not started its lanes yet)"
echo
echo "-- arms (R<n> of 150, last Mean_mIoU) -----------------------"
for d in save/ACDCDataset/backbone_ablation/*/ save/ACDCDataset/seed_variance/*/; do
  [ -d "$d" ] || continue
  f="$d/results_all_rounds.txt"
  n=$(grep -c '^Round' "$f" 2>/dev/null || echo 0)
  last=$(tail -1 "$f" 2>/dev/null | awk -F',' '{print $NF}')
  printf "  %-26s R%-4s %s\n" "$(basename $d)" "$n" "${last:- --}"
done
echo
echo "-- scheduler ------------------------------------------------"
if [ -f save/_backbone_logs/pool.pid ] && kill -0 "$(cat save/_backbone_logs/pool.pid)" 2>/dev/null; then
  echo "  job_pool ALIVE (pid $(cat save/_backbone_logs/pool.pid))"
else
  echo "  job_pool NOT RUNNING"
fi
echo
echo "-- live processes -------------------------------------------"
ps -eo etime,cmd | grep main_continual | grep -v grep \
  | sed -E 's#^ *([^ ]+).*--save_dir ([^ ]*).*#  \1  \2#' | sort -k2
echo
echo "-- gpu ------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo
echo "-- figures --------------------------------------------------"
ls -l --time-style=+%m-%d_%H:%M figures/backbone/*.png 2>/dev/null | awk '{printf "  %s  %s\n", $6, $7}'
echo
echo "regenerate figures + tables:  python plot_backbone.py"
echo "stop the queue (running jobs continue):  pkill -f overnight_queue.sh"
