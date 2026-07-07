#!/bin/bash
# Launch ALL cityscapes_continual ACDC-matched jobs IN PARALLEL across GPUs 0/1/2/3.
# Each job nohup-detached with its own GPU + log. Staggered to avoid a simultaneous
# model-load / disk spike. Assignment is balanced against the other experiments that
# were already running (smooth_sweep on 0/1, sar_mlmp on 3) — edit GPU map as needed.
#
# Usage:  bash bash/cityscapes_continual/run_all_parallel.sh

cd "$(dirname "$0")/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MLMP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

LOGDIR=save/CityscapesDataset/_run_logs
mkdir -p "$LOGDIR"

# method:gpu  (3 on GPU2/0/1, 2 on the more-loaded GPU3)
JOBS=(
    "sar_mlmp_smooth_anchor_continual:2"
    "mlmp_continual:2"
    "mlmp_divgate_continual:2"
    "mlmp_episodic:0"
    "tent_divgate_continual:0"
    "no_adapt:0"
    "sar_continual:1"
    "tent_divgate_smooth_anchor:1"
    "tent_continual:1"
    "cotta:3"
    "sar_divgate_continual:3"
)

for j in "${JOBS[@]}"; do
    m="${j%%:*}"; g="${j##*:}"
    echo "[$(date '+%F %T')] launch $m -> GPU $g" | tee -a "$LOGDIR/_parallel.log"
    GPU_ID=$g nohup bash "bash/cityscapes_continual/${m}.sh" > "$LOGDIR/${m}.log" 2>&1 &
    sleep 8
done
echo "[$(date '+%F %T')] all ${#JOBS[@]} jobs launched" | tee -a "$LOGDIR/_parallel.log"
