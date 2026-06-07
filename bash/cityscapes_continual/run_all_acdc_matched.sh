#!/bin/bash
# Sequentially run ALL cityscapes_continual ACDC-matched experiments on GPU 2.
# One method at a time (heavy jobs would thrash a shared GPU). Per-method stdout
# goes to save/CityscapesDataset/_run_logs/<method>.log; progress + exit codes to
# _progress.log. Ordered lightest -> heaviest so quick baselines land first.
#
# Usage:  bash bash/cityscapes_continual/run_all_acdc_matched.sh

cd "$(dirname "$0")/../.." || exit 1   # repo root
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MLMP
export GPU_ID=2

LOGDIR=save/CityscapesDataset/_run_logs
mkdir -p "$LOGDIR"

METHODS=(
    no_adapt
    mlmp_episodic
    tent_continual
    cotta
    sar_continual
    tent_divgate_continual
    tent_divgate_smooth_anchor
    sar_divgate_continual
    mlmp_continual
    mlmp_divgate_continual
    sar_mlmp_smooth_anchor_continual
)

echo "===== [$(date '+%F %T')] QUEUE START (${#METHODS[@]} methods, GPU $GPU_ID) =====" | tee -a "$LOGDIR/_progress.log"
for m in "${METHODS[@]}"; do
    echo "----- [$(date '+%F %T')] START $m -----" | tee -a "$LOGDIR/_progress.log"
    bash "bash/cityscapes_continual/${m}.sh" > "$LOGDIR/${m}.log" 2>&1
    rc=$?
    echo "----- [$(date '+%F %T')] DONE  $m (exit $rc) -----" | tee -a "$LOGDIR/_progress.log"
done
echo "===== [$(date '+%F %T')] QUEUE COMPLETE =====" | tee -a "$LOGDIR/_progress.log"
