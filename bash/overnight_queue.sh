#!/bin/bash
# ============================================================================
# Overnight queue: keep the free GPUs busy after the first six backbone arms.
# ============================================================================
# It first WAITS for bash/backbone_launch_all.sh to finish (it does not kill or
# interfere with anything), then runs four sequential lanes, one per free GPU.
#
#   GPU 0   clearclip_B16_gate    -> flagship seed 1
#   GPU 3   clearclip_B16_nogate  -> flagship seed 2
#   GPU 5   sclip_B16_uaml1_gate  -> sclip_B16_uaml1_nogate
#
# GPU 3 is now free (the all-LayerNorm ablation finished at R150 = 31.94).
# GPUs 1, 2 and 4 are left alone -- other users hold them.
#
# Why these four jobs:
#   ClearCLIP@ViT-B/16 completes the {NA-CLIP, ClearCLIP} x {ViT-L/14, ViT-B/16}
#     factorial, so formulation and scale can be separated instead of confounded.
#   Seeds 1 and 2 attack the standing "everything is seed=0" caveat on the
#     headline number (31.92), which is the first thing a reviewer asks.
#   SCLIP at UAML depth 1 is the appendix arm -- the only way SCLIP can be run
#     at all -- and is deliberately last, since it changes the method.
#
# Figures are regenerated after every completed job, and a watchdog refreshes
# them every 20 min, so whatever is on disk at wake-up is already plotted.
#
# usage: nohup bash bash/overnight_queue.sh > save/_backbone_logs/overnight.out 2>&1 &
# stop:  pkill -f overnight_queue.sh     (running python jobs keep going)
set -u
cd /home/tekai324/MLMP
PY=~/miniconda3/envs/MLMP/bin/python
LOGDIR=save/_backbone_logs
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/overnight_status.txt"

say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$STATUS"; }

replot() { $PY plot_backbone.py > "$LOGDIR/replot.log" 2>&1 || true; }

# ---- 1. wait for the running six-arm launcher ------------------------------
say "queue armed; waiting for backbone_launch_all.sh (skipped if not running)"
while pgrep -f "backbone_launch_all.sh" > /dev/null; do
  sleep 120
done
say "launcher finished; starting overnight lanes"
replot

# ---- 2. watchdog: keep figures fresh --------------------------------------
( while true; do sleep 1200; replot; done ) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

# ---- 3. lanes --------------------------------------------------------------
lane() {   # lane <gpu> <cmd...>
  local gpu="$1"; shift
  say "GPU$gpu -> $*"
  "$@" >> "$LOGDIR/overnight_gpu${gpu}.log" 2>&1
  say "GPU$gpu done: $*"
  replot
}

(
  lane 0 bash bash/backbone_ablation.sh clearclip_B16_gate 0 150
  lane 0 bash bash/seed_variance.sh 0 1 150 0.01
) &
(
  lane 3 bash bash/backbone_ablation.sh clearclip_B16_nogate 3 150
  lane 3 bash bash/seed_variance.sh 3 2 150 0.01
) &
(
  lane 5 bash bash/backbone_ablation.sh sclip_B16_uaml1_gate 5 150
  lane 5 bash bash/backbone_ablation.sh sclip_B16_uaml1_nogate 5 150
) &
wait

replot
say "OVERNIGHT QUEUE COMPLETE"
