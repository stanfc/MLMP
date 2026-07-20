#!/bin/bash
# SEED VALIDATION for the entropy-weighted-prompt result.
# Question: is the +0.2-0.3 mIoU gain real, or seed noise?
#
# Minimal sufficient design — per dataset, ONLY the best config vs its control,
# at 2 additional seeds (seed 0 already exists) -> 3 seeds per cell:
#   ACDC        adapt b0.5   vs  adapt b0.0 (= GDG-PA control)
#   Cityscapes  eval  b1.0   vs  adapt b0.0
#   VOC20       adapt b0.5   vs  adapt b0.0   (tests the "does not hurt" claim)
# 3 datasets x 2 configs x 2 seeds = 12 runs.
#
# The image subset is pinned (--subset_seed 0) so seeds vary ONLY algorithmic
# stochasticity -> a PAIRED comparison against the same data.
#
# Pinned to GPU3 by default (GPUS env overrides). Mem-gated: fills GPU3 and queues
# the rest as room frees.
#
# Run:  nohup bash bash/run_seedval_queue.sh > save/_sweep_logs/seedval_queue_main.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"; POLL="${POLL:-60}"; SETTLE="${SETTLE:-90}"
BUFFER="${BUFFER:-3000}"; GPUS="${GPUS:-3}"
SEEDS="${SEEDS:-1 2}"
mkdir -p save/_sweep_logs

# "dataset:side:beta:mem"  — best config and control per dataset
CELLS="acdc:adapt:0.5:16000
acdc:adapt:0.0:16000
cityscapes:eval:1.0:16000
cityscapes:adapt:0.0:16000
v20:adapt:0.5:9000
v20:adapt:0.0:9000"

JOBS=""
for s in $SEEDS; do
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    JOBS+="${c}:${s}"$'\n'
  done <<< "$CELLS"
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
echo "[sv] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending" | grep -c .) jobs, GPUs: $GPUS, seeds: $SEEDS, rounds=$ROUNDS"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -a RES FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    IFS=':' read -r ds side beta need seed <<< "$job"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[sv] $(date '+%H:%M') PLACE $ds $side b=$beta seed=$seed -> GPU$g (free ${FREE[$g]}, res ${RES[$g]})"
        GPU=$g SEQ=1 SEED=$seed ROUNDS=$ROUNDS bash bash/sweep_promptw.sh "$ds" "$side" "$beta" \
          > "save/_sweep_logs/sv_${ds}_${side}_b${beta}_s${seed}.log" 2>&1 &
        RES[$g]=$(( ${RES[$g]} + need )); placed=1; progress=1; break
      fi
    done
    [ "$placed" -eq 0 ] && remaining+="${job}"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  nleft=$(printf '%s\n' "$pending" | grep -c .)
  if [ "$progress" -eq 1 ]; then
    echo "[sv] $(date '+%H:%M') placed some; $nleft left; settling ${SETTLE}s"
    [ "$nleft" -gt 0 ] && sleep "$SETTLE"
  else
    echo "[sv] $(date '+%H:%M') nothing fit; $nleft left; waiting ${POLL}s"; sleep "$POLL"
  fi
done
echo "[sv] $(date '+%H:%M') all jobs launched; waiting..."
wait
echo "[sv] $(date '+%m-%d %H:%M') ALL SEED-VALIDATION JOBS COMPLETE."
