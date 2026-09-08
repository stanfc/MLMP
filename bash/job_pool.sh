#!/bin/bash
# ============================================================================
# Generic GPU job pool: N workers, each bound to one GPU, pulling from a shared
# job file until it is empty.
# ============================================================================
# Each worker waits until ITS gpu is actually free (used memory below MINFREE)
# before taking a job, so this can be started while other work is still running
# and it will fill each GPU the moment that GPU frees -- instead of idling until
# every lane of a previous batch has finished.
#
# The job file holds one shell command per line; the literal token {GPU} is
# replaced with the worker's gpu id. Blank lines and #comments are skipped.
# Jobs are popped under flock, so no two workers can take the same line.
#
# usage: nohup bash bash/job_pool.sh <jobfile> <gpu> [gpu...] &
set -u
cd /home/tekai324/MLMP
JOBS="${1:?job file}"; shift
GPUS="$*"
LOGDIR=save/_backbone_logs; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/pool_status.txt"
LOCK="$JOBS.lock"; : > "$LOCK"; : > "$LOCK.adm"
# Admission is by FREE VRAM, not by idleness. These cards are 97.8 GB and a
# ViT-L/14 run at batch 1 peaks around 16 GB, so a card already running someone
# else's job still has room for several of ours. NEED_MIB is the free-memory
# floor a worker requires before it launches; it carries a deliberate margin over
# the largest arm. Workers also stagger their launches (SETTLE) so two of them
# cannot both see the same free memory and start at once.
#
# This DOES contend for compute with whatever else is on the card. It is the
# right trade when the alternative is leaving 80 GB idle, but it is a choice.
NEED_MIB=${NEED_MIB:-22000}
SETTLE=${SETTLE:-90}

say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$STATUS"; }

pop() {   # prints the next job line, or returns 1 when the file is empty
  flock 9
  local line
  line=$(grep -vE '^\s*(#|$)' "$JOBS" | head -1)
  [ -z "$line" ] && return 1
  grep -vxF "$line" "$JOBS" > "$JOBS.tmp" && mv "$JOBS.tmp" "$JOBS"
  printf '%s\n' "$line"
} 9>"$LOCK"

worker() {
  local gpu="$1" job
  while job=$(pop); do
    while :; do
      free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "${free:-0}" -ge "$NEED_MIB" ]; then
        # claim the slot under the same lock the job file uses, then let the new
        # process allocate before any other worker re-reads memory.free
        flock 8
        free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        if [ "${free:-0}" -ge "$NEED_MIB" ]; then
          flock -u 8
          break
        fi
        flock -u 8
      fi
      sleep 60
    done
    job="${job//\{GPU\}/$gpu}"
    say "GPU$gpu start: $job"
    ( eval "$job" >> "$LOGDIR/pool_gpu${gpu}.log" 2>&1 ) &
    local pid=$!
    sleep "$SETTLE"   # let it allocate before another worker samples memory.free
    wait $pid
    say "GPU$gpu done : $job"
    ~/miniconda3/envs/MLMP/bin/python plot_backbone.py > "$LOGDIR/replot.log" 2>&1 || true
  done
  say "GPU$gpu: job file empty, worker exiting"
} 8>"$LOCK.adm"

echo $$ > "$LOGDIR/pool.pid"
say "pool started on GPUs [$GPUS] x${WORKERS_PER_GPU:-2} slots, NEED_MIB=${NEED_MIB}, $(grep -cvE '^\s*(#|$)' "$JOBS") jobs"
# WORKERS_PER_GPU slots per card; free-VRAM admission throttles them further
WPG=${WORKERS_PER_GPU:-2}
for g in $GPUS; do
  for _ in $(seq 1 "$WPG"); do worker "$g" & sleep 3; done
done
wait
rm -f "$LOGDIR/pool.pid"
say "POOL COMPLETE"
