#!/bin/bash
# GPU job QUEUE for the divreg sweep. Each tick it queries live free VRAM per GPU
# and greedily places as many pending (dataset,lambda) jobs as fit (local
# reservation accounting so we don't double-book before nvidia-smi catches up).
# Jobs that don't fit wait; the queue keeps filling GPUs as room frees up (e.g.
# when the user's composite/hmgate2 jobs finish). Placement is gated by MEMORY
# (need + BUFFER) so it will NOT OOM-crash existing jobs.
#
# Each job is launched via sweep_divreg.sh (SEQ=1, pinned GPU) so all per-dataset
# config (paths/resize/corruptions/subset) stays DRY in one place.
#
# Run in background:  nohup bash bash/run_divreg_queue.sh > save/_sweep_logs/queue_main.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"
POLL="${POLL:-60}"          # seconds to wait when nothing fit this tick
SETTLE="${SETTLE:-90}"      # seconds after a placement round (let new jobs grab VRAM)
BUFFER="${BUFFER:-3000}"    # MiB headroom required beyond each job's need
GPUS="${GPUS:-0 1 2 3}"

mkdir -p save/_sweep_logs

# "dataset:lambda:mem_MiB"  (full-res ACDC/Citys ~13GB, VOC20 224 ~6GB)
JOBS="acdc:0.0:14000
acdc:0.1:14000
acdc:0.3:14000
acdc:1.0:14000
acdc:3.0:14000
cityscapes:0.0:14000
cityscapes:0.1:14000
cityscapes:0.3:14000
cityscapes:1.0:14000
cityscapes:3.0:14000
v20:0.0:7000
v20:0.1:7000
v20:0.3:7000
v20:1.0:7000
v20:3.0:7000"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

pending="$JOBS"
echo "[queue] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending" | grep -c .) jobs, GPUs: $GPUS, rounds=$ROUNDS"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -a RES FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); done
  progress=0
  remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; rest="${job#*:}"; lam="${rest%%:*}"; need="${rest##*:}"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[queue] $(date '+%H:%M') PLACE $ds lam=$lam (need ${need}+${BUFFER}) -> GPU$g (free ${FREE[$g]}, reserved ${RES[$g]})"
        GPU=$g SEQ=1 ROUNDS=$ROUNDS bash bash/sweep_divreg.sh "$ds" "$lam" \
          > "save/_sweep_logs/queue_${ds}_lam${lam}.log" 2>&1 &
        RES[$g]=$(( ${RES[$g]} + need ))
        placed=1; progress=1
        break
      fi
    done
    [ "$placed" -eq 0 ] && remaining+="${job}"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  nleft=$(printf '%s\n' "$pending" | grep -c .)
  if [ "$progress" -eq 1 ]; then
    echo "[queue] $(date '+%H:%M') placed some; $nleft left; settling ${SETTLE}s"
    [ "$nleft" -gt 0 ] && sleep "$SETTLE"
  else
    echo "[queue] $(date '+%H:%M') nothing fit; $nleft left; waiting ${POLL}s"
    sleep "$POLL"
  fi
done

echo "[queue] $(date '+%H:%M') all jobs launched; waiting for training to finish..."
wait
echo "[queue] $(date '+%m-%d %H:%M') ALL DIVREG JOBS COMPLETE."
