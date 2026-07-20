#!/bin/bash
# GPU job QUEUE for the entropy-weighted-prompt sweep on GDG-PA.
# 3 datasets x (adapt beta{0,0.5,1,2,4} + eval beta{0,0.5,1,2,4}) = 30 jobs.
# adapt beta=0 == bit-identical GDG-PA control; eval beta=0 == uniform ensemble.
# Mem-gated greedy placement with a BUFFER so it never OOMs other jobs.
#
# Run:  nohup bash bash/run_promptw_queue.sh > save/_sweep_logs/promptw_queue_main.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"; POLL="${POLL:-60}"; SETTLE="${SETTLE:-90}"
BUFFER="${BUFFER:-3000}"; GPUS="${GPUS:-0 1 2 3}"
mkdir -p save/_sweep_logs

BETAS="0.0 0.5 1.0 2.0 4.0"
# per-dataset conservative mem (MiB): full-res ACDC/Cityscapes vs VOC20 224.
declare -A MEM=( [acdc]=16000 [v20]=9000 [cityscapes]=16000 )
JOBS=""
for ds in acdc v20 cityscapes; do
  for side in adapt eval; do
    for b in $BETAS; do
      JOBS+="${ds}:${side}:${b}:${MEM[$ds]}"$'\n'
    done
  done
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
echo "[pwq] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending" | grep -c .) jobs, GPUs: $GPUS, rounds=$ROUNDS"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -a RES FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; r1="${job#*:}"; side="${r1%%:*}"; r2="${r1#*:}"; beta="${r2%%:*}"; need="${r2##*:}"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[pwq] $(date '+%H:%M') PLACE $ds $side b=$beta (need ${need}+${BUFFER}) -> GPU$g (free ${FREE[$g]}, res ${RES[$g]})"
        GPU=$g SEQ=1 ROUNDS=$ROUNDS bash bash/sweep_promptw.sh "$ds" "$side" "$beta" \
          > "save/_sweep_logs/pwq_${ds}_${side}_b${beta}.log" 2>&1 &
        RES[$g]=$(( ${RES[$g]} + need )); placed=1; progress=1; break
      fi
    done
    [ "$placed" -eq 0 ] && remaining+="${job}"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  nleft=$(printf '%s\n' "$pending" | grep -c .)
  if [ "$progress" -eq 1 ]; then
    echo "[pwq] $(date '+%H:%M') placed some; $nleft left; settling ${SETTLE}s"
    [ "$nleft" -gt 0 ] && sleep "$SETTLE"
  else
    echo "[pwq] $(date '+%H:%M') nothing fit; $nleft left; waiting ${POLL}s"; sleep "$POLL"
  fi
done
echo "[pwq] $(date '+%H:%M') all jobs launched; waiting..."
wait
echo "[pwq] $(date '+%m-%d %H:%M') ALL PROMPTW JOBS COMPLETE."
