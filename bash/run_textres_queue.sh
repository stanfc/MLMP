#!/bin/bash
# GPU job QUEUE for the text-residual + orthogonality sweep on GDG-PA.
# 3 datasets x 5 arms (ctrl, orth 0.0/0.1/1.0/10.0) = 15 jobs.
#   ctrl = --text_res_lr 0 = bit-identical GDG-PA (verified by smoke test).
#   0.0  = residual on, regularizer OFF (does text-side degradation happen at all?).
# Mem-gated greedy placement with a BUFFER so it never OOMs the user's other jobs.
#
# Run:  nohup bash bash/run_textres_queue.sh > save/_sweep_logs/textres_queue_main.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"; POLL="${POLL:-60}"; SETTLE="${SETTLE:-90}"
BUFFER="${BUFFER:-3000}"; GPUS="${GPUS:-0 1 2 3}"
mkdir -p save/_sweep_logs

ARMS="${ARMS:-ctrl 0.0 0.1 1.0 10.0}"
# per-dataset conservative mem (MiB): full-res ACDC/Cityscapes vs VOC20 224.
declare -A MEM=( [acdc]=16000 [v20]=9000 [cityscapes]=16000 )
JOBS=""
for ds in acdc v20 cityscapes; do
  for a in $ARMS; do
    JOBS+="${ds}:${a}:${MEM[$ds]}"$'\n'
  done
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
echo "[trq] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending" | grep -c .) jobs, GPUs: $GPUS, rounds=$ROUNDS"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -a RES FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; r1="${job#*:}"; arm="${r1%%:*}"; need="${r1##*:}"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[trq] $(date '+%H:%M') PLACE $ds arm=$arm (need ${need}+${BUFFER}) -> GPU$g (free ${FREE[$g]}, res ${RES[$g]})"
        GPU=$g SEQ=1 ROUNDS=$ROUNDS bash bash/sweep_textres.sh "$ds" "$arm" \
          > "save/_sweep_logs/trq_${ds}_${arm}.log" 2>&1 &
        RES[$g]=$(( ${RES[$g]} + need )); placed=1; progress=1; break
      fi
    done
    [ "$placed" -eq 0 ] && remaining+="${job}"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  nleft=$(printf '%s\n' "$pending" | grep -c .)
  if [ "$progress" -eq 1 ]; then
    echo "[trq] $(date '+%H:%M') placed some; $nleft left; settling ${SETTLE}s"
    [ "$nleft" -gt 0 ] && sleep "$SETTLE"
  else
    echo "[trq] $(date '+%H:%M') nothing fit; $nleft left; waiting ${POLL}s"; sleep "$POLL"
  fi
done
echo "[trq] $(date '+%H:%M') all jobs launched; waiting..."
wait
echo "[trq] $(date '+%m-%d %H:%M') ALL TEXTRES JOBS COMPLETE."
