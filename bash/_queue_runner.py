"""GPU-memory-aware queue runner for the 52 (13 method x 4 dataset) full+15corr
jobs. Launches by free MEMORY only (ignores utilization %, per instruction --
util can be bursty/misleading, memory headroom is what actually matters for
whether a new job will fit). Stacks multiple jobs per GPU whenever there's
room. Priority: all 4 `adagate` jobs first, then everything else in the given
order. Runs until the queue is empty; polls every POLL seconds.

Usage: nohup python bash/_queue_runner.py > /tmp/tta_full_runs_logs/queue_runner.log 2>&1 &
"""
import os
import subprocess
import time
import re

ROOT = "/home/stanfc/TTA-on-OVSS/MLMP"
# coco_obj is BLOCKED: COCOObjectDataset needs *_instanceTrainIds.png label
# files, which need to be generated from instances_val2017.json (now
# downloaded to .data/coco/annotations/) via a converter that doesn't exist
# yet. p59/p60 were unblocked by generating SegmentationClassContext/*.png
# from PASCAL-in-Detail's raw trainval .mat files (see
# .data/VOC2010/_convert_context.py) -- both smoke-tested clean (p59
# mIoU=24.91, p60 mIoU=23.13, R1, gaussian_noise). Re-add coco_obj once its
# label conversion is written.
FOLDERS = ["coco_stuff", "p59", "p60"]
MEM_THRESHOLD_MIB = 20000     # minimum free memory to consider a GPU open
POLL = 45                     # seconds between scheduling passes
N_GPUS = 6

# adagate (ours) was already launched manually on coco_stuff as the smoke
# test and is a real full run in progress
# (save/COCOStuffDataset/adagate_full_15corr/) -- don't re-queue it there,
# that would double-launch into the same save_dir. Still needs queuing for
# p59/p60.
ADAGATE_ALREADY_RUNNING = {"coco_stuff"}
PRIORITY_METHODS = ["adagate"]
REST_METHODS = ["vida_continual", "spa_continual", "ptta_continual", "come_continual",
                "tribe_continual", "mlmp_episodic", "no_adapt", "cotta",
                "tent_continual", "sar_continual", "deyo_continual", "mgp_deyo_mlmp_continual"]

QUEUE = []
for m in PRIORITY_METHODS:
    for f in FOLDERS:
        if f in ADAGATE_ALREADY_RUNNING:
            continue
        QUEUE.append((f, m))
for m in REST_METHODS:
    for f in FOLDERS:
        QUEUE.append((f, m))

LOG_DIR = "/tmp/tta_full_runs_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def free_mem_per_gpu():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"]).decode()
    free = []
    for i, line in enumerate(out.strip().splitlines()):
        used, total = [int(x) for x in line.split(",")]
        free.append(total - used)
    return free


CONDA_BIN = "/home/stanfc/miniconda3/envs/mlmp/bin"


def launch(folder, method, gpu):
    script = f"bash/{folder}/{method}_full_15corr.sh"
    log = f"{LOG_DIR}/{folder}_{method}.log"
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    # the launching shell may not have the mlmp conda env activated (bare
    # `python` in the .sh scripts would then resolve to nothing and the
    # script silently no-ops, since these scripts have no `set -e`) --
    # force it onto PATH regardless of the parent shell's state.
    env["PATH"] = CONDA_BIN + os.pathsep + env.get("PATH", "")
    with open(log, "w") as lf:
        p = subprocess.Popen(["bash", script], cwd=ROOT, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {folder}/{method} on GPU{gpu}  (pid {p.pid})  log={log}",
          flush=True)
    return p


def main():
    running = []   # list of (proc, folder, method, gpu)
    print(f"queue: {len(QUEUE)} jobs. threshold={MEM_THRESHOLD_MIB}MiB free, poll={POLL}s", flush=True)
    while QUEUE or running:
        # reap finished jobs
        still_running = []
        for p, folder, method, gpu in running:
            rc = p.poll()
            if rc is None:
                still_running.append((p, folder, method, gpu))
            else:
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                print(f"[{time.strftime('%H:%M:%S')}] DONE {folder}/{method} on GPU{gpu}  {status}", flush=True)
        running = still_running

        # Launch AT MOST ONE job per pass. A freshly-started process can take
        # 10-30s to actually claim its CUDA memory, so nvidia-smi under-reports
        # usage for a while after Popen returns -- looping "fill every open
        # GPU now" (the original design) raced this and dumped all 38 jobs
        # onto 6 GPUs within 2 seconds before any of them showed up as used,
        # which is exactly what happened the first time this ran. One launch
        # per POLL-second pass gives each process time to settle before the
        # next real nvidia-smi read is trusted.
        if QUEUE:
            free = free_mem_per_gpu()
            for gpu in range(N_GPUS):
                if free[gpu] >= MEM_THRESHOLD_MIB:
                    folder, method = QUEUE.pop(0)
                    p = launch(folder, method, gpu)
                    running.append((p, folder, method, gpu))
                    break

        if not QUEUE and not running:
            break
        time.sleep(POLL)

    print(f"[{time.strftime('%H:%M:%S')}] QUEUE EMPTY, all jobs finished.", flush=True)


if __name__ == "__main__":
    main()
