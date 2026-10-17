"""GPU-memory-aware queue runner for the sub100+15corr batch (13 methods x 4
launchable datasets: v21, p59, p60, coco_stuff -- coco_obj still blocked on
missing *_instanceTrainIds.png label conversion, scripts exist but aren't
queued yet). Supersedes the full-dataset _queue_runner.py: full dataset x 150
rounds measured at weeks-to-months per (method,dataset) under real GPU
contention, so the whole batch moved to the project's existing 100-image
subset convention (see bash/cityscapes_continual/) while keeping all 15
ImageNet-C corruptions and 150 rounds.

Launches by free MEMORY only (ignores utilization%). One launch per POLL-
second pass (not a burst-fill) so nvidia-smi has time to reflect a freshly
started process's real usage before the next launch decision -- see the
comment in the original _queue_runner.py for why a burst-fill races CUDA
startup and over-launches.

Usage: nohup python bash/_queue_runner_sub100.py > /tmp/tta_full_runs_logs/queue_runner_sub100.log 2>&1 &
"""
import os
import subprocess
import time

ROOT = "/home/stanfc/TTA-on-OVSS/MLMP"
FOLDERS = ["v21", "p59", "p60", "coco_stuff"]
MEM_THRESHOLD_MIB = 20000
POLL = 45
N_GPUS = 6

PRIORITY_METHODS = ["adagate"]
REST_METHODS = ["vida_continual", "spa_continual", "ptta_continual", "come_continual",
                "tribe_continual", "mlmp_episodic", "no_adapt", "cotta",
                "tent_continual", "sar_continual", "deyo_continual", "mgp_deyo_mlmp_continual"]

QUEUE = []
for m in PRIORITY_METHODS:
    for f in FOLDERS:
        QUEUE.append((f, m))
for m in REST_METHODS:
    for f in FOLDERS:
        QUEUE.append((f, m))

LOG_DIR = "/tmp/tta_full_runs_logs"
os.makedirs(LOG_DIR, exist_ok=True)

CONDA_BIN = "/home/stanfc/miniconda3/envs/mlmp/bin"


def free_mem_per_gpu():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"]).decode()
    free = []
    for line in out.strip().splitlines():
        used, total = [int(x) for x in line.split(",")]
        free.append(total - used)
    return free


def launch(folder, method, gpu):
    script = f"bash/{folder}/{method}_15corr_sub100.sh"
    log = f"{LOG_DIR}/sub100_{folder}_{method}.log"
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    env["PATH"] = CONDA_BIN + os.pathsep + env.get("PATH", "")
    with open(log, "w") as lf:
        p = subprocess.Popen(["bash", script], cwd=ROOT, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {folder}/{method} on GPU{gpu}  (pid {p.pid})  log={log}",
          flush=True)
    return p


def main():
    running = []
    print(f"queue: {len(QUEUE)} jobs. threshold={MEM_THRESHOLD_MIB}MiB free, poll={POLL}s", flush=True)
    while QUEUE or running:
        still_running = []
        for p, folder, method, gpu in running:
            rc = p.poll()
            if rc is None:
                still_running.append((p, folder, method, gpu))
            else:
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                print(f"[{time.strftime('%H:%M:%S')}] DONE {folder}/{method} on GPU{gpu}  {status}", flush=True)
        running = still_running

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
