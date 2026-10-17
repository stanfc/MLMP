"""GPU-memory-aware queue runner for the 4 remaining full-scale jobs (V20-full
and Cityscapes-full, x {adagate, daf_mlmp_continual}) that were written but
never launched due to GPU pressure from the main sub100 batch. Same logic as
_queue_runner_sub100.py: launch by free MEMORY only, one launch per POLL-
second pass (not a burst-fill), ours (adagate) first.

Usage: nohup python bash/_queue_runner_full_extra.py > /tmp/tta_full_runs_logs/queue_runner_full_extra.log 2>&1 &
"""
import os
import subprocess
import time

ROOT = "/home/stanfc/TTA-on-OVSS/MLMP"
MEM_THRESHOLD_MIB = 20000
POLL = 45
N_GPUS = 6

# (script relative path, log name)
QUEUE = [
    ("bash/v20/adagate_full_15corr.sh",                    "v20_adagate_full15corr"),
    ("bash/cityscapes/adagate_full_15corr.sh",              "cityscapes_adagate_full15corr"),
    ("bash/v20/daf_mlmp_continual_full_15corr.sh",          "v20_daf_full15corr"),
    ("bash/cityscapes/daf_mlmp_continual_full_15corr.sh",   "cityscapes_daf_full15corr"),
]

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


def launch(script, name, gpu):
    log = f"{LOG_DIR}/{name}.log"
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    env["PATH"] = CONDA_BIN + os.pathsep + env.get("PATH", "")
    with open(log, "w") as lf:
        p = subprocess.Popen(["bash", script], cwd=ROOT, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {name} on GPU{gpu}  (pid {p.pid})  log={log}",
          flush=True)
    return p


def main():
    queue = list(QUEUE)
    running = []
    print(f"queue: {len(queue)} jobs. threshold={MEM_THRESHOLD_MIB}MiB free, poll={POLL}s", flush=True)
    while queue or running:
        still_running = []
        for p, script, name, gpu in running:
            rc = p.poll()
            if rc is None:
                still_running.append((p, script, name, gpu))
            else:
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                print(f"[{time.strftime('%H:%M:%S')}] DONE {name} on GPU{gpu}  {status}", flush=True)
        running = still_running

        if queue:
            free = free_mem_per_gpu()
            for gpu in range(N_GPUS):
                if free[gpu] >= MEM_THRESHOLD_MIB:
                    script, name = queue.pop(0)
                    p = launch(script, name, gpu)
                    running.append((p, script, name, gpu))
                    break

        if not queue and not running:
            break
        time.sleep(POLL)

    print(f"[{time.strftime('%H:%M:%S')}] QUEUE EMPTY, all jobs finished.", flush=True)


if __name__ == "__main__":
    main()
