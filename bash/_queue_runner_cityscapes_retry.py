"""Retry queue for the 3 Cityscapes-full jobs that OOM'd under the old 20GB
threshold (spa_continual, tribe_continual, mgp_deyo_mlmp_continual) --
root cause was another user's (pinyu0121) two ~87GB VLLM processes eating
GPU headroom that a 20GB free-at-launch check didn't protect against.
Threshold raised to 30GB here.
"""
import os
import subprocess
import time

ROOT = "/home/stanfc/TTA-on-OVSS/MLMP"
MEM_THRESHOLD_MIB = 30000
POLL = 45
N_GPUS = 6

QUEUE = [
    ("cityscapes", "spa_continual"),
    ("cityscapes", "tribe_continual"),
    ("cityscapes", "mgp_deyo_mlmp_continual"),
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


def launch(folder, method, gpu):
    script = "bash/%s/%s_full_15corr.sh" % (folder, method)
    log = "%s/%s_%s_full15corr.log" % (LOG_DIR, folder, method)
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    env["PATH"] = CONDA_BIN + os.pathsep + env.get("PATH", "")
    with open(log, "w") as lf:
        p = subprocess.Popen(["bash", script], cwd=ROOT, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    print("[%s] LAUNCH %s/%s on GPU%d  (pid %d)  log=%s" %
          (time.strftime('%H:%M:%S'), folder, method, gpu, p.pid, log), flush=True)
    return p


def main():
    queue = list(QUEUE)
    running = []
    print("queue: %d jobs. threshold=%dMiB free, poll=%ds" % (len(queue), MEM_THRESHOLD_MIB, POLL), flush=True)
    while queue or running:
        still = []
        for p, folder, method, gpu in running:
            rc = p.poll()
            if rc is None:
                still.append((p, folder, method, gpu))
            else:
                status = "OK" if rc == 0 else "FAILED(rc=%s)" % rc
                print("[%s] DONE %s/%s on GPU%d  %s" %
                      (time.strftime('%H:%M:%S'), folder, method, gpu, status), flush=True)
        running = still

        if queue:
            free = free_mem_per_gpu()
            for gpu in range(N_GPUS):
                if free[gpu] >= MEM_THRESHOLD_MIB:
                    folder, method = queue.pop(0)
                    p = launch(folder, method, gpu)
                    running.append((p, folder, method, gpu))
                    break

        if not queue and not running:
            break
        time.sleep(POLL)

    print("[%s] QUEUE EMPTY, all jobs finished." % time.strftime('%H:%M:%S'), flush=True)


if __name__ == "__main__":
    main()
