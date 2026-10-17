#!/usr/bin/env python3
"""Generate V20/Cityscapes sub100-15corr variants of the *_full_15corr scripts,
for the exact method set used in the two full-scale progress figures (CAGA plain
only -- no scale6/hgate variants). Same corruptions, same everything, just adds
--subset_size 100 --subset_seed 0 and a distinct save dir so it never collides
with the full-scale runs.
"""
import re
import os

METHODS = [
    "adagate", "no_adapt", "daf_mlmp_continual", "tent_continual", "sar_continual",
    "spa_continual", "tribe_continual", "deyo_continual", "vida_continual",
    "ptta_continual", "come_continual", "mgp_deyo_mlmp_continual", "mlmp_episodic",
]

for folder in ("v20", "cityscapes"):
    for m in METHODS:
        src_path = f"bash/{folder}/{m}_full_15corr.sh"
        dst_path = f"bash/{folder}/{m}_sub100_15corr.sh"
        with open(src_path) as f:
            content = f.read()

        # save dir: _full_15corr/ -> _sub100_15corr/
        content = content.replace(f"{m}_full_15corr/", f"{m}_sub100_15corr/")

        # insert SUBSET_SIZE/SUBSET_SEED vars right after CONTINUAL_ROUNDS or TRIALS line
        if "CONTINUAL_ROUNDS=" in content:
            content = re.sub(
                r"(CONTINUAL_ROUNDS=\d+\n)",
                r"\1SUBSET_SIZE=100\nSUBSET_SEED=0\n",
                content, count=1)
        elif "TRIALS=" in content:
            content = re.sub(
                r"(TRIALS=\d+\n)",
                r"\1SUBSET_SIZE=100\nSUBSET_SEED=0\n",
                content, count=1)
        else:
            raise RuntimeError(f"no anchor line found in {src_path}")

        # insert --subset_size/--subset_seed flags right after --seed 0 \ line
        content = re.sub(
            r"(--seed 0 \\\n)",
            r"\1                        --subset_size $SUBSET_SIZE --subset_seed $SUBSET_SEED \\\n",
            content, count=1)

        with open(dst_path, "w") as f:
            f.write(content)
        os.chmod(dst_path, 0o755)
        print("wrote", dst_path)
