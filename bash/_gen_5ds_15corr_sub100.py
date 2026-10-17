"""Generate 100-image-subset + 15-corruption continual bash scripts for the
13 baselines, across 5 datasets (PascalVOC21, PascalContext59/60, COCOStuff,
COCOObject). Supersedes bash/_gen_4ds_full15corr.py's full-dataset scope --
full dataset x 150 rounds was estimated at weeks-to-months per (method,
dataset) for the heavy 18-layer methods, so this reuses the same method
templates but adds `--subset_size 100 --subset_seed 0` (the project's
existing sub100 convention, see bash/cityscapes_continual/) and renames the
save dir / script suffix from `_full_15corr` to `_15corr_sub100` so it
doesn't collide with the full-dataset scripts already in bash/coco_stuff/,
bash/p59/, bash/p60/.

coco_obj is still blocked (COCOObjectDataset needs *_instanceTrainIds.png,
not yet generated) -- scripts are written anyway so they're ready the
moment that's fixed, but don't queue them until then.
"""
import os
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))  # MLMP/bash

# reuse the already-written, already-corrected method templates instead of
# duplicating them
spec = importlib.util.spec_from_file_location(
    "_gen_4ds_full15corr", os.path.join(ROOT, "_gen_4ds_full15corr.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

DATASETS = {
    "v21":         ("PascalVOC21Dataset",     ".data/VOC2012/"),
    "p59":         ("PascalContext59Dataset", ".data/VOC2010/"),
    "p60":         ("PascalContext60Dataset", ".data/VOC2010/"),
    "coco_stuff":  ("COCOStuffDataset",       ".data/coco_stuff164k/"),
    "coco_obj":    ("COCOObjectDataset",      ".data/coco/"),  # still blocked, see module docstring
}

SUBSET_MARKER = "--corruptions_list $CONDITIONS --workers $WORKERS \\\n"
SUBSET_INSERT = ("--corruptions_list $CONDITIONS --workers $WORKERS \\\n"
                 "                        --subset_size $SUBSET_SIZE --subset_seed $SUBSET_SEED \\\n")


def to_sub100(body: str) -> str:
    assert SUBSET_MARKER in body, "expected marker line not found in template"
    body = body.replace(SUBSET_MARKER, SUBSET_INSERT, 1)
    body = body.replace("_full_15corr", "_15corr_sub100")
    # inject the SUBSET_SIZE/SUBSET_SEED shell vars right after WORKERS=0
    body = body.replace("WORKERS=0\n", "WORKERS=0\nSUBSET_SIZE=100\nSUBSET_SEED=0\n", 1)
    return body


if __name__ == "__main__":
    n = 0
    for folder, (dataset, data_dir) in DATASETS.items():
        outdir = os.path.join(ROOT, folder)
        os.makedirs(outdir, exist_ok=True)
        for name, fn in base.METHODS:
            body = to_sub100(fn(dataset, data_dir))
            path = os.path.join(outdir, f"{name}_15corr_sub100.sh")
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, 0o755)
            n += 1
    print(f"wrote {n} scripts across {len(DATASETS)} datasets x {len(base.METHODS)} methods")
