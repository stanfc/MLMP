"""Generate the missing baseline full-scale scripts for the paper's 3 main
figures (V20-full, Cityscapes-full, ACDC -- ACDC is already 100% complete).
Reuses the 12 method templates from _gen_4ds_full15corr.py, re-pointed at
V20/Cityscapes with each dataset's own resize convention and the established
"_full_15corr" 50-round precedent already used by adagate_full_15corr.sh /
daf_mlmp_continual_full_15corr.sh in these folders.
"""
import os
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "_gen_4ds_full15corr", os.path.join(ROOT, "_gen_4ds_full15corr.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

TARGETS = {
    "v20":        ("PascalVOC20Dataset", ".data/VOC2012/",   "224 224",   ["mgp_deyo_mlmp_continual"]),
    "cityscapes": ("CityscapesDataset",  ".data/cityscapes/", "1120 560", []),
}

METHOD_FNS = {name: fn for name, fn in base.METHODS if name != "adagate"}


def adjust(body, init_resize):
    body = body.replace('INIT_RESIZE="224 224"', 'INIT_RESIZE="' + init_resize + '"')
    body = body.replace("CONTINUAL_ROUNDS=150", "CONTINUAL_ROUNDS=50")
    return body


if __name__ == "__main__":
    n = 0
    for folder, (dataset, data_dir, init_resize, skip) in TARGETS.items():
        outdir = os.path.join(ROOT, folder)
        os.makedirs(outdir, exist_ok=True)
        for name, fn in METHOD_FNS.items():
            if name in skip:
                continue
            path = os.path.join(outdir, name + "_full_15corr.sh")
            if os.path.exists(path):
                print("skip (exists):", path)
                continue
            body = adjust(fn(dataset, data_dir), init_resize)
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, 0o755)
            n += 1
            print("wrote", path)
    print("wrote", n, "scripts total")
