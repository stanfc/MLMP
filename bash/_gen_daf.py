"""Generate daf_mlmp_continual scripts for the sub100+15corr batch, matching
the exact convention of bash/_gen_5ds_15corr_sub100.py (same HEADER, same
folders, same corruption list, same 150-round/sub100 scope) so DAF's results
land alongside the other 13 methods for direct comparison.

Hyperparameters follow the official DAF README example (LR=1e-3, batch=8 --
DAF's Filter/SAFS module is a batch statistic that degenerates to a no-op at
batch_size=1, so batch=8 is kept rather than forced to our usual batch=1;
same precedent as DPCore's batch_size=16 in this codebase), not the
1e-5/5e-6 used by our other continual baselines.
"""
import os
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "_gen_4ds_full15corr", os.path.join(ROOT, "_gen_4ds_full15corr.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

DATASETS = {
    "v21":        ("PascalVOC21Dataset",     ".data/VOC2012/"),
    "p59":        ("PascalContext59Dataset", ".data/VOC2010/"),
    "p60":        ("PascalContext60Dataset", ".data/VOC2010/"),
    "coco_stuff": ("COCOStuffDataset",       ".data/coco_stuff164k/"),
}

VISION18 = base.VISION18
CONDITIONS = base.CONDITIONS
HEADER = base.HEADER
NOTIFY = base.NOTIFY


def daf(dataset, data_dir):
    return HEADER.format(comment="DAF (Diversify, Anchor, Filter) on MLMP-continual, sub100+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + f'''
SUBSET_SIZE=100
SUBSET_SEED=0

METHOD="daf_mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
LAMB_ENT=1.0
LAMB_DIV=2.0
LAMB_CMAC=0.5
ALPHA_SAFS=0.5

BATCH_SIZE=8
LR=0.001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${{DATASET}}/daf_mlmp_continual_15corr_sub100/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --prompt_dir $PROMPT_DIR \\
                        --alpha_cls $ALPHA_CLS \\
                        --lamb_ent $LAMB_ENT --lamb_div $LAMB_DIV --lamb_cmac $LAMB_CMAC \\
                        --alpha_safs $ALPHA_SAFS \\
                        --vision_outputs {VISION18} \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --subset_size $SUBSET_SIZE --subset_seed $SUBSET_SEED \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


if __name__ == "__main__":
    n = 0
    for folder, (dataset, data_dir) in DATASETS.items():
        outdir = os.path.join(ROOT, folder)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "daf_mlmp_continual_15corr_sub100.sh")
        with open(path, "w") as f:
            f.write(daf(dataset, data_dir))
        os.chmod(path, 0o755)
        n += 1
    print(f"wrote {n} DAF scripts")
