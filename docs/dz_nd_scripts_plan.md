# DZ / ND / Combined Scripts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new bash folders (21 scripts total) for native night-driving CTTA: Dark Zurich alone, Nighttime Driving alone, and DZ+ND combined. Adds two `BaseSegDataset` subclasses and dispatch wiring in `utils/segmentation_datasets.py`. No method-side code changes.

**Architecture:** Two new dataset classes (mirror `ACDCDataset`), two mmseg configs, three branches in `prepare_data()`, and one gated line so combined-mode keeps per-condition `data_root` overrides. All seven methods (no_adapt, mlmp_episodic, mlmp_continual, cotta, tent_continual, tent_divgate_continual, sar_continual) are dataset-agnostic and need no edits — only new bash scripts.

**Tech Stack:** Python 3.10, PyTorch 2.1.2, mmseg, NA-CLIP ViT-L/14. Bash for runners. No pytest in this repo; verification uses targeted `python -c` smoke checks and 1-round bash runs.

**Spec:** [docs/dz_nd_scripts_spec.md](dz_nd_scripts_spec.md)

---

## File Structure

| File | Responsibility | Change type |
|---|---|---|
| [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) | Dataset classes, mmseg configs, `prepare_data()` dispatch | Modify (single file, ~80 new lines + 1 line gated) |
| `bash/dark_zurich/{no_adapt,mlmp_episodic,mlmp_continual,cotta,tent_continual,tent_divgate_continual,sar_continual}.sh` | Per-method DZ-alone runners | Create (7 files) |
| `bash/nighttime_driving/{...}.sh` | Per-method ND-alone runners | Create (7 files) |
| `bash/dz_nd_combined/{...}.sh` | Per-method DZ→ND combined-stream runners | Create (7 files) |
| [CLAUDE.md](../CLAUDE.md) | Project guide — add three rows for new bash folders | Modify (1 section) |

**Why one big Python file**: existing convention. All datasets live in [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) (lines 22–339 for classes, 403–495 for configs, 500–600 for dispatch). Don't restructure — follow the pattern.

**Why three bash folders** (not one): the three experiments produce distinct `save/{dataset}/...` trees and have different `--corruptions_list` semantics. Keeping them separate matches the existing `bash/ACDC_10_round/` vs `bash/v20/` vs `bash/cityscapes_continual/` vs `bash/v20_acdc_matched/` pattern.

---

## Task 1: Add `DarkZurichDataset` class

**Files:**
- Modify: [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) — insert new class after `ACDCDataset` (currently ends at line 77)

- [ ] **Step 1: Insert the class definition**

After [utils/segmentation_datasets.py:77](../utils/segmentation_datasets.py#L77) (immediately after `ACDCDataset` closes), insert:

```python
@DATASETS.register_module()
class DarkZurichDataset(BaseSegDataset):
    """Dark Zurich dataset (val/night split only).

    Real-world night driving images from Zurich, anonymised (faces/plates blurred).
    Same 19 Cityscapes classes and label convention as ACDC.
    img_suffix     = '_rgb_anon.png'
    seg_map_suffix = '_gt_labelTrainIds.png'
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
```

- [ ] **Step 2: Verify import + registration**

Run:
```bash
conda run -n MLMP python3 -c "
from utils.segmentation_datasets import DarkZurichDataset
from mmseg.registry import DATASETS
print('Class loaded:', DarkZurichDataset.__name__)
print('Registered  :', 'DarkZurichDataset' in DATASETS.module_dict)
print('Classes count:', len(DarkZurichDataset.METAINFO['classes']))
"
```

Expected output:
```
Class loaded: DarkZurichDataset
Registered  : True
Classes count: 19
```

- [ ] **Step 3: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(dataset): add DarkZurichDataset class

Mirrors ACDCDataset structure for val/night split.
Same 19 Cityscapes classes; suffixes _rgb_anon.png / _gt_labelTrainIds.png."
```

---

## Task 2: Add `NighttimeDrivingDataset` class

**Files:**
- Modify: [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) — insert new class after `DarkZurichDataset` (from Task 1)

- [ ] **Step 1: Insert the class definition**

After the closing of `DarkZurichDataset` from Task 1, insert:

```python
@DATASETS.register_module()
class NighttimeDrivingDataset(BaseSegDataset):
    """Nighttime Driving Test dataset (Dai & Van Gool, ICCV 2018).

    Real-world night driving images with coarse Cityscapes-style GT.
    Same 19 Cityscapes classes and label convention as Cityscapes.
    img_suffix     = '_leftImg8bit.png'
    seg_map_suffix = '_gtCoarse_labelTrainIds.png'
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtCoarse_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
```

- [ ] **Step 2: Verify import + registration**

```bash
conda run -n MLMP python3 -c "
from utils.segmentation_datasets import NighttimeDrivingDataset
from mmseg.registry import DATASETS
print('Class loaded:', NighttimeDrivingDataset.__name__)
print('Registered  :', 'NighttimeDrivingDataset' in DATASETS.module_dict)
print('Classes count:', len(NighttimeDrivingDataset.METAINFO['classes']))
"
```

Expected:
```
Class loaded: NighttimeDrivingDataset
Registered  : True
Classes count: 19
```

- [ ] **Step 3: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(dataset): add NighttimeDrivingDataset class

Mirrors Cityscapes-style suffixes for the Dai & Van Gool 2018 test set.
Same 19 Cityscapes classes; suffixes _leftImg8bit.png / _gtCoarse_labelTrainIds.png."
```

---

## Task 3: Add `mm_darkzurich_cfg` and `mm_nightdriving_cfg`

**Files:**
- Modify: [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) — insert after `mm_acdc_cfg_base` (currently ends at line 446)

- [ ] **Step 1: Insert two mmseg config dicts**

After `mm_acdc_cfg_base` closes (line 446), insert:

```python
mm_darkzurich_cfg = {
    'type': 'DarkZurichDataset',
    'data_root': data_dir,  # overridden by prepare_data() per call
    'data_prefix': {'img_path': 'rgb_anon/val/night',
                    'seg_map_path': 'gt/val/night'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                 {'type': 'LoadAnnotations'},
                 {'type': 'ResizeAndPatchify', 'resize': resize,
                  'patch_size': patch_size, 'patch_stride': patch_stride},
                 {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                 ]
}

mm_nightdriving_cfg = {
    'type': 'NighttimeDrivingDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'leftImg8bit/test/night',
                    'seg_map_path': 'gtCoarse_daytime_trainvaltest/test/night'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                 {'type': 'LoadAnnotations'},
                 {'type': 'ResizeAndPatchify', 'resize': resize,
                  'patch_size': patch_size, 'patch_stride': patch_stride},
                 {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                 ]
}
```

- [ ] **Step 2: Verify the configs parse**

```bash
conda run -n MLMP python3 -c "
from utils.segmentation_datasets import mm_darkzurich_cfg, mm_nightdriving_cfg
print('DZ img_path :', mm_darkzurich_cfg['data_prefix']['img_path'])
print('DZ gt_path  :', mm_darkzurich_cfg['data_prefix']['seg_map_path'])
print('ND img_path :', mm_nightdriving_cfg['data_prefix']['img_path'])
print('ND gt_path  :', mm_nightdriving_cfg['data_prefix']['seg_map_path'])
"
```

Expected:
```
DZ img_path : rgb_anon/val/night
DZ gt_path  : gt/val/night
ND img_path : leftImg8bit/test/night
ND gt_path  : gtCoarse_daytime_trainvaltest/test/night
```

- [ ] **Step 3: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(dataset): add mmseg configs for DarkZurich and NighttimeDriving

Per-dataset data_prefix follows native folder layout."
```

---

## Task 4: Add `prepare_data()` dispatch for the two single-dataset modes + extend native-shift whitelist

**Files:**
- Modify: [utils/segmentation_datasets.py:519-539](../utils/segmentation_datasets.py#L519-L539) — add two `elif` branches
- Modify: [utils/segmentation_datasets.py:553](../utils/segmentation_datasets.py#L553) — extend whitelist

- [ ] **Step 1: Add the two single-dataset dispatch branches**

Find the existing dispatch (around line 519–539) and insert two new `elif` branches after `PascalContext60Dataset` (before the `else: raise ValueError`):

```python
    elif dataset == "DarkZurichDataset":
        mm_config = copy.deepcopy(mm_darkzurich_cfg)
    elif dataset == "NighttimeDrivingDataset":
        mm_config = copy.deepcopy(mm_nightdriving_cfg)
```

- [ ] **Step 2: Extend the native-shift whitelist on line 553**

Replace the existing `if dataset == "ACDCDataset" or corruption == "original":` with:

```python
    # Datasets where condition is encoded in the data path itself (native shift)
    # do not need a synthetic CorruptTransform inserted into the pipeline.
    _NATIVE_SHIFT_DATASETS = (
        "ACDCDataset", "DarkZurichDataset", "NighttimeDrivingDataset",
    )
    if dataset in _NATIVE_SHIFT_DATASETS or corruption == "original":
        print("No corruption added to the pipeline")
```

Note: `DZ_ND_Combined` is added to this tuple in Task 6 (after its dispatch branch exists).

- [ ] **Step 3: Verify DZ loads exactly 50 samples**

```bash
conda run -n MLMP python3 -c "
import sys; sys.path.insert(0, '.')
from utils.segmentation_datasets import prepare_data
loader, classes = prepare_data(
    dataset='DarkZurichDataset',
    data_dir='data/Dark_Zurich_val_anon/',
    init_resize=(560, 1120),
    patch_size=(224, 224),
    patch_stride=112,
    corruption='night',
    batch_size=1,
    num_workers=0,
    shuffle=False,
)
print('DZ dataset size:', len(loader.dataset))
print('Classes        :', len(classes))
"
```

Expected:
```
No corruption added to the pipeline
DZ dataset size: 50
Classes        : 19
```

- [ ] **Step 4: Verify ND loads exactly 50 samples**

```bash
conda run -n MLMP python3 -c "
import sys; sys.path.insert(0, '.')
from utils.segmentation_datasets import prepare_data
loader, classes = prepare_data(
    dataset='NighttimeDrivingDataset',
    data_dir='data/NighttimeDrivingTest/',
    init_resize=(560, 1120),
    patch_size=(224, 224),
    patch_stride=112,
    corruption='night',
    batch_size=1,
    num_workers=0,
    shuffle=False,
)
print('ND dataset size:', len(loader.dataset))
print('Classes        :', len(classes))
"
```

Expected:
```
No corruption added to the pipeline
ND dataset size: 50
Classes        : 19
```

- [ ] **Step 5: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(dataset): wire DarkZurich/NighttimeDriving into prepare_data

Adds two elif branches and extends the native-shift whitelist tuple
so no CorruptTransform is inserted for these path-encoded datasets."
```

---

## Task 5: Gate the `data_root` override (line 543) for combined-mode

**Files:**
- Modify: [utils/segmentation_datasets.py:543](../utils/segmentation_datasets.py#L543)

This task prepares the ground for `DZ_ND_Combined` in Task 6, which needs to set `data_root` per condition. The blanket `mm_config['data_root'] = data_dir` assignment would clobber that, so we gate it.

- [ ] **Step 1: Replace line 543**

Find:
```python
    mm_config['data_root'] = data_dir
```

Replace with:
```python
    # DZ_ND_Combined sets data_root per-condition in its dispatch branch;
    # the blanket assignment would clobber it. Other datasets behave as before.
    if dataset != "DZ_ND_Combined":
        mm_config['data_root'] = data_dir
```

- [ ] **Step 2: Verify DZ and ND alone still work after gating**

The dataset name `DZ_ND_Combined` does not exist as a dispatch branch yet, so existing single-dataset code paths should be unaffected. Confirm by re-running Task 4 Step 3 verification (DZ → 50 samples).

```bash
conda run -n MLMP python3 -c "
import sys; sys.path.insert(0, '.')
from utils.segmentation_datasets import prepare_data
loader, _ = prepare_data(
    dataset='DarkZurichDataset',
    data_dir='data/Dark_Zurich_val_anon/',
    init_resize=(560, 1120),
    patch_size=(224, 224),
    patch_stride=112,
    corruption='night',
    batch_size=1, num_workers=0, shuffle=False,
)
print('DZ still loads:', len(loader.dataset))
"
```

Expected:
```
DZ still loads: 50
```

- [ ] **Step 3: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(dataset): gate line-543 data_root override

Prepares prepare_data() for the upcoming DZ_ND_Combined branch, which
needs per-condition data_root because the two sub-datasets live in
separate parent directories."
```

---

## Task 6: Add `prepare_data()` dispatch for `DZ_ND_Combined`

**Files:**
- Modify: [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) — add third branch + add to whitelist tuple

- [ ] **Step 1: Insert the combined-mode branch**

Immediately after the `NighttimeDrivingDataset` branch added in Task 4 Step 1 (and still before `else: raise ValueError`), insert:

```python
    elif dataset == "DZ_ND_Combined":
        # Combined-stream mode: args.data_dir is IGNORED. Sub-paths are
        # hardcoded per-condition because DZ and ND live in separate parents.
        # `corruption` is overloaded to name the sub-dataset.
        if corruption == "dark_zurich":
            mm_config = copy.deepcopy(mm_darkzurich_cfg)
            mm_config['data_root'] = "data/Dark_Zurich_val_anon/"
        elif corruption == "nighttime_driving":
            mm_config = copy.deepcopy(mm_nightdriving_cfg)
            mm_config['data_root'] = "data/NighttimeDrivingTest/"
        else:
            raise ValueError(
                f"DZ_ND_Combined: unknown sub-dataset '{corruption}', "
                f"expected 'dark_zurich' or 'nighttime_driving'")
```

- [ ] **Step 2: Add `DZ_ND_Combined` to the native-shift whitelist tuple**

Find the tuple inserted in Task 4 Step 2:
```python
    _NATIVE_SHIFT_DATASETS = (
        "ACDCDataset", "DarkZurichDataset", "NighttimeDrivingDataset",
    )
```

Replace with:
```python
    _NATIVE_SHIFT_DATASETS = (
        "ACDCDataset", "DarkZurichDataset", "NighttimeDrivingDataset",
        "DZ_ND_Combined",
    )
```

- [ ] **Step 3: Verify combined mode dispatches DZ sub-dataset**

```bash
conda run -n MLMP python3 -c "
import sys; sys.path.insert(0, '.')
from utils.segmentation_datasets import prepare_data
loader, classes = prepare_data(
    dataset='DZ_ND_Combined',
    data_dir='data/',  # ignored in combined mode
    init_resize=(560, 1120),
    patch_size=(224, 224),
    patch_stride=112,
    corruption='dark_zurich',
    batch_size=1, num_workers=0, shuffle=False,
)
print('DZ sub-dataset size:', len(loader.dataset))
print('Classes            :', len(classes))
"
```

Expected:
```
No corruption added to the pipeline
DZ sub-dataset size: 50
Classes            : 19
```

- [ ] **Step 4: Verify combined mode dispatches ND sub-dataset**

```bash
conda run -n MLMP python3 -c "
import sys; sys.path.insert(0, '.')
from utils.segmentation_datasets import prepare_data
loader, _ = prepare_data(
    dataset='DZ_ND_Combined',
    data_dir='data/',
    init_resize=(560, 1120),
    patch_size=(224, 224),
    patch_stride=112,
    corruption='nighttime_driving',
    batch_size=1, num_workers=0, shuffle=False,
)
print('ND sub-dataset size:', len(loader.dataset))
"
```

Expected:
```
No corruption added to the pipeline
ND sub-dataset size: 50
```

- [ ] **Step 5: Verify invalid sub-dataset name raises**

```bash
conda run -n MLMP python3 -c "
import sys; sys.path.insert(0, '.')
from utils.segmentation_datasets import prepare_data
try:
    prepare_data(
        dataset='DZ_ND_Combined',
        data_dir='data/',
        init_resize=(560, 1120),
        patch_size=(224, 224),
        patch_stride=112,
        corruption='night',  # invalid in combined mode
        batch_size=1, num_workers=0, shuffle=False,
    )
    print('FAIL: no exception raised')
except ValueError as e:
    print('OK ValueError:', e)
"
```

Expected:
```
OK ValueError: DZ_ND_Combined: unknown sub-dataset 'night', expected 'dark_zurich' or 'nighttime_driving'
```

- [ ] **Step 6: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(dataset): add DZ_ND_Combined dispatch in prepare_data

Combined-stream mode overloads --corruptions_list to name sub-datasets.
Sub-paths are hardcoded; args.data_dir is ignored in this mode."
```

---

## Task 7: Create `bash/dark_zurich/no_adapt.sh` + smoke test (1 round)

**Files:**
- Create: `bash/dark_zurich/no_adapt.sh`

- [ ] **Step 1: Create the folder**

```bash
mkdir -p bash/dark_zurich
```

- [ ] **Step 2: Write the script**

Save to `bash/dark_zurich/no_adapt.sh`:

```bash
#!/bin/bash
# No-Adaptation baseline on Dark Zurich (CTTA setup, 150 rounds).
# Runs the off-the-shelf NA-CLIP model without any weight updates.
# Establishes the zero-shot source mIoU for this dataset.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
INIT_RESIZE="1120 560"
CONDITIONS="night"
WORKERS=4

# Model Configuration
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
METHOD="tent_continual"   # lightest method; --adapt omitted so no updates occur

# Experiment
CONTINUAL_ROUNDS=150
BATCH_SIZE=1

# Output
SAVE_DIR="save/${DATASET}/No_Adaptation/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 3: Smoke run with `CONTINUAL_ROUNDS=1` override**

```bash
CONTINUAL_ROUNDS=1 SAVE_DIR=save/_smoke/dz_no_adapt/ bash bash/dark_zurich/no_adapt.sh 2>&1 | tail -20
```

Note: the script uses a literal `CONTINUAL_ROUNDS=150`, not an env-var fallback. Either edit inline first to `CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}` for this smoke run, or replace `--continual_rounds 150` to `--continual_rounds 1` manually for the test. The env-var-friendly form (`${CONTINUAL_ROUNDS:-150}`) is the cleaner choice — update the script body to use it, then re-run.

Expected: completes one round and writes:
```
save/_smoke/dz_no_adapt/results_all_rounds.txt
```

containing one row like:
```
Round, night, Mean_mIoU
Round 01, XX.XX, XX.XX
```

- [ ] **Step 4: Verify output file header**

```bash
head -2 save/_smoke/dz_no_adapt/results_all_rounds.txt
```

Expected: header `Round, night, Mean_mIoU` followed by one data row.

- [ ] **Step 5: Restore `CONTINUAL_ROUNDS=150` if you edited inline; commit**

```bash
git add bash/dark_zurich/no_adapt.sh
git commit -m "feat(bash): add Dark Zurich no_adapt script

Mirrors bash/ACDC_10_round/no_adapt.sh; CONDITIONS=night; 150 rounds."
```

(Smoke output under `save/_smoke/` is ignored — clean it after all smoke tests pass.)

---

## Task 8: Create the remaining 6 `bash/dark_zurich/` scripts

**Files:**
- Create: `bash/dark_zurich/{mlmp_episodic,mlmp_continual,cotta,tent_continual,tent_divgate_continual,sar_continual}.sh`

For each script, copy the corresponding `bash/ACDC_10_round/` file and apply the substitutions below. Three scripts also need explicit hyperparameter overrides because the source has stale or buggy values.

- [ ] **Step 1: `mlmp_episodic.sh`**

Source: `bash/ACDC_10_round/mlmp.sh` (uses `main.py`, not `main_continual.py`).

Changes vs source:
```
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
CONDITIONS="night"
SAVE_DIR="save/${DATASET}/mlmp/"   # (was mlmp_episodic_step_X; we drop the suffix)
```
All other lines verbatim (`LR=0.001`, `STEPS=1`, `TRIALS=1`, `BATCH_SIZE=1`, prompt+vision args).

- [ ] **Step 2: `mlmp_continual.sh`**

Source: `bash/ACDC_10_round/mlmp_continual.sh`.

Changes:
```
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
CONDITIONS="night"
SAVE_DIR="save/${DATASET}/${METHOD}/"
```
All other lines verbatim (`LR=0.00001`, `STEPS=1`, `CONTINUAL_ROUNDS=150`).

- [ ] **Step 3: `cotta.sh` — also override `RST` to 0.01**

Source: `bash/ACDC_10_round/cotta.sh`. **Note: ACDC source has `RST=0.00` (same bug observed on bash/v20_acdc_matched/). Set explicitly to 0.01.**

Changes:
```
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
CONDITIONS="night"
RST=0.01                     # was 0.00 in ACDC source — the stochastic-restore bug fix
SAVE_DIR="save/${DATASET}/${METHOD}/"
```

- [ ] **Step 4: `tent_continual.sh` — also override `STEPS` and `CONTINUAL_ROUNDS`**

Source: `bash/ACDC_10_round/tent_continual.sh`. **Note: source has `STEPS=10` and `CONTINUAL_ROUNDS=10` (legacy short experiment). Use `STEPS=1` and `CONTINUAL_ROUNDS=150` for consistency with the rest of the new folder.**

Changes:
```
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
CONDITIONS="night"
STEPS=1                      # was 10
CONTINUAL_ROUNDS=150         # was 10
SAVE_DIR="save/${DATASET}/${METHOD}/"
```

- [ ] **Step 5: `tent_divgate_continual.sh` — also override `H_THRESHOLD`**

Source: `bash/ACDC_10_round/tent_divgate_continual.sh`. **Note: source has `H_THRESHOLD=1.8` but CLAUDE.md notes the "best confirmed" value is 1.6 (mean=31.59 vs 1.8's 30.14 on ACDC). Set explicitly to 1.6.**

Changes:
```
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
CONDITIONS="night"
H_THRESHOLD=1.6              # was 1.8 in ACDC source; CLAUDE.md best-confirmed value
SAVE_DIR="save/${DATASET}/${METHOD}/"
```
Keep `H_WARNING=1.4`, `CAUTIOUS_RST=0.01`, `BRAKE_RST=0.05`, `MONITOR_INTERVAL=50` from the source.

- [ ] **Step 6: `sar_continual.sh`**

Source: `bash/ACDC_10_round/sar_continual.sh`.

Changes:
```
DATASET=DarkZurichDataset
DATA_DIR="data/Dark_Zurich_val_anon/"
CONDITIONS="night"
SAVE_DIR="save/${DATASET}/${METHOD}/"
```
All other lines verbatim (`E_MARGIN=1.8`, `SAM_RHO=0.05`, `E_0=0.1`, `EMA_FACTOR=0.9`, `RECOVERY_WARMUP=50`).

- [ ] **Step 7: Verify all 7 files exist and are not empty**

```bash
ls -la bash/dark_zurich/
for f in bash/dark_zurich/*.sh; do
  echo "$f: $(wc -l < $f) lines"
done
```

Expected: 7 files, each 40–80 lines.

- [ ] **Step 8: Commit**

```bash
git add bash/dark_zurich/
git commit -m "feat(bash): add 6 more Dark Zurich method scripts

mlmp_episodic, mlmp_continual, cotta, tent_continual,
tent_divgate_continual, sar_continual.

Overrides vs ACDC source: cotta RST=0.01 (bug fix), tent_continual
STEPS=1 + ROUNDS=150 (was 10/10 legacy), tent_divgate H_THRESHOLD=1.6
(CLAUDE.md best-confirmed)."
```

---

## Task 9: Create all 7 `bash/nighttime_driving/` scripts

**Files:**
- Create: `bash/nighttime_driving/{no_adapt,mlmp_episodic,mlmp_continual,cotta,tent_continual,tent_divgate_continual,sar_continual}.sh`

- [ ] **Step 1: Create the folder and clone DZ scripts**

```bash
mkdir -p bash/nighttime_driving
for f in bash/dark_zurich/*.sh; do
    name=$(basename "$f")
    cp "$f" "bash/nighttime_driving/$name"
done
ls bash/nighttime_driving/
```

Expected: 7 files copied.

- [ ] **Step 2: Apply the two text substitutions in every file**

```bash
sed -i \
    -e 's|DATASET=DarkZurichDataset|DATASET=NighttimeDrivingDataset|' \
    -e 's|DATA_DIR="data/Dark_Zurich_val_anon/"|DATA_DIR="data/NighttimeDrivingTest/"|' \
    bash/nighttime_driving/*.sh
```

`SAVE_DIR` derives from `${DATASET}` so it follows automatically. `CONDITIONS=night` is identical for DZ and ND (the path-encoded condition name is "night" in both).

- [ ] **Step 3: Update header comments**

Each script's leading comment block currently says "Dark Zurich". Update with sed:

```bash
sed -i \
    -e 's|Dark Zurich|Nighttime Driving|g' \
    -e 's|on Dark Zurich|on Nighttime Driving|g' \
    bash/nighttime_driving/*.sh
```

(Idempotent on second run.)

- [ ] **Step 4: Spot-check one script**

```bash
head -15 bash/nighttime_driving/tent_divgate_continual.sh
```

Expected: title comment mentions "Nighttime Driving", `DATASET=NighttimeDrivingDataset`, `DATA_DIR="data/NighttimeDrivingTest/"`, `CONDITIONS="night"`.

- [ ] **Step 5: Smoke run `no_adapt.sh` (1 round)**

```bash
# Same env-var override pattern as DZ smoke. If the script uses literal 150,
# update to ${CONTINUAL_ROUNDS:-150} first.
CONTINUAL_ROUNDS=1 SAVE_DIR=save/_smoke/nd_no_adapt/ bash bash/nighttime_driving/no_adapt.sh 2>&1 | tail -10
head -2 save/_smoke/nd_no_adapt/results_all_rounds.txt
```

Expected: file with header `Round, night, Mean_mIoU` and one data row.

- [ ] **Step 6: Commit**

```bash
git add bash/nighttime_driving/
git commit -m "feat(bash): add Nighttime Driving method scripts

Same 7 methods as bash/dark_zurich/. Differs only in
DATASET / DATA_DIR / header comments."
```

---

## Task 10: Create all 7 `bash/dz_nd_combined/` scripts

**Files:**
- Create: `bash/dz_nd_combined/{no_adapt,mlmp_episodic,mlmp_continual,cotta,tent_continual,tent_divgate_continual,sar_continual}.sh`

- [ ] **Step 1: Create the folder and clone DZ scripts**

```bash
mkdir -p bash/dz_nd_combined
for f in bash/dark_zurich/*.sh; do
    name=$(basename "$f")
    cp "$f" "bash/dz_nd_combined/$name"
done
```

- [ ] **Step 2: Apply combined-mode substitutions**

```bash
sed -i \
    -e 's|DATASET=DarkZurichDataset|DATASET=DZ_ND_Combined|' \
    -e 's|DATA_DIR="data/Dark_Zurich_val_anon/"|DATA_DIR="data/"  # ignored in combined-stream mode|' \
    -e 's|CONDITIONS="night"|CONDITIONS="dark_zurich nighttime_driving"|' \
    bash/dz_nd_combined/*.sh
```

`SAVE_DIR` becomes `save/DZ_ND_Combined/{METHOD}/` automatically.

- [ ] **Step 3: Update header comments**

```bash
sed -i \
    -e 's|on Dark Zurich|on DZ+ND combined stream (2-sub-dataset night CTTA)|g' \
    -e 's|Dark Zurich dataset|DZ+ND combined stream|g' \
    bash/dz_nd_combined/*.sh
```

- [ ] **Step 4: Spot-check one script**

```bash
head -15 bash/dz_nd_combined/tent_divgate_continual.sh
```

Expected: `DATASET=DZ_ND_Combined`, `DATA_DIR="data/"  # ignored ...`, `CONDITIONS="dark_zurich nighttime_driving"`.

- [ ] **Step 5: Smoke run `no_adapt.sh` (1 round)**

```bash
CONTINUAL_ROUNDS=1 SAVE_DIR=save/_smoke/combined_no_adapt/ \
    bash bash/dz_nd_combined/no_adapt.sh 2>&1 | tail -10
head -2 save/_smoke/combined_no_adapt/results_all_rounds.txt
```

Expected header:
```
Round, dark_zurich, nighttime_driving, Mean_mIoU
Round 01, XX.XX, YY.YY, ZZ.ZZ
```

Where `ZZ.ZZ` is the arithmetic mean of `XX.XX` and `YY.YY`.

- [ ] **Step 6: Cleanup smoke output (now that all 3 folders verified)**

```bash
rm -rf save/_smoke/
```

- [ ] **Step 7: Commit**

```bash
git add bash/dz_nd_combined/
git commit -m "feat(bash): add DZ+ND combined-stream method scripts

CONDITIONS='dark_zurich nighttime_driving' streams DZ then ND blockwise
within each round. DATA_DIR placeholder is ignored by prepare_data."
```

---

## Task 11: Update [CLAUDE.md](../CLAUDE.md) — add the three new bash folders

**Files:**
- Modify: [CLAUDE.md](../CLAUDE.md) — add section after the existing `bash/v20_acdc_matched/` description (find heading "VOC20 ACDC-Matched Scripts")

- [ ] **Step 1: Insert new section after the v20_acdc_matched section ends**

The existing CLAUDE.md ends the v20_acdc_matched section with a paragraph about "Smoke-verified..." (search for it). Immediately after, add this new section:

```markdown
---

## Dark Zurich / Nighttime Driving / Combined Scripts

Three new bash folders (21 scripts total) targeting **native night-driving CTTA** — both datasets are real adverse-condition images (no synthetic CorruptTransform), Cityscapes 19-class compatible. See [docs/dz_nd_scripts_spec.md](docs/dz_nd_scripts_spec.md).

| Folder | DATASET | DATA_DIR | CONDITIONS | imgs/round | Purpose |
|---|---|---|---|---|---|
| `bash/dark_zurich/` | DarkZurichDataset | data/Dark_Zurich_val_anon/ | night | 50 | Single-domain stream A |
| `bash/nighttime_driving/` | NighttimeDrivingDataset | data/NighttimeDrivingTest/ | night | 50 | Single-domain stream B (external validity) |
| `bash/dz_nd_combined/` | DZ_ND_Combined | data/ (ignored) | dark_zurich nighttime_driving | 100 | 2-sub-dataset native-shift CTTA |

Each folder has the same 7 method scripts: `no_adapt`, `mlmp_episodic`, `mlmp_continual`, `cotta` (RST=0.01), `tent_continual` (STEPS=1, ROUNDS=150), `tent_divgate_continual` (H_THRESHOLD=1.6), `sar_continual`.

Combined-mode internals: `args.data_dir` is overloaded to a placeholder; `prepare_data()` dispatches per-condition to either DZ or ND with hardcoded sub-paths. See [utils/segmentation_datasets.py](utils/segmentation_datasets.py) `DZ_ND_Combined` branch.

Results saved to `save/{DarkZurichDataset|NighttimeDrivingDataset|DZ_ND_Combined}/{method}/results_all_rounds.txt`. Single-condition folders have columns `Round, night, Mean_mIoU`; combined has `Round, dark_zurich, nighttime_driving, Mean_mIoU`.
```

- [ ] **Step 2: Verify the section renders cleanly**

```bash
grep -A 5 "Dark Zurich / Nighttime Driving" CLAUDE.md | head -20
```

Expected: section header and table visible.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): document DZ/ND/Combined bash folders

Adds quick-reference table + key hyperparameter overrides (CoTTA RST,
TENT-DivGate H_THRESHOLD) so future sessions immediately know the
non-stale values vs the legacy ACDC source scripts."
```

---

## Self-Review

**Spec coverage check** (run mentally against [docs/dz_nd_scripts_spec.md](dz_nd_scripts_spec.md)):

| Spec section | Implementing task(s) |
|---|---|
| §3.1 Dataset classes | Tasks 1, 2 |
| §3.2 mmseg configs | Task 3 |
| §3.3 prepare_data dispatch + line-543 gate + whitelist | Tasks 4, 5, 6 |
| §3.4 main.py / main_continual.py untouched | (no task — confirms via Task 7+ smoke runs) |
| §3.5 adapt/* untouched | (no task) |
| §4.1 Per-folder variables | Tasks 7, 8, 9, 10 |
| §4.2 Per-script bodies | Tasks 8, 9, 10 |
| §4.3 Stream sequencing (DZ→ND block) | Task 10 Step 2 (CONDITIONS order) |
| §4.4 CoTTA RST=0.01 | Task 8 Step 3 |
| §4.5 mlmp_episodic uses main.py | Task 8 Step 1 |
| §5 Output formats | Verified in smoke runs (Tasks 7 Step 4, 9 Step 5, 10 Step 5) |
| §6 ACDC hyperparams | Tasks 8 (with documented overrides for cotta/tent/tent_divgate) |
| §7 Files created/modified | All 11 tasks |
| §8 Sanity checks 1–8 | Distributed across Task 1 Step 2, Task 2 Step 2, Tasks 4–6 verify, Tasks 7/9/10 smoke runs |

All spec items mapped.

**Placeholder scan**: no TBD/TODO/"fill in"/"similar to". Every code block is complete. ✓

**Type consistency**:
- `DZ_ND_Combined` (no quotes around `DZ`) used consistently across Tasks 5, 6, 8, 10. ✓
- `DarkZurichDataset` / `NighttimeDrivingDataset` used consistently. ✓
- `mm_darkzurich_cfg` / `mm_nightdriving_cfg` referenced in Task 6 match Task 3 definitions. ✓
- `_NATIVE_SHIFT_DATASETS` added in Task 4 and extended in Task 6 — same name. ✓

Plan ready.
