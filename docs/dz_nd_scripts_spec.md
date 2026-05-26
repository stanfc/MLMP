# Dark Zurich / Nighttime Driving / Combined — Bash Scripts Spec

**Status**: Design approved, implementation pending
**Created**: 2026-05-26
**Owner**: Tekai-Yen
**Related docs**: [docs/EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md), [bash/ACDC_10_round/](../bash/ACDC_10_round/)

---

## 1. Motivation

The existing CTTA evidence on the project has two structural gaps:

1. **Only one native-shift benchmark.** ACDC is the only dataset with real adverse-condition images; VOC20 and Cityscapes use synthetic ImageNet-C corruptions and have shown low adaptation headroom (4.5 and 1.3 mIoU respectively), inconclusive for the headroom hypothesis. A second native-shift benchmark would strengthen external validity.
2. **`continual` always means cross-condition switching on ACDC.** Within ACDC, rounds alternate fog/night/rain/snow, so within-domain drift and cross-condition switching are confounded. A single-domain stream isolates "stability against pure drift."

Dark Zurich (DZ) and Nighttime Driving (ND) are both native night-driving datasets compatible with Cityscapes 19-class labels. Used three ways — DZ alone, ND alone, and DZ+ND combined — they answer:

| Experiment | Tests |
|---|---|
| DZ alone | Method stability under single-domain stream, native shift, dataset A |
| ND alone | Method stability under single-domain stream, native shift, dataset B (external validity) |
| DZ+ND combined | Method stability under a 2-domain native-shift stream (cross-dataset switch, no synthetic corruption) |

---

## 2. Data Layout (as found on local machine)

Both datasets share the **Cityscapes 19-class taxonomy** (same `METAINFO.classes`, same palette, same `trainIds` convention as `ACDCDataset` / `CityscapesDataset`). Only file naming and folder structure differ.

### 2.1 Dark Zurich (val/night split only)

- **Root**: `data/Dark_Zurich_val_anon/`
- **Images**: `rgb_anon/val/night/GOPR0356/*_rgb_anon.png` (50 files, single sequence folder)
- **GT (trainIds)**: `gt/val/night/GOPR0356/*_gt_labelTrainIds.png` (50 files)
- **File suffixes**: identical to `ACDCDataset` (`_rgb_anon.png` / `_gt_labelTrainIds.png`)
- Other GT variants present (`labelIds`, `labelColor`, `invIds`, `invGray`) are ignored.

### 2.2 Nighttime Driving (test/night split only)

- **Root**: `data/NighttimeDrivingTest/`
- **Images**: `leftImg8bit/test/night/*_leftImg8bit.png` (50 files, flat)
- **GT (trainIds)**: `gtCoarse_daytime_trainvaltest/test/night/*_gtCoarse_labelTrainIds.png` (50 files)
- **File suffixes**: Cityscapes-style with `gtCoarse` substituted for `gtFine`
- Other GT variants (`labelIds`, `polygons.json`) are ignored.

### 2.3 mmseg recurses into sequence subfolders

Verified empirically: `ACDCDataset` with `data_prefix.img_path='rgb_anon/fog/val'` finds all 100 fog images even though they live in sequence subfolders like `GOPR0476/`, `GP010476/`. The same will apply to Dark Zurich's single `GOPR0356/` subfolder. No recursion flag needed.

---

## 3. Code Changes

### 3.1 New dataset classes in [utils/segmentation_datasets.py](../utils/segmentation_datasets.py)

Both classes are near-verbatim copies of `ACDCDataset` (lines 50–77). Only docstring and `seg_map_suffix` differ.

```python
@DATASETS.register_module()
class DarkZurichDataset(BaseSegDataset):
    """Dark Zurich dataset (val/night split).

    Real-world night driving images from Zurich, anonymised.
    Same 19 Cityscapes classes and label convention as ACDC.
    img_suffix      = '_rgb_anon.png'
    seg_map_suffix  = '_gt_labelTrainIds.png'
    """
    METAINFO = dict(
        classes=...,   # identical 19-tuple as ACDCDataset.METAINFO['classes']
        palette=...,   # identical 19-entry palette as ACDCDataset.METAINFO['palette']
    )
    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class NighttimeDrivingDataset(BaseSegDataset):
    """Nighttime Driving Test dataset (Dai & Van Gool, ICCV 2018).

    Real-world night driving images with coarse Cityscapes-style GT.
    Same 19 Cityscapes classes and label convention as Cityscapes.
    img_suffix      = '_leftImg8bit.png'
    seg_map_suffix  = '_gtCoarse_labelTrainIds.png'
    """
    METAINFO = dict(
        classes=...,   # identical to ACDCDataset.METAINFO['classes']
        palette=...,   # identical to ACDCDataset.METAINFO['palette']
    )
    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtCoarse_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
```

### 3.2 New mmseg configs in [utils/segmentation_datasets.py](../utils/segmentation_datasets.py)

Added next to `mm_acdc_cfg_base` (around line 437):

```python
mm_darkzurich_cfg = {
    'type': 'DarkZurichDataset',
    'data_root': data_dir,  # overridden by prepare_data()
    'data_prefix': {'img_path': 'rgb_anon/val/night',
                    'seg_map_path': 'gt/val/night'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                 {'type': 'LoadAnnotations'},
                 {'type': 'ResizeAndPatchify', 'resize': resize,
                  'patch_size': patch_size, 'patch_stride': patch_stride},
                 {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD}],
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
                 {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD}],
}
```

### 3.3 `prepare_data()` dispatch (two changes)

**Change A** — add three new branches in the dataset dispatch (around line 539, after the existing `PascalContext60Dataset` branch). All three branches are independent; the combined branch references the two single configs.

```python
elif dataset == "DarkZurichDataset":
    mm_config = copy.deepcopy(mm_darkzurich_cfg)
elif dataset == "NighttimeDrivingDataset":
    mm_config = copy.deepcopy(mm_nightdriving_cfg)
elif dataset == "DZ_ND_Combined":
    # In combined-stream mode, args.data_dir is IGNORED. Sub-paths are
    # hardcoded per-condition because DZ and ND live in separate parents.
    # `corruption` here is overloaded to name the sub-dataset.
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

**Note**: The subsequent line `mm_config['data_root'] = data_dir` (line 543) would clobber the per-condition `data_root` we set for `DZ_ND_Combined`. Gate that assignment so combined-mode keeps its hardcoded sub-paths:

```python
# line 543 becomes:
if dataset != "DZ_ND_Combined":
    mm_config['data_root'] = data_dir
```

For `DarkZurichDataset` and `NighttimeDrivingDataset` alone, line 543 still runs and sets `data_root` from the bash script's `DATA_DIR`, matching existing behaviour.

**Change B** — add the two new single-condition datasets to the "no CorruptTransform" whitelist (line 553):

```python
_NATIVE_SHIFT_DATASETS = (
    "ACDCDataset", "DarkZurichDataset", "NighttimeDrivingDataset", "DZ_ND_Combined",
)
if dataset in _NATIVE_SHIFT_DATASETS or corruption == "original":
    print("No corruption added to the pipeline")
else:
    # ... existing CorruptTransform insertion ...
```

### 3.4 No changes to `main.py` / `main_continual.py`

The existing CLI plumbing already supports everything needed:
- `--dataset` accepts any string (passed to `prepare_data` and `get_method`)
- `--corruptions_list` accepts arbitrary tokens (already a free-form list)
- Round loop iterates `for cond in conditions` — works whether conditions are 1 (DZ/ND alone) or 2 (combined)

### 3.5 No changes to `adapt/__init__.py` or any `adapt/*` file

All seven methods (`no_adapt` via `tent_continual --no-adapt`, `mlmp`, `mlmp_continual`, `cotta`, `tent_continual`, `tent_divgate_continual`, `sar_continual`) are dataset-agnostic. No method-specific code needs to know about DZ/ND.

---

## 4. Bash Script Layout

Three new folders, seven scripts each, **21 scripts total**.

```
bash/dark_zurich/
├── no_adapt.sh
├── mlmp_episodic.sh           (main.py)
├── mlmp_continual.sh
├── cotta.sh
├── tent_continual.sh
├── tent_divgate_continual.sh
└── sar_continual.sh

bash/nighttime_driving/        # same 7 file names
bash/dz_nd_combined/           # same 7 file names
```

### 4.1 Per-folder shared variables

| Folder | DATASET | DATA_DIR | CORRUPTIONS_LIST | imgs/round |
|---|---|---|---|---|
| `bash/dark_zurich/` | `DarkZurichDataset` | `data/Dark_Zurich_val_anon/` | `night` | 50 |
| `bash/nighttime_driving/` | `NighttimeDrivingDataset` | `data/NighttimeDrivingTest/` | `night` | 50 |
| `bash/dz_nd_combined/` | `DZ_ND_Combined` | `data/` (unused) | `dark_zurich nighttime_driving` | 100 |

### 4.2 Per-script body

Every script is created by copying the matching file from [bash/ACDC_10_round/](../bash/ACDC_10_round/) and applying the substitutions in §4.1. All other hyperparameters (`LR`, `STEPS`, `BATCH_SIZE`, method-specific knobs like `H_THRESHOLD` / `CAUTIOUS_RST` / `SAM_RHO`) are copied **verbatim** — these are the values tuned for native shift on ACDC and should transfer best to DZ/ND.

`CONTINUAL_ROUNDS=150` for all continual scripts (matches existing ACDC convention; user noted "150 round 先做好了 不過這應該不是什麼大問題 我要跑的時候自己改的數字就好").

`SAVE_DIR=save/${DATASET}/${METHOD}/` follows existing convention; produces:
- `save/DarkZurichDataset/{method}/`
- `save/NighttimeDrivingDataset/{method}/`
- `save/DZ_ND_Combined/{method}/`

### 4.3 Stream sequencing for combined

Within each round of `bash/dz_nd_combined/`, conditions iterate in declaration order:

```
Round N: [50 DZ images, block] → [50 ND images, block]
```

This matches ACDC's `fog → night → rain → snow` block-by-block sequencing. No interleaving within a round. `shuffle=False` (existing default) preserves deterministic image order inside each block.

### 4.4 CoTTA `RST` parameter

CoTTA's `RST` hyperparam (stochastic restoration probability) is its main stability mechanism. The matching ACDC script uses `RST=0.01`. **This is the value to copy** — the user previously hit the `RST=0` bug on `bash/v20_acdc_matched/cotta.sh` (run produced all-zero output by round 150). For all three new CoTTA scripts: `RST=0.01`.

### 4.5 mlmp_episodic (uses `main.py`, not `main_continual.py`)

Single-condition datasets (DZ alone, ND alone) produce:
- `save/DarkZurichDataset/mlmp/00_night/results.txt`
- `save/NighttimeDrivingDataset/mlmp/00_night/results.txt`

Combined produces:
- `save/DZ_ND_Combined/mlmp/00_dark_zurich/results.txt`
- `save/DZ_ND_Combined/mlmp/01_nighttime_driving/results.txt`

Episodic semantics (`reset() → adapt() → evaluate()` per sample) are unchanged. The combined output gives two independent per-sub-dataset episodic upper bounds.

---

## 5. Output File Format

### 5.1 Continual methods

`results_all_rounds.txt`, comma-separated.

| Folder | Header | Example row |
|---|---|---|
| DZ alone | `Round, night, Mean_mIoU` | `Round 01, 24.31, 24.31` |
| ND alone | `Round, night, Mean_mIoU` | `Round 01, 22.18, 22.18` |
| Combined | `Round, dark_zurich, nighttime_driving, Mean_mIoU` | `Round 01, 24.31, 22.18, 23.25` |

`Mean_mIoU` is the arithmetic mean of per-condition mIoU within the round (matches existing convention).

### 5.2 Episodic (mlmp)

Per-condition subdirectory under `save/{dataset}/mlmp/`, each containing `results.txt` formatted `<mIoU>, <mDice>, <mAcc>` (existing convention).

### 5.3 Divgate log (TENT-DivGate only)

`save/{dataset}/{method}/divgate_log.txt` — mode transitions logged every `monitor_interval` batches. Existing format unchanged.

---

## 6. Hyperparameters (copy from ACDC)

All seven methods inherit hyperparams from the corresponding `bash/ACDC_10_round/` script. Listed here for reference; **do not re-tune**.

| Method | Key hyperparameters (source: bash/ACDC_10_round/) |
|---|---|
| no_adapt | `batch_size=1`, no `--adapt` flag |
| mlmp_episodic | `LR=1e-3`, `STEPS=10`, `TRIALS=1`, episodic mode (`main.py`) |
| mlmp_continual | `LR=5e-6`, `STEPS=1`, batch_size=1 |
| cotta | `LR=1e-5`, `STEPS=1`, `MT=0.999`, `RST=0.01`, `AP=0.92`, `AUG_N=32` |
| tent_continual | `LR=1e-5`, `STEPS=1`, batch_size=1 |
| tent_divgate_continual | `LR=1e-5`, `STEPS=1`, `H_THRESHOLD=1.6`, `H_WARNING=1.4`, `CAUTIOUS_RST=0.01`, `BRAKE_RST=0.05`, `MONITOR_INTERVAL=50` |
| sar_continual | `LR=1e-5`, `STEPS=1`, `e_margin=1.8`, `sam_rho=0.05`, `E_0=0.1`, `ema_factor=0.9`, `recovery_warmup=50` |

GPU id per script is **copied from the matching `bash/ACDC_10_round/` file** (varies: typically 1/2/3 across methods); user adjusts inline if needed.

---

## 7. Files Created / Modified

| Action | Path | Purpose |
|---|---|---|
| Modify | [utils/segmentation_datasets.py](../utils/segmentation_datasets.py) | Add 2 dataset classes, 2 mmseg configs, 3 dispatch branches, native-shift whitelist tuple |
| Create | `bash/dark_zurich/no_adapt.sh` | |
| Create | `bash/dark_zurich/mlmp_episodic.sh` | uses main.py |
| Create | `bash/dark_zurich/mlmp_continual.sh` | |
| Create | `bash/dark_zurich/cotta.sh` | RST=0.01 |
| Create | `bash/dark_zurich/tent_continual.sh` | |
| Create | `bash/dark_zurich/tent_divgate_continual.sh` | h_thr=1.6, cau_rst=0.01 |
| Create | `bash/dark_zurich/sar_continual.sh` | E_0=0.1 |
| Create | `bash/nighttime_driving/*.sh` | same 7 files |
| Create | `bash/dz_nd_combined/*.sh` | same 7 files |

**No new adapt files. No new model files.** The whole feature is one Python file diff + 21 bash scripts.

---

## 8. Sanity Checks

1. **Dataset registration**: `from utils.segmentation_datasets import DarkZurichDataset, NighttimeDrivingDataset` succeeds; `'DarkZurichDataset' in DATASETS.module_dict` is True.
2. **Image discovery**: `prepare_data('DarkZurichDataset', 'data/Dark_Zurich_val_anon/', ...)` → `len(loader.dataset) == 50`. Same for ND.
3. **Combined dispatch**: `prepare_data('DZ_ND_Combined', 'data/', ..., corruption='dark_zurich')` → 50 DZ samples; `corruption='nighttime_driving'` → 50 ND samples; `corruption='night'` raises `ValueError`.
4. **No CorruptTransform inserted** for any of `DarkZurichDataset`, `NighttimeDrivingDataset`, `DZ_ND_Combined`: pipeline length unchanged (4 transforms, no `CorruptTransform`).
5. **GT trainIds in [0, 18] ∪ {255}** for a random sample from each dataset (verifies label-space alignment with Cityscapes 19 classes).
6. **End-to-end smoke**: `bash bash/dark_zurich/no_adapt.sh` (with `CONTINUAL_ROUNDS=1` override) completes one round and writes `results_all_rounds.txt` with 1 row, columns `Round, night, Mean_mIoU`.
7. **End-to-end smoke combined**: same for `bash/dz_nd_combined/no_adapt.sh` — `results_all_rounds.txt` has columns `Round, dark_zurich, nighttime_driving, Mean_mIoU`.
8. **Hyperparam parity with ACDC**: `diff bash/ACDC_10_round/tent_divgate_continual.sh bash/dark_zurich/tent_divgate_continual.sh` shows only the three documented variable changes (DATASET, DATA_DIR, CORRUPTIONS_LIST) and `SAVE_DIR` derivative.

---

## 9. Success Criteria

This is infrastructure, not a research method. Success = scripts run, produce expected files, and remain comparable to ACDC.

| Tier | Criterion | Meaning |
|---|---|---|
| Basic | All 21 scripts run one round without crash | Dataset wiring + dispatch + bash plumbing all work |
| Target | 7-method × 150-round runs complete for all 3 folders | Folder operationally usable for cross-dataset analysis |
| Ideal | Trajectory overlay (DZ alone, ND alone, Combined, ACDC) reveals whether single-domain vs cross-condition drift differs | Headroom-conditional stability hypothesis tested on three new streams |

---

## 10. Out of Scope

- **COCO Object / COCO Stuff** scripts — blocked on mask conversion (separate spec).
- **Cross-dataset trajectory overlay plot** — written after data exists.
- **Reverse-order combined stream** (ND→DZ) — only DZ→ND will be implemented; reverse can be added later if ordering effects are suspected.
- **Larger round count for low-image streams** — user explicitly opted for 150 across the board, will adjust per-script if needed.
- **Hyperparameter re-tuning for DZ/ND** — ACDC values are copied verbatim. Tuning on DZ/ND is a separate research task once these baseline runs exist.
- **DZ test split** (151 images, GT on server) — only public val GT is used; submitting to the server is out of scope.

---

*End of design.*
