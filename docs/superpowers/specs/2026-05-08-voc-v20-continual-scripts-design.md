# Design: PascalVOC20Dataset 150-Round Continual Scripts (`bash/v20/`)

**Date**: 2026-05-08
**Status**: approved, pending implementation
**Goal**: Provide a complete set of CTTA experiment scripts for the PascalVOC20Dataset (v20) under the same 150-round continual protocol used for ACDC and Cityscapes. Establish v20 as a third generalisation target for TENT-DivGate alongside ACDC (real adverse weather) and Cityscapes (synthetic ImageNet-C).

---

## 1. Context

The v20 dataset is `PascalVOC20Dataset` defined in [utils/segmentation_datasets.py:256](utils/segmentation_datasets.py#L256). VOC2012 val split has 1449 images. Native image size is around 500×500. Images are real photographs with no built-in corruptions, so we apply 15 ImageNet-C corruptions on-the-fly via `CorruptTransform` (severity=5), identical to the Cityscapes setup.

A previous `bash/v20/` directory existed (commit 2dd84c3, deleted in current branch). It used:
- `INIT_RESIZE="224 224"` (single-patch evaluation)
- `DATA_DIR=".data/VOC2012/"` (old path; current data lives at `data/VOC/VOC2012/`)
- Per-method ad-hoc batch sizes (tent batch=64, mlmp batch=2)
- Mixed LR conventions (1e-4 / 1e-5)

This design replaces those with the **current Phase-2 convention** (mirrors `bash/cityscapes_continual/`) so v20 results are directly comparable to the latest ACDC and Cityscapes runs.

---

## 2. Critical Convention: Patch Configuration (★ EXPERIMENT RECORD ITEM)

**This is the single most important experimental setting to record for v20 — it directly determines per-image compute, mIoU ceiling, and comparability across runs.**

| Setting | Value | Effect |
|---|---|---|
| `INIT_RESIZE` | `"448 448"` | Image is upsampled to 448×448 before patching |
| `PATCH_SIZE` | `"224 224"` | Each forward pass operates on 224×224 |
| `PATCH_STRIDE` | `112` | Half-overlap sliding window |
| → patches per image | **3 × 3 = 9** | (448 − 224)/112 + 1 = 3 along each axis |

**Why 448×448 (not 224 single-patch, not native ~500×500)**:
- Single-patch (224×224) loses spatial detail — the entire image is one CLIP forward pass with no patch averaging. This was the deleted v20 default; rejected because OVSS pipelines benefit from patched evaluation.
- Native size (~500×500, no resize) gives a non-uniform patch count per image (depends on actual size), making batches statistically unstable and complicating compute estimates.
- 448×448 with stride=112 gives a clean, fixed 3×3 grid per image and matches Cityscapes' multi-patch convention. Per-image compute is ≈9× single-patch but still tractable.

**Compute estimate**: 1449 images × 9 patches × 15 corruptions × 150 rounds = ~29.3M patch forward passes per method. At ViT-L/14 throughput of ~50 patches/s/GPU, this is ≈160 GPU-hours per full-15 method. Caching halves rounds 2+ I/O cost.

**This setting MUST be reported in any v20 result table or paper text.** Different patch settings produce different absolute mIoU numbers and are not comparable.

---

## 3. Deliverables

Six bash scripts under `bash/v20/`:

| Script | Adaptation method | Entry point | Purpose |
|---|---|---|---|
| `no_adapt.sh` | none (calls `tent_continual` without `--adapt`) | `main_continual.py` | Source-model baseline; constant per round |
| `tent_continual.sh` | `tent_continual` | `main_continual.py` | Naive TENT continual baseline |
| `tent_divgate_continual.sh` | `tent_divgate_continual` | `main_continual.py` | **Primary method** — best ACDC config |
| `mlmp_continual.sh` | `mlmp_continual` | `main_continual.py` | Naive MLMP continual baseline |
| `mlmp_episodic.sh` | `mlmp` (episodic) | `main.py` | Per-sample reset upper bound |
| `cotta.sh` | `cotta` | `main_continual.py` | Stable-but-flat baseline |

**No Python code changes required.** All entry points and methods already support `PascalVOC20Dataset` ([main_continual.py:61](main_continual.py#L61), [main.py:69](main.py#L69)).

---

## 4. Common Configuration Block

Every script (with method-specific variations noted in §5) shares:

```bash
GPU_ID=0
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"          # ★ 9 patches per image — see design doc §2
PATCH_SIZE="224 224"
PATCH_STRIDE=112
WORKERS=4

OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
STEPS=1
SEED=0
CONTINUAL_ROUNDS=150
```

Each script also contains, near the top, a comment block reproducing the patch convention so it is visible without leaving the file:

```bash
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.
```

---

## 5. Corruption List

Mirrors `bash/cityscapes_continual/` exactly — 15 ImageNet-C corruptions in standard order, one-per-line, all uncommented by default:

```bash
CORRUPTIONS_ARRAY=(
    # --- noise ---
    gaussian_noise
    shot_noise
    impulse_noise
    # --- blur ---
    defocus_blur
    glass_blur
    motion_blur
    zoom_blur
    # --- weather ---
    snow
    frost
    fog
    brightness
    contrast
    # --- digital ---
    elastic_transform
    pixelate
    jpeg_compression
)
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"
```

**Subset override**: `CORRUPTIONS_LIST="fog snow frost brightness contrast" bash bash/v20/<script>.sh` — same convention as Cityscapes scripts.

---

## 6. Caching

The `prepare_data()` function ([utils/segmentation_datasets.py:501](utils/segmentation_datasets.py#L501)) auto-derives `cache_dir` from `data_dir`:

| `DATA_DIR` | Auto cache dir |
|---|---|
| `data/VOC/VOC2012/` | `data/VOC/.cache/corruptions/` |

Cache files use `<md5(img_path)>_<corruption>_s5.npy`. VOC image paths cannot collide with Cityscape paths, so the cache directory could even be shared safely — but auto-derivation already places it under `data/VOC/`. Round 1 generates and caches; rounds 2-150 load from cache (≈10× faster I/O than regeneration, especially for `glass_blur` which is the slow generator).

**Estimated cache size**: 1449 images × 15 corruptions × 448×448×3 bytes ≈ 13 GB total. (Smaller than Cityscapes' ~45 GB because VOC images are smaller.)

---

## 7. Per-Method Hyperparameters

All values mirror `bash/cityscapes_continual/`. ★ marks the items that differ from the default cityscape script of the same name.

### `no_adapt.sh`
- `METHOD=tent_continual` (any method works, since no `--adapt` flag)
- `LR=0.00001` (unused — no gradient steps)
- Run line: `python main_continual.py` **without** `--adapt`
- Expected output: constant mIoU across all 150 rounds (sanity check)

### `tent_continual.sh`
- `METHOD=tent_continual`
- `LR=0.00001`

### `tent_divgate_continual.sh` ★ (primary method)
- `METHOD=tent_divgate_continual`
- `LR=0.00001`
- `H_THRESHOLD=1.6` ← **ACDC-best, not the cityscape script's 2.0**
- `H_WARNING=1.4`
- `MONITOR_INTERVAL=50`
- `CAUTIOUS_RST=0.01`
- `BRAKE_RST=0.05`

**Why h_thr=1.6 not 2.0**: ACDC sweep (`save/ACDCDataset/tent_divgate_continual_cau_rst_0.01/`) confirmed 1.6 is best (mean=31.59). Cityscapes was switched to 2.0 because Cityscapes' H_margin distribution is shifted +0.25 nats higher, making 1.6 fire too rarely. We have no prior on VOC's H_margin distribution; starting from the validated ACDC config is the principled default. After R20-R30 we can read `divgate_log.txt` to decide whether to sweep.

### `mlmp_continual.sh`
- `METHOD=mlmp_continual`
- `LR=0.00001` (1e-5, matches cityscape `mlmp_continual.sh`)
- `OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"`
- `PROMPT_DIR=prompts.yaml`
- `ALPHA_CLS=1.0`

### `mlmp_episodic.sh`
- `METHOD=mlmp`
- Entry point `main.py` (NOT `main_continual.py`)
- `LR=0.001` (1e-3, higher because state resets per sample)
- `STEPS=1`, `TRIALS=1`
- Same `OUT_VISION`, `PROMPT_DIR`, `ALPHA_CLS` as `mlmp_continual.sh`
- No `--continual_rounds` (episodic)

### `cotta.sh`
- `METHOD=cotta`
- `LR=0.00001`
- `MT=0.999`, `RST=0.01`, `AP=0.92`, `AUG_N=32`

---

## 8. Save Directory Convention

Default per script:
```bash
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"
# e.g. save/PascalVOC20Dataset/tent_divgate_continual/
```

Override pattern (matches cityscape convention):
```bash
SAVE_DIR="save/PascalVOC20Dataset/tent_divgate_continual_hthr1.8/" \
H_THRESHOLD=1.8 \
  bash bash/v20/tent_divgate_continual.sh
```

---

## 9. Output Files

Identical to Cityscapes:
- `save/PascalVOC20Dataset/<method>/results_all_rounds.txt` — one row per round; columns = each corruption + Mean_mIoU
- `save/PascalVOC20Dataset/<method>/divgate_log.txt` (DivGate only) — H_margin trajectory, one row per `monitor_interval=50` batches
- Per-round per-condition `results.txt` under `round_NN/<corruption>/`

---

## 10. Out of Scope

- **v21** (PascalVOC21Dataset) — explicitly deferred to a follow-up; user said "先做v20"
- **`mlmp_divgate_continual.sh`** — user did not list it
- **Result parser** — `parse_acdc_results.py` is ACDC-specific; a v20 parser can be added later
- **Plot scripts** — none in this iteration
- **Python code changes** — not needed; all dataset/method support is already in place

---

## 11. Validation Checklist (post-implementation)

Before declaring the scripts complete:
1. `bash bash/v20/no_adapt.sh` runs at least 2 rounds and produces a results file.
2. The results file mIoU is **constant** across rounds (no_adapt is a sanity check that data flow works correctly).
3. Cache files appear at `data/VOC/.cache/corruptions/` after round 1.
4. Round 2 is significantly faster than round 1 (cache hit).
5. `tent_divgate_continual.sh` produces `divgate_log.txt` with at least one mode-transition line.

---

## 12. Experiment Record Convention (★)

**For every v20 result included in any paper / table / discussion, the following MUST be recorded next to the mIoU number**:

> v20 patch convention: `INIT_RESIZE=448x448, patch=224x224, stride=112` (9 patches/image)

This will be added to the project memory after the scripts land so that future sessions automatically include it in result discussions.
