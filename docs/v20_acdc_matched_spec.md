# `bash/v20_acdc_matched/` — VOC20 Bash Folder with ACDC-Matched Round Size

**Status**: Design approved, implementation pending
**Created**: 2026-05-21
**Owner**: Tekai-Yen
**Related docs**: `docs/EXPERIMENT_STATUS.md` (cross-dataset framing), existing folders `bash/v20/` and `bash/ACDC_10_round/`

---

## 1. Motivation

ACDC and VOC20 are not directly comparable in their current bash setups: a "round" means very different amounts of work and very different data exposure.

| Dataset | Conditions per round | Images per condition | **Images per round** |
|---|---|---|---|
| ACDC | 4 (fog/night/rain/snow) | 100 / 106 / 100 / 100 | **406** |
| VOC20 weather (`bash/v20/`) | 5 (snow/frost/fog/brightness/contrast) | 1449 (full val set) | **7245** |

VOC20 currently sees **18× more data per round** than ACDC. Two consequences:

- **Wall time per round**: VOC20 takes ~18× longer per round than ACDC at the same per-batch cost. 150-round runs are not practical on full v20.
- **Adaptation pressure**: more data per round = more gradient updates per round, so a method that's "stable to round 150 on ACDC" may collapse at round 8 on VOC20 simply because it sees 18× the gradient steps. Cross-dataset comparison of stability claims becomes meaningless.

`bash/v20/` was designed for absolute VOC20 performance numbers (where seeing all 1449 images matters). This new folder is designed for **cross-dataset comparability**: same number of round-equivalent batches, so trajectories can be overlaid directly.

---

## 2. Design Principle

**Match ACDC's 406 images per round exactly.** With 4 corruptions (matching ACDC's 4 conditions), that's ~101 images per corruption.

Subsampling is done via a **pre-generated deterministic split file** that lists 101 random VOC20 val image IDs. The same file is used by every method in this folder, so all method runs see the exact same 101 images per corruption — direct trajectory comparison.

`prepare_data()` already builds the dataset via mmseg's `DATASETS.build(mm_config)`, where `mm_config['ann_file']` points to a split file. Override that field via a new `--ann_file` CLI flag and the whole thing works without touching the dataset class.

---

## 3. Per-Round Size Decision

ACDC totals 406 images/round, unevenly split (100/106/100/100). Possible matching strategies for VOC20:

| Strategy | Per corruption | Total / round | Notes |
|---|---|---|---|
| **Uniform 101** | 101 × 4 | **404** | 0.5% under ACDC, simple, default |
| Uniform 102 | 102 × 4 | 408 | 0.5% over |
| Mirror ACDC counts | 100/106/100/100 | 406 | Requires 4 different split files; muddies "subset" concept |
| Round 100 | 100 × 4 | 400 | -1.5%, clean number |

**Decision: 101 per corruption, uniform.** Total 404 ≈ ACDC's 406. One split file. Parameter `IMAGES_PER_CORRUPTION=101` is exposed in every bash script for override.

---

## 4. Subsetting Mechanism

### 4.1 Subset file generation (one-shot helper)

`scripts/make_voc_subset.py` — generate a deterministic split file by random-sampling `N` image IDs from `data/VOC/VOC2012/ImageSets/Segmentation/val.txt`.

```python
#!/usr/bin/env python3
"""
Generate a deterministic VOC val subset split file.

Usage: python scripts/make_voc_subset.py --n 101 --seed 0
  → writes data/VOC/VOC2012/ImageSets/Segmentation/val_subset_101_seed0.txt
"""
import argparse, os, random

VOC_ROOT = "data/VOC/VOC2012"
SPLIT_DIR = os.path.join(VOC_ROOT, "ImageSets", "Segmentation")
FULL_VAL = os.path.join(SPLIT_DIR, "val.txt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="Number of images to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full_split", default=FULL_VAL,
                   help="Source split file (default: VOC val.txt)")
    args = p.parse_args()

    with open(args.full_split) as f:
        ids = [l.strip() for l in f if l.strip()]
    if args.n > len(ids):
        raise ValueError(f"n={args.n} exceeds source size {len(ids)}")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(ids, args.n))  # sort for stable file ordering

    out = os.path.join(SPLIT_DIR, f"val_subset_{args.n}_seed{args.seed}.txt")
    with open(out, "w") as f:
        f.write("\n".join(sampled) + "\n")
    print(f"Wrote {out} ({args.n} images, seed={args.seed})")

if __name__ == "__main__":
    main()
```

Determinism: `random.Random(seed)` + `sorted()` → bit-identical file across invocations and machines.

**Output path convention**: `data/VOC/VOC2012/ImageSets/Segmentation/val_subset_{N}_seed{SEED}.txt`. Bash scripts derive this path from `IMAGES_PER_CORRUPTION` and `SEED` variables.

### 4.2 `prepare_data` change

Add optional `ann_file` parameter. When supplied, override `mm_config['ann_file']` for VOC20/VOC21 datasets. No effect on ACDC/Cityscapes/COCO (those don't use ann_file).

```python
def prepare_data(dataset, data_dir, init_resize, patch_size, patch_stride,
                 corruption="original", batch_size=128, num_workers=1, shuffle=True,
                 corruption_cache_dir=None,
                 ann_file=None):                              # ← new
    ...
    elif dataset == "PascalVOC20Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc20_cfg)
    ...
    # near the end, before DATASETS.build:
    if ann_file is not None:
        if dataset not in ("PascalVOC20Dataset", "PascalVOC21Dataset"):
            print(f"+++ ann_file override ignored: {dataset} does not use ann_file")
        else:
            mm_config['ann_file'] = ann_file
            print(f"+++ ann_file overridden -> {ann_file}")
```

### 4.3 CLI arg in `main_continual.py` and `main.py`

Add to the shared argparser block in **both** files:

```python
parser.add_argument('--ann_file', type=str, default=None,
                    help='Override dataset ann_file (e.g., a VOC subset split). '
                         'Only affects PascalVOC20Dataset/PascalVOC21Dataset.')
```

Pass through to `prepare_data` at the **stream loader** call sites only (not source-stat loaders, which need the clean unmodified source data):

**`main_continual.py` modifications**:
- Line ~396 (`first_loader`): add `ann_file=args.ann_file`
- Line ~510 (`data_loader` inside round loop): add `ann_file=args.ann_file`
- **Do NOT add** to src_loader call sites at lines 431 / 455 / 477 (EATA / CMA-Proto / DPCore source-stat loaders — they use `corruption='original'` and should see the full source distribution)

**`main.py` modifications**:
- Line ~359 (`_first_loader`): add `ann_file=args.ann_file`
- Line ~384 (`data_loader` episodic): add `ann_file=args.ann_file`
- Line ~521 (alternate `data_loader` site): add `ann_file=args.ann_file`
- **Do NOT add** to src_loader call sites at lines 377 / 550

---

## 5. Folder Structure

```
bash/v20_acdc_matched/
├── _README.md                          # (optional) — purpose, override conventions
├── no_adapt.sh                         # tent_continual without --adapt flag
├── tent_continual.sh
├── mlmp_continual.sh
├── cotta.sh
├── mlmp_episodic.sh                    # uses main.py, not main_continual.py
├── mlmp_divgate_continual.sh
├── tent_divgate_continual.sh
└── sar_continual.sh
```

Eight scripts total. All except `mlmp_episodic.sh` use `main_continual.py`.

---

## 6. Shared Bash Template

Every script in this folder shares this header block (variables at top, identical across scripts):

```bash
#!/bin/bash
# {METHOD} on PascalVOC20Dataset with ACDC-matched round size.
# Each round samples IMAGES_PER_CORRUPTION images per corruption (default 101),
# matching ACDC's ~406 images/round for direct cross-dataset comparison.
# See docs/v20_acdc_matched_spec.md.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=2

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

# ── Subset (matches ACDC's 406 imgs/round; 4 corruptions × 101 = 404) ─
IMAGES_PER_CORRUPTION=101
SUBSET_SEED=0
ANN_FILE="${DATA_DIR}ImageSets/Segmentation/val_subset_${IMAGES_PER_CORRUPTION}_seed${SUBSET_SEED}.txt"

# Auto-generate the subset file if missing (one-shot, deterministic).
if [ ! -f "$ANN_FILE" ]; then
    echo "+++ Subset file missing, generating: $ANN_FILE"
    python scripts/make_voc_subset.py --n $IMAGES_PER_CORRUPTION --seed $SUBSET_SEED
fi

# ── Corruption conditions (pick exactly 4 — comment out 11 of 15) ──
CORRUPTIONS_ARRAY=(
    # --- noise ---
    # gaussian_noise
    # shot_noise
    # impulse_noise
    # --- blur ---
    # defocus_blur
    # glass_blur
    # motion_blur
    # zoom_blur
    # --- weather ---
    snow
    frost
    fog
    brightness
    # contrast              ← only 4 enabled by default
    # --- digital ---
    # elastic_transform
    # pixelate
    # jpeg_compression
)
# One-liner subset override: CORRUPTIONS_LIST="snow frost fog brightness" bash ...
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="{METHOD_KEY}"                   # ← per-script
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters (match bash/v20/{METHOD}.sh) ──────────
# ... per-script hyperparams ...

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/v20_acdc_matched/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        ...
                        --ann_file $ANN_FILE \
                        --corruptions_list $CORRUPTIONS_LIST \
                        ...
```

Default 4 corruptions: **snow, frost, fog, brightness** (matches ACDC's "weather" theme). User comments lines in `CORRUPTIONS_ARRAY` to pick a different 4-set, or uses `CORRUPTIONS_LIST` env var.

---

## 7. Per-Script Details

Each script is a verbatim copy of the corresponding `bash/v20/{METHOD}.sh` with these substitutions:

1. **Header comment** updated to reference v20_acdc_matched purpose
2. **Subset block** inserted after `WORKERS=1`
3. **Corruption array** trimmed to 4 default-enabled lines (snow/frost/fog/brightness)
4. **`SAVE_DIR`** changed from `save/${DATASET}/${METHOD}/` to `save/${DATASET}/v20_acdc_matched/${METHOD}/`
5. **`--ann_file $ANN_FILE`** added to the python invocation
6. All method-specific args (LR, STEPS, method-specific knobs like `H_THRESHOLD`, `SAM_RHO`, `MT/RST/AP/AUG_N`, etc.) copied **verbatim** from the corresponding `bash/v20/` script

| Script | Method key | Source (copy from) | Notes |
|---|---|---|---|
| `no_adapt.sh` | tent_continual | `bash/v20/no_adapt.sh` | omits `--adapt` flag |
| `tent_continual.sh` | tent_continual | `bash/v20/tent_continual.sh` | |
| `mlmp_continual.sh` | mlmp_continual | `bash/v20/mlmp_continual.sh` | LR=5e-6, has --vision_outputs / --prompt_dir |
| `cotta.sh` | cotta | `bash/v20/cotta.sh` | mt/rst/ap/aug_n |
| `mlmp_episodic.sh` | mlmp | `bash/v20/mlmp_episodic.sh` | **uses `main.py`**, LR=1e-3, has `--trials` |
| `mlmp_divgate_continual.sh` | mlmp_divgate_continual | `bash/v20/mlmp_divgate_continual.sh` | h_thr=1.6 + MLMP args |
| `tent_divgate_continual.sh` | tent_divgate_continual | `bash/v20/tent_divgate_continual.sh` | h_thr=1.6, cau_rst=0.01 |
| `sar_continual.sh` | sar_continual | `bash/v20/sar_continual.sh` | e_margin=1.8, sam_rho=0.05, e_0=0.1 |

Episodic note: `mlmp_episodic.sh` calls `main.py` (not `main_continual.py`) and produces per-condition `results.txt` files — not `results_all_rounds.txt`. The `--ann_file` arg still applies (added to main.py argparser per §4.3).

---

## 8. Hyperparameters

| Parameter | Default | Override mechanism |
|---|---|---|
| `IMAGES_PER_CORRUPTION` | 101 | Inline edit (also requires re-running `make_voc_subset.py` for new N) |
| `SUBSET_SEED` | 0 | Inline edit (also re-run `make_voc_subset.py`) |
| `CONTINUAL_ROUNDS` | 150 | Inline edit |
| Default 4 corruptions | snow, frost, fog, brightness | Comment lines in `CORRUPTIONS_ARRAY` or set `CORRUPTIONS_LIST` env var |
| `SAVE_DIR` | `save/{DATASET}/v20_acdc_matched/{METHOD}/` | env var |
| GPU | 2 | Inline edit per script |
| All method-specific hyperparams | unchanged from `bash/v20/{METHOD}.sh` | Inline edit |

---

## 9. Files Created / Modified

| Action | Path | Purpose |
|---|---|---|
| Create | `scripts/make_voc_subset.py` | Generate deterministic VOC subset split file |
| Modify | `utils/segmentation_datasets.py` | Add `ann_file` kwarg to `prepare_data()` |
| Modify | `main_continual.py` | Add `--ann_file` arg + pass-through at 2 stream-loader sites |
| Modify | `main.py` | Add `--ann_file` arg + pass-through at 3 stream-loader sites |
| Create | `bash/v20_acdc_matched/no_adapt.sh` | |
| Create | `bash/v20_acdc_matched/tent_continual.sh` | |
| Create | `bash/v20_acdc_matched/mlmp_continual.sh` | |
| Create | `bash/v20_acdc_matched/cotta.sh` | |
| Create | `bash/v20_acdc_matched/mlmp_episodic.sh` | uses main.py |
| Create | `bash/v20_acdc_matched/mlmp_divgate_continual.sh` | |
| Create | `bash/v20_acdc_matched/tent_divgate_continual.sh` | |
| Create | `bash/v20_acdc_matched/sar_continual.sh` | |

**No new adapt classes** — all 8 methods exist already.
**No new dataset class** — VOC20 with custom ann_file is the entire data-side mechanism.

---

## 10. Sanity Checks

1. **Subset file generation is deterministic**: `python scripts/make_voc_subset.py --n 101 --seed 0` produces a bit-identical file on re-run; `md5sum` matches across machines.
2. **Subset size matches CLI N**: `wc -l val_subset_101_seed0.txt` returns 101.
3. **Subset IDs are valid VOC val IDs**: every line appears in `data/VOC/VOC2012/ImageSets/Segmentation/val.txt`.
4. **`prepare_data` honours the override**: after a small script run with `ann_file=val_subset_101_seed0.txt`, `len(data_loader.dataset)` should be 101.
5. **`prepare_data` warns on non-VOC datasets**: passing `ann_file=...` with `dataset='ACDCDataset'` prints the warning and proceeds normally (no crash).
6. **Source-stat loaders are NOT subsetted**: a CMA-Proto / EATA / DPCore run via `bash/v20_acdc_matched/...` (when those scripts exist — out of scope here, but the mechanism must support them) should still see the full source distribution.
7. **End-to-end smoke**: `bash bash/v20_acdc_matched/no_adapt.sh --debug` runs without error; first round writes `results_all_rounds.txt` with 1 row of 6 columns (Round, snow, frost, fog, brightness, Mean_mIoU).
8. **Total per round = `4 × IMAGES_PER_CORRUPTION`**: from the run log, batch count for the first round equals `4 × 101 = 404` (with batch_size=1).

---

## 11. Success Criteria

This is infrastructure, not a research method. Success = the folder exists, the mechanism works, and the methods produce comparable trajectories to ACDC.

| Tier | Criterion | Meaning |
|---|---|---|
| Basic | All 8 scripts run a single round without crash | Subsetting + bash plumbing work end-to-end |
| Target | 4-corruption × 150-round runs complete for all 8 methods | Folder is operationally usable |
| **Ideal** | Plot overlay of `bash/v20_acdc_matched/` trajectories vs `bash/ACDC_10_round/` trajectories shows cleanly-aligned rounds | Cross-dataset comparison enabled |

---

## 12. Out of Scope

- **Plotting script for cross-dataset overlay** — separate task once data exists; can extend `plot_acdc_4methods.py` to accept a second dataset
- **EATA / DPCore / CMA-family scripts** — not in user's requested 8 methods; out of scope. Future addition is straightforward (same template + same `--ann_file` plumbing)
- **Cityscapes-matched variant** — same principle applies (Cityscapes has 500 imgs/condition × 15 = 7500 by default), but separate spec
- **Mirroring ACDC's uneven per-condition counts** (100/106/100/100) — requires 4 different split files; defer until uniform 101 proves insufficient
- **`sar_divgate_continual.sh`** — exists in `bash/v20/`, could be added but user's list was specific (8 methods, no sar_divgate). Trivial 1-line addition later
- **Seed-sweep for subset robustness** (running with seed=1, seed=2 to check subset choice doesn't bias results) — defer until single-seed results justify the analysis

---

*End of design.*
