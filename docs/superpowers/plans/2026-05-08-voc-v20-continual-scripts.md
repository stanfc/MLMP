# PascalVOC20Dataset 150-Round Continual Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 bash scripts under `bash/v20/` that run 150-round CTTA experiments on PascalVOC20Dataset (1449 val images × 15 ImageNet-C corruptions) using the cityscape continual convention, with corruption caching enabled and a clearly recorded 448×448 / 9-patch evaluation setting.

**Architecture:** Pure bash scripts. No Python changes. Each script calls `main_continual.py` (or `main.py` for episodic MLMP) with the right CLI flags. Corruption caching is provided by `prepare_data()` automatically deriving `cache_dir = data/VOC/.cache/corruptions/` from `DATA_DIR=data/VOC/VOC2012/`. The patch convention (`INIT_RESIZE="448 448"`, `patch_size=224 224`, `patch_stride=112` → 3×3=9 patches/image) is documented in a comment block at the top of every script and recorded in project memory after sanity check passes.

**Tech Stack:** bash, existing PyTorch / NA-CLIP / MLMP code in this repo.

**Spec:** [docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md](../specs/2026-05-08-voc-v20-continual-scripts-design.md)

---

## File Structure

| File | Responsibility |
|---|---|
| `bash/v20/no_adapt.sh` | Source-model baseline (constant mIoU sanity check) |
| `bash/v20/tent_continual.sh` | Naive TENT continual baseline |
| `bash/v20/tent_divgate_continual.sh` | TENT-DivGate (primary method, ACDC-best config) |
| `bash/v20/mlmp_continual.sh` | Naive MLMP continual baseline |
| `bash/v20/mlmp_episodic.sh` | MLMP episodic upper bound (calls `main.py`) |
| `bash/v20/cotta.sh` | CoTTA stable-but-flat baseline |

Memory file (created in Task 8 after sanity check passes):

| File | Responsibility |
|---|---|
| `~/.claude/projects/-home-tekai324-MLMP/memory/voc_patch_convention.md` | Records the 448×448 / 9-patch convention as an experiment-record requirement |

---

## Convention Used Across All Scripts (reference)

This block is reproduced verbatim in every bash script (the patch comment is THE record-of-experiment item, per spec §2):

```bash
#!/bin/bash
# <one-line description matching cityscape sibling>
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.
```

Common variables (placed near the top of every script, dataset block):

```bash
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4
```

Common run-line tail (last block before the python invocation):

```bash
BATCH_SIZE=1
CONTINUAL_ROUNDS=150
SEED=0
```

Cache flag is **not** passed explicitly — `prepare_data()` derives it automatically. See [utils/segmentation_datasets.py:566](../../../utils/segmentation_datasets.py#L566).

---

## Task 1: Create `bash/v20/no_adapt.sh`

**Files:**
- Create: `bash/v20/no_adapt.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# No-Adaptation baseline on PascalVOC20Dataset (CTTA, N rounds).
# Runs off-the-shelf NA-CLIP without any weight updates.
# Results are constant across all rounds — zero-shot source model performance.
# Use this to establish the lower bound before comparing continual TTA methods.
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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
# One-liner subset override: CORRUPTIONS_LIST="fog snow" bash script.sh
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_continual"   # lightest runner; --adapt is omitted so no updates occur
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
BATCH_SIZE=1
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/No_Adaptation/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make executable**

```bash
chmod +x bash/v20/no_adapt.sh
```

- [ ] **Step 3: Lint with bash -n**

```bash
bash -n bash/v20/no_adapt.sh
```

Expected: no output (syntax OK).

- [ ] **Step 4: Commit**

```bash
git add bash/v20/no_adapt.sh
git commit -m "bash/v20: add no_adapt baseline script"
```

---

## Task 2: Create `bash/v20/tent_continual.sh`

**Files:**
- Create: `bash/v20/tent_continual.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# TENT-Continual baseline on PascalVOC20Dataset (CTTA, N rounds).
# Naive entropy minimization WITHOUT per-sample reset — no anti-forgetting mechanism.
# Expected to drift and eventually degrade over long continual runs.
# 15 ImageNet-C corruptions applied on-the-fly.
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make executable & lint**

```bash
chmod +x bash/v20/tent_continual.sh && bash -n bash/v20/tent_continual.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bash/v20/tent_continual.sh
git commit -m "bash/v20: add tent_continual script"
```

---

## Task 3: Create `bash/v20/tent_divgate_continual.sh`

**Files:**
- Create: `bash/v20/tent_divgate_continual.sh`

Uses ACDC best config: `h_thr=1.6, h_warn=1.4, cau_rst=0.01, brake_rst=0.05`.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# TENT-DivGate-Continual on PascalVOC20Dataset (CTTA, N rounds).
# 15 ImageNet-C corruptions applied on-the-fly; same gate mechanism
# as cityscape, hyperparameters set to ACDC-best.
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Diversity gate (ACDC-best confirmed) ───────────────────────────
H_THRESHOLD=1.6       # H_margin >= this        -> aggressive (rst=0)
H_WARNING=1.4         # h_warning <= H < h_thr  -> cautious; < h_warning -> brake
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make executable & lint**

```bash
chmod +x bash/v20/tent_divgate_continual.sh && bash -n bash/v20/tent_divgate_continual.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bash/v20/tent_divgate_continual.sh
git commit -m "bash/v20: add tent_divgate_continual script (ACDC-best config)"
```

---

## Task 4: Create `bash/v20/mlmp_continual.sh`

**Files:**
- Create: `bash/v20/mlmp_continual.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# MLMP-Continual on PascalVOC20Dataset (CTTA, N rounds).
# Multi-prompt entropy + ILE loss, model state NEVER reset between samples.
# 15 ImageNet-C corruptions applied on-the-fly.
#
# Expected behaviour: gradual performance decline over rounds (no anti-forgetting
# mechanism). Use as ablation baseline against cotta.sh and tent_divgate_continual.sh.
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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

# ── Method ─────────────────────────────────────────────────────────
METHOD="mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── MLMP multi-level: last 18 layers of ViT-L/14 ──────────────────
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make executable & lint**

```bash
chmod +x bash/v20/mlmp_continual.sh && bash -n bash/v20/mlmp_continual.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bash/v20/mlmp_continual.sh
git commit -m "bash/v20: add mlmp_continual script"
```

---

## Task 5: Create `bash/v20/mlmp_episodic.sh`

**Files:**
- Create: `bash/v20/mlmp_episodic.sh`

This script calls `main.py` (NOT `main_continual.py`). Per-sample reset → adapt → evaluate. No `--continual_rounds` flag.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# MLMP (episodic) on PascalVOC20Dataset.
# Standard episodic TTA: reset -> adapt -> evaluate for every sample.
# Model state does NOT carry over between samples — per-sample upper bound.
# 15 ImageNet-C corruptions applied on-the-fly; uses main.py (not main_continual.py).
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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

# ── Method ─────────────────────────────────────────────────────────
METHOD="mlmp"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── MLMP multi-level: last 18 layers of ViT-L/14 ──────────────────
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# ── Training hyperparameters ───────────────────────────────────────
# Higher LR than continual — safe because state resets every sample.
BATCH_SIZE=1
LR=0.001
STEPS=1
TRIALS=1

# ── Experiment ─────────────────────────────────────────────────────
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/mlmp_episodic/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make executable & lint**

```bash
chmod +x bash/v20/mlmp_episodic.sh && bash -n bash/v20/mlmp_episodic.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bash/v20/mlmp_episodic.sh
git commit -m "bash/v20: add mlmp_episodic upper-bound script"
```

---

## Task 6: Create `bash/v20/cotta.sh`

**Files:**
- Create: `bash/v20/cotta.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# CoTTA on PascalVOC20Dataset (CTTA, N rounds) with NA-CLIP backbone.
# Open-vocabulary segmentation: class names are text prompts, no closed-set head.
# 15 ImageNet-C corruptions applied on-the-fly.
#
# CoTTA three mechanisms:
#   1. EMA teacher (mt=0.999) for stable pseudo-labels
#   2. Augmentation-averaged pseudo-labels when anchor confidence < ap
#   3. Stochastic restoration (rst=0.01) to prevent catastrophic forgetting
#
# ─── Patch convention (DO NOT CHANGE without noting in result file) ───
# INIT_RESIZE 448x448 + patch 224x224 stride 112 → 3x3=9 patches/image.
# This is THE comparable v20 setting; results from other patch settings
# are not directly comparable. See
# docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="448 448"
WORKERS=4

# ── Corruption conditions (ImageNet-C standard order) ──────────────
# Comment out individual lines to run a subset.
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

# ── Method ─────────────────────────────────────────────────────────
METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── CoTTA hyperparameters (matching original CoTTA paper values) ────
MT=0.999        # EMA smoothing factor for teacher
RST=0.01        # stochastic restoration probability
AP=0.92         # anchor confidence threshold (augment when mean conf < AP)
AUG_N=32        # number of augmented teacher views

# Use last layer only (standard CoTTA spirit — no multi-level fusion)
OUT_VISION="-1"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

# ───────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --mt $MT \
                        --rst $RST \
                        --ap $AP \
                        --aug_n $AUG_N \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make executable & lint**

```bash
chmod +x bash/v20/cotta.sh && bash -n bash/v20/cotta.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bash/v20/cotta.sh
git commit -m "bash/v20: add cotta script"
```

---

## Task 7: Sanity check — run `no_adapt.sh` for ≥2 rounds

This validates: (a) data flow works end-to-end on VOC, (b) corruption cache is created and reused, (c) results format matches expectations, (d) no_adapt produces constant mIoU across rounds.

**Files:**
- Read-only check; produces `save/PascalVOC20Dataset/No_Adaptation_sanity/results_all_rounds.txt`

- [ ] **Step 1: Pick a small corruption subset to keep round-1 cache build short**

The full 15-corruption × 1449-image cache build for round 1 may take a long time (esp. `glass_blur`). Use a 2-corruption subset for sanity. Run:

```bash
CORRUPTIONS_LIST="brightness fog" \
SAVE_DIR="save/PascalVOC20Dataset/No_Adaptation_sanity/" \
bash bash/v20/no_adapt.sh
```

- [ ] **Step 2: Watch for cache directory creation**

In another terminal, after the run starts, verify:

```bash
ls data/VOC/.cache/corruptions/ | head -5
```

Expected: at least one `.npy` file appears within the first few minutes (file pattern `<md5>_brightness_s5.npy` or `<md5>_fog_s5.npy`).

- [ ] **Step 3: Wait for the run to reach Round 02**

The run will write a row to `results_all_rounds.txt` after each round. Monitor:

```bash
tail -f save/PascalVOC20Dataset/No_Adaptation_sanity/results_all_rounds.txt
```

Stop the run (Ctrl-C) once Round 02 has been logged. Two rounds is sufficient — we just need to confirm cache reuse and constant mIoU.

- [ ] **Step 4: Verify results file format and content**

```bash
cat save/PascalVOC20Dataset/No_Adaptation_sanity/results_all_rounds.txt
```

Expected format (numbers will differ based on actual mIoU):

```
Round 01, <miou_brightness>, <miou_fog>, <mean_miou>
Round 02, <miou_brightness>, <miou_fog>, <mean_miou>
```

**Pass criteria**:
1. Two `Round NN, ...` rows present.
2. Round 01 and Round 02 mIoU values are **identical** (no_adapt → no weight updates → deterministic results).
3. Round 02 wall-clock time (visible in stdout) is significantly shorter than Round 01 (cache hit on round 2).

If criteria fail, fix the script and re-run before proceeding.

- [ ] **Step 5: Verify cache files exist**

```bash
ls data/VOC/.cache/corruptions/ | wc -l
```

Expected: ≥ 1449 × 2 = 2898 files (one per image per corruption used). Slightly fewer if the run was interrupted before all images were processed in round 1, but rounds 2+ would have used cached files for the images already processed.

- [ ] **Step 6: Commit nothing (sanity check produces no source changes)**

This task is verification-only. The save dir is gitignored (under `save/`).

---

## Task 8: Save the patch convention to project memory

**Files:**
- Create: `~/.claude/projects/-home-tekai324-MLMP/memory/voc_patch_convention.md`
- Modify: `~/.claude/projects/-home-tekai324-MLMP/memory/MEMORY.md`

This task is mandated by spec §12 — "★ EXPERIMENT RECORD ITEM" — and must run only after Task 7 sanity check passes (writing it earlier risks recording an unverified convention).

- [ ] **Step 1: Create the memory file**

Write to `/home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/voc_patch_convention.md`:

```markdown
---
name: VOC v20/v21 patch convention for 150-round continual experiments
description: Required INIT_RESIZE / patch_size / patch_stride for v20 (and v21) continual TTA results to be comparable
type: project
---
For PascalVOC20Dataset (and PascalVOC21Dataset) 150-round continual experiments, all bash scripts under `bash/v20/` use:

- INIT_RESIZE = "448 448"
- patch_size = 224 224
- patch_stride = 112
- → 3 × 3 = 9 patches per image

**Why:** This is THE comparable v20 setting. Different patch settings produce different absolute mIoU numbers and are NOT comparable. The previous (deleted) `bash/v20/` used INIT_RESIZE="224 224" (single patch); we deliberately switched to 448×448 to match the cityscape multi-patch convention and to retain spatial detail.

**How to apply:** Whenever the user discusses or requests v20/v21 results, mIoU numbers, or comparisons across runs:
1. Confirm the run used the 448x448 / 9-patch setting (check the bash script header comment block).
2. Any v20 mIoU number reported in tables/papers/discussion MUST be accompanied by this patch setting.
3. If a result was produced with a different patch setting (e.g. 224 single-patch from older runs), it is NOT comparable to current v20 results — flag this explicitly.

**Compute consequence:** 1449 images × 9 patches × 15 corruptions × 150 rounds ≈ 29.3M patch forward passes per full-15 method.

**Cache location:** `data/VOC/.cache/corruptions/<md5>_<corruption>_s5.npy`. Auto-derived by `prepare_data()` from `DATA_DIR=data/VOC/VOC2012/`. Validated by Task 7 sanity check on 2026-05-08.

**Source of truth:** docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md §2 and §12. Bash scripts under `bash/v20/` carry an in-file comment block reproducing this convention.
```

- [ ] **Step 2: Add an index entry to MEMORY.md**

Read `/home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/MEMORY.md` first to see existing entries, then append:

```
- [voc_patch_convention.md](voc_patch_convention.md) — VOC v20/v21 must use INIT_RESIZE=448x448 / patch=224 / stride=112 (9 patches/image) for results to be comparable
```

- [ ] **Step 3: Verify memory was saved correctly**

```bash
cat /home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/voc_patch_convention.md
grep voc_patch_convention /home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/MEMORY.md
```

Expected: file contents print, grep finds the index line.

- [ ] **Step 4: Commit nothing**

Memory files live outside the git repo (under `~/.claude/`), so no commit step.

---

## Final Sanity Check (after all tasks complete)

- [ ] All 6 scripts exist and are executable:

```bash
ls -l bash/v20/*.sh
```

Expected: 6 files, all with `+x` permission.

- [ ] All scripts pass `bash -n` syntax check:

```bash
for f in bash/v20/*.sh; do bash -n "$f" && echo "OK: $f"; done
```

Expected: 6 lines of `OK: bash/v20/<name>.sh`.

- [ ] Git log shows clean per-script commit history:

```bash
git log --oneline bash/v20/
```

Expected: 6 commits, one per script (Tasks 1–6), most recent first.

- [ ] Memory entry exists:

```bash
ls /home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/voc_patch_convention.md
```

Expected: file exists.
