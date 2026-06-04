# Cityscapes Continual DivGate Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write `bash/cityscapes_continual/tent_divgate_continual.sh` — the Cityscapes analogue of the ACDC DivGate script, running TENT-DivGate over 15 ImageNet-C corruptions × 150 continual rounds.

**Architecture:** Single bash script. No Python changes. `main_continual.py` already accepts `CityscapesDataset` and applies `CorruptTransform` on-the-fly for any non-ACDC dataset. The script uses a bash array for the corruption list so individual corruptions can be commented out, with an env-var override for one-liner subset runs.

**Tech Stack:** bash, `main_continual.py` (existing), CityscapesDataset (existing), tent_divgate_continual method (existing).

---

## File Map

| Action | Path |
|--------|------|
| Create (new dir) | `bash/cityscapes_continual/` |
| Create | `bash/cityscapes_continual/tent_divgate_continual.sh` |

No Python files are created or modified.

---

### Task 1: Create the script

**Files:**
- Create: `bash/cityscapes_continual/tent_divgate_continual.sh`

- [ ] **Step 1: Create the directory and write the script**

```bash
mkdir -p bash/cityscapes_continual
```

Create `bash/cityscapes_continual/tent_divgate_continual.sh` with this exact content:

```bash
#!/bin/bash
# TENT-DivGate-Continual on CityscapesDataset (CTTA, N rounds).
# 15 ImageNet-C corruptions applied on-the-fly; same gate mechanism
# and best confirmed hyperparameters as the ACDC experiment.
# See docs/2026-05-03-cityscapes-continual-divgate-design.md.

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=0

# ── Dataset ────────────────────────────────────────────────────────
DATASET=CityscapesDataset
DATA_DIR="data/Cityscape/"
INIT_RESIZE="1120 560"
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
# One-liner subset override: CORRUPTIONS_LIST="fog snow frost brightness" bash script.sh
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Diversity gate (best confirmed from ACDC sweeps) ───────────────
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

- [ ] **Step 2: Make it executable**

```bash
chmod +x bash/cityscapes_continual/tent_divgate_continual.sh
```

- [ ] **Step 3: Smoke test — verify argument parsing (5 batches per condition)**

The `--debug` flag limits each condition to 5 batches and exits cleanly. This confirms the script reaches Python successfully and that all arguments are accepted, without running a full experiment.

```bash
CONTINUAL_ROUNDS=1 \
SAVE_DIR="save/CityscapesDataset/tent_divgate_continual_debug/" \
CUDA_VISIBLE_DEVICES=0 python main_continual.py \
    --adapt \
    --method tent_divgate_continual \
    --ovss_type naclip \
    --ovss_backbone ViT-L/14 \
    --dataset CityscapesDataset \
    --data_dir data/Cityscape/ \
    --init_resize 1120 560 \
    --patch_size 224 224 \
    --patch_stride 112 \
    --corruptions_list gaussian_noise fog \
    --workers 4 \
    --lr 0.00001 \
    --steps 1 \
    --batch_size 1 \
    --continual_rounds 1 \
    --seed 0 \
    --h_threshold 1.6 \
    --h_warning 1.4 \
    --monitor_interval 50 \
    --cautious_rst 0.01 \
    --brake_rst 0.05 \
    --save_dir save/CityscapesDataset/tent_divgate_continual_debug/ \
    --class_extensions \
    --debug
```

Expected: prints startup banner, runs 2 conditions × 5 batches each, prints per-condition mIoU, writes `results_all_rounds.txt` and `divgate_log.txt` to the debug save dir, exits with no error.

- [ ] **Step 4: Verify output files were created**

```bash
ls save/CityscapesDataset/tent_divgate_continual_debug/
# Expected: args.json  divgate_log.txt  results_all_rounds.txt  round_01/

head -3 save/CityscapesDataset/tent_divgate_continual_debug/results_all_rounds.txt
# Expected: header line "Round, gaussian_noise, fog, Mean_mIoU"
#           followed by round 1 values
```

- [ ] **Step 5: Commit**

```bash
git add bash/cityscapes_continual/tent_divgate_continual.sh
git commit -m "Add TENT-DivGate continual script for CityscapesDataset (15 ImageNet-C corruptions)"
```

---

## Self-Review

**Spec coverage:**
- ✅ `bash/cityscapes_continual/tent_divgate_continual.sh` created
- ✅ All 15 ImageNet-C corruptions in standard order, grouped with comments
- ✅ Comment-out pattern works (bash array, one entry per line)
- ✅ Env-var override pattern: `CORRUPTIONS_LIST="fog snow" bash script.sh` and `SAVE_DIR=... bash script.sh`
- ✅ Best ACDC DivGate params as defaults (h_thr=1.6, h_warn=1.4, cau_rst=0.01, brake_rst=0.05)
- ✅ CONTINUAL_ROUNDS=150, SAVE_DIR=save/CityscapesDataset/tent_divgate_continual/
- ✅ No Python changes
- ✅ Output format matches ACDC (results_all_rounds.txt, divgate_log.txt)

**Placeholder scan:** No TBDs, TODOs, or vague steps. All code is complete.

**Type consistency:** Single task; no cross-task type dependencies.
