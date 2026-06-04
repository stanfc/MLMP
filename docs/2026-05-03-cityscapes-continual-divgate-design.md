# Design: TENT-DivGate Continual on CityscapesDataset

**Date**: 2026-05-03  
**Status**: approved, pending implementation  
**Goal**: Evaluate TENT-DivGate under the same multi-round continual protocol used for ACDC, but on CityscapesDataset with synthetic ImageNet-C corruptions. This establishes whether the method generalises beyond real adverse-weather conditions.

---

## Context

TENT-DivGate on ACDC achieved mean=31.59, peak=32.96@R27, R150=31.34 — beating MLMP-episodic (+1.0 mIoU), stable to 150 rounds. Best confirmed config: `h_thr=1.6, h_warn=1.4, cau_rst=0.01, brake_rst=0.05`.

ACDC uses 4 real adverse conditions (fog/night/rain/snow), each with ~2000 val images → ~8012 images/round.

Cityscapes val has 500 clean images. Corruptions are applied on-the-fly via `CorruptTransform` (not stored to disk). Using all 15 ImageNet-C corruptions gives 15 × 500 = 7500 images/round — comparable compute to ACDC.

---

## Deliverable

**One new bash script**: `bash/cityscapes_continual/tent_divgate_continual.sh`

No changes to `main_continual.py` or any Python code are required. The main loop is already condition-agnostic; `CityscapesDataset` is already in `--dataset` choices; `prepare_data()` already inserts `CorruptTransform` for non-ACDC datasets.

---

## Script Design

### Key variables (all at top, adjustable)

| Variable | Default value | Notes |
|---|---|---|
| `GPU_ID` | 0 | |
| `DATASET` | `CityscapesDataset` | |
| `DATA_DIR` | `data/Cityscape/` | |
| `INIT_RESIZE` | `"1120 560"` | matches existing Cityscapes scripts |
| `CORRUPTIONS_LIST` | all 15 in ImageNet-C order | one-per-line with comments for easy subsetting |
| `METHOD` | `tent_divgate_continual` | |
| `LR` | `0.00001` | same as ACDC |
| `STEPS` | `1` | same as ACDC |
| `BATCH_SIZE` | `1` | |
| `CONTINUAL_ROUNDS` | `150` | |
| `H_THRESHOLD` | `1.6` | best from ACDC h_threshold sweep |
| `H_WARNING` | `1.4` | |
| `CAUTIOUS_RST` | `0.01` | best from ACDC cautious_rst sweep |
| `BRAKE_RST` | `0.05` | |
| `MONITOR_INTERVAL` | `50` | |
| `SAVE_DIR` | `save/CityscapesDataset/tent_divgate_continual/` | |

### Corruption list (ImageNet-C standard order)

```
gaussian_noise  shot_noise  impulse_noise
defocus_blur    glass_blur  motion_blur   zoom_blur
snow  frost  fog  brightness  contrast
elastic_transform  pixelate  jpeg_compression
```

Listed as a multi-line variable with a comment before each group (noise / blur / weather / digital) so individual corruptions can be commented out to run a subset.

### Override pattern (matching ACDC convention)

To run a weather-only subset or a different hyperparameter without editing the script:

```bash
CORRUPTIONS_LIST="fog snow frost brightness" \
H_THRESHOLD=1.7 \
SAVE_DIR="save/CityscapesDataset/tent_divgate_continual_weather/" \
  bash bash/cityscapes_continual/tent_divgate_continual.sh
```

---

## Output

Same format as ACDC:
- `save/CityscapesDataset/tent_divgate_continual/results_all_rounds.txt` — one row per round, columns = each corruption + Mean_mIoU
- `save/CityscapesDataset/tent_divgate_continual/divgate_log.txt` — H_margin trajectory (one row per 50-batch window)
- Per-round per-condition `results.txt` under `round_NN/{corruption}/`

---

## What is NOT in scope

- No new Python files
- No result parser for Cityscapes (user reads `results_all_rounds.txt` directly; a parser can be added later)
- No baseline scripts (no_adapt, tent_continual) in this iteration — those can be added separately
