# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MLMP — Project Guide for Claude

## Research Goal

**MLMP** (Multi-Level Multi-Prompt, NeurIPS 2025, arXiv:2505.21844) is a TTA framework for Open-Vocabulary Semantic Segmentation (OVSS) using NA-CLIP (ViT-L/14).

**Current focus**: Evaluate **Continual TTA (CTTA)** on the ACDC dataset (fog/night/rain/snow), comparing 5 methods across 10 continual rounds. Episodic TTA (reset per sample) is an unrealistic upper bound; CTTA never resets the model.

---

## Key Experimental Results (ACDC, 10 rounds)

| Method | Round 1 Mean | Round 10 Mean | Verdict |
|--------|-------------|--------------|---------|
| No Adaptation | 23.3 | 23.3 | Stable baseline |
| TENT-continual | ~25 | ~3.7 | Catastrophic forgetting |
| MLMP-continual | 33.2 | 1.7 | Severe forgetting (95% drop) |
| CoTTA | 23.4 | 23.4 | Stable (anti-forgetting works) |
| MLMP episodic | 30.6 | 30.6 | Best quality, but needs reset |
| DPCore | TBD | TBD | Visual prompt coreset (in progress) |

**Key insight**: Naive entropy minimization (TENT/MLMP without anti-forgetting) completely collapses in CTTA on OVSS. Anti-forgetting mechanisms (CoTTA) prevent collapse but don't improve over No Adaptation. **Goal: combine MLMP's quality (30.6) with CoTTA's stability**.

**DPCore hypothesis**: Instead of updating LayerNorm, learn visual prompt tokens. Maintain a coreset of (prompt, feature-stats) pairs — reuse nearest-match prompt for in-distribution (ID) batches, learn a new prompt for out-of-distribution (OOD) batches. Avoids catastrophic forgetting by never overwriting stored prompts.

---

## MLMP Core Ideas

Only **LayerNorm (γ, β) of the visual encoder** are updated. Text encoder is frozen.

1. **UAML**: Extract features from 18 ViT-L/14 layers. Weight by `α^l ∝ exp(-β·h^l)` (entropy). β=0 during adapt (uniform), β=1 during evaluate (sharpened).
2. **Multi-Prompt Loss**: 7 prompts from `prompts.yaml`. Average losses (not embeddings): `L = (1/T)Σ_t (L_pixel(T_t) + L_ILE(T_t))`.
3. **ILE**: CLS token entropy term weighted by `alpha_cls=1.0`.

**Episodic**: `reset() → adapt() → evaluate()` per sample. **Continual**: `evaluate() → adapt()`, no reset, state persists across all samples.

---

## CoTTA Three Mechanisms (reference implementation: `adapt/cotta.py`)

1. **EMA Teacher** (`mt=0.999`): `θ_teacher = mt·θ_teacher + (1-mt)·θ_student`
2. **Aug-Averaged Pseudo-labels** (`ap=0.92`, `aug_n=32`): If anchor confidence < ap, average N augmented teacher predictions.
3. **Stochastic Restoration** (`rst=0.01`): After each step, randomly restore 1% of weights to source.

---

## Codebase Architecture

```
MLMP/
├── main.py                  # Episodic entry point (--dataset choices include ACDCDataset)
├── main_continual.py        # CTTA entry point — 10-round protocol, evaluate-before-adapt
├── adapt/
│   ├── __init__.py          # get_method() factory (uses inspect.Parameter.empty for optional args)
│   ├── mlmp.py              # Episodic MLMP
│   ├── mlmp_continual.py    # Naive continual MLMP (no reset, has continual_adapt() alias)
│   ├── tent_continual.py    # Naive continual TENT (has continual_adapt() alias)
│   ├── cotta.py             # CoTTA with NA-CLIP backbone (open-vocab)
│   ├── dpcore.py            # DPCore: visual prompt coreset (see below)
│   ├── prompt_vit.py        # PromptVisualEncoder wrapper — injects learnable prompt tokens
│   ├── tent.py / tpt.py / clipartt.py / watt.py  # Other episodic baselines
├── ovss/clip/model.py       # Modified CLIP: multi-layer output, vision_out_type, CLS return
├── utils/
│   ├── segmentation_datasets.py  # All datasets incl. ACDCDataset
│   ├── metrics.py / misc.py / mm_transforms.py
├── bash/ACDC_10_round/      # {no_adapt, tent_continual, mlmp_continual, cotta, mlmp, dpcore}.sh
├── parse_acdc_results.py    # Reads save/ACDCDataset/ → LaTeX table (acdc_table.tex)
└── prompts.yaml             # 7 text prompt templates
```

---

## Two Entry Points

### `main.py` (episodic)
- `reset() → adapt() → evaluate()` per sample
- Re-instantiates model per corruption type
- Used by: `mlmp.sh` (episodic upper bound)

### `main_continual.py` (CTTA, 10 rounds)
- Instantiates model **once**, runs `evaluate() → continual_adapt()` per sample
- Loops through fog→night→rain→snow × 10 rounds
- Saves `results_all_rounds.txt` (Round, fog, night, rain, snow, Mean_mIoU)
- Used by: `no_adapt.sh`, `tent_continual.sh`, `mlmp_continual.sh`, `cotta.sh`

---

## ACDCDataset

Real adverse-condition images. **No synthetic CorruptTransform** — condition is in path.
- Images: `data/ACDC/rgb_anon/{fog|night|rain|snow}/val/`
- Labels: `data/ACDC/gt/{fog|night|rain|snow}/val/`
- 19 Cityscapes classes, same palette
- `img_suffix='_rgb_anon.png'`, `seg_map_suffix='_gt_labelTrainIds.png'`
- Defined in `utils/segmentation_datasets.py`, registered as `ACDCDataset`

---

## Critical Implementation Details

### `adapt/__init__.py` — `get_method()`
Uses `inspect.Parameter.empty` to skip optional constructor args (not in argparse). Required args with no default AND not in argparse raise `ValueError`. This is necessary because `CoTTA.__init__` has params like `mt`, `rst`, `ap` with defaults.

### `adapt/mlmp_continual.py` / `adapt/tent_continual.py`
- No `reset()` call in `adapt()`
- Have `continual_adapt(x)` alias (called by `main_continual.py`)
- `mlmp_continual`: LR=1e-5 (lower than episodic 1e-3), STEPS=1

### `main_continual.py` — Evaluate-Before-Adapt Protocol
```python
with torch.no_grad():
    patch_preds = adapt_method.evaluate(inputs)   # pre-update prediction
if args.adapt:
    adapt_method.continual_adapt(inputs)          # then update
```

### ACDC in `main_continual.py`
No `CorruptTransform` added when `dataset == "ACDCDataset"`. Method-specific args added via `add_method_specific_args`: CoTTA uses `--mt/--rst/--ap/--aug_n`; DPCore uses `--temp_tau/--ema_alpha/--thr_rho/--prompt_num/--verbose_dpcore`.

### DPCore (`adapt/dpcore.py`) — Key Differences from Other Methods
- **What's trained**: Visual prompt tokens (via `PromptVisualEncoder` wrapper in `adapt/prompt_vit.py`), NOT LayerNorm. Only `model.visual.prompts` has `requires_grad=True`.
- **Source statistics required**: Must call `adapt_method.obtain_src_stat(src_loader)` before adaptation. For `ACDCDataset` (no clean 'original' split), `main_continual.py` uses the first condition (fog) as source proxy.
- **ID/OOD decision**: Each batch is classified as In-Distribution (ID) or Out-of-Distribution (OOD) by comparing feature-stats loss with vs. without coreset prompts. Threshold: `loss_new < loss_raw * thr_rho`.
  - **ID path** (`E_ID=1` step): load nearest coreset prompt, fine-tune, then update coreset via EMA.
  - **OOD path** (`E_OOD=steps` steps): reset to source, learn from scratch, append new (prompt, stats) to coreset.
- **No forgetting**: stored coreset entries are never overwritten, only EMA-updated when ID.
- **`--steps` sets `E_OOD`** (OOD update steps), not the per-batch step count for ID samples.
- **`--batch_size` matters**: DPCore uses `batch_size=16` (not 1) — larger batches give stable feature statistics for the ID/OOD decision and source-discrepancy loss.

---

## ACDC Bash Scripts (`bash/ACDC_10_round/`)

| Script | Method | Entry Point | Key Params |
|--------|--------|-------------|------------|
| `no_adapt.sh` | tent_continual (no --adapt) | main_continual.py | batch=1 |
| `tent_continual.sh` | tent_continual | main_continual.py | LR=1e-5, steps=1 |
| `mlmp_continual.sh` | mlmp_continual | main_continual.py | LR=5e-6, steps=1 |
| `cotta.sh` | cotta | main_continual.py | mt=0.999, rst=0.01, ap=0.92, aug_n=32, LR=1e-5 |
| `mlmp.sh` | mlmp (episodic) | main.py | LR=1e-3, steps=10, trials=1 |
| `dpcore.sh` | dpcore | main_continual.py | LR=1e-5, steps=50, batch=16, thr_rho=0.95 |

Results saved to `save/ACDCDataset/{method_name}/` (or custom `SAVE_DIR` in the script).

---

## Result Parsing

`parse_acdc_results.py` auto-generates a LaTeX table:
- Reads `results_all_rounds.txt` for continual methods
- Reads per-condition `results.txt` for episodic MLMP
- Shows Rounds 1/4/7/10 × {fog, night, rain, snow}
- Extracts `Total Duration (s)` → time per round (seconds)
- Output: `save/ACDCDataset/acdc_table.tex`

Run: `python parse_acdc_results.py`
