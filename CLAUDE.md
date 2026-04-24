# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MLMP — Project Guide for Claude

> **🟢 Current status (2026-04-24)**: `cma_proto_continual` implemented and sanity-tested; not yet run. Next action: `bash bash/ACDC_10_round/cma_proto_continual.sh`.
> **For a fast orientation on the active experiment, read [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md) first** — it consolidates the research arc (what was tried, what we learned, what's next) in one place.

## Research Goal

**MLMP** (Multi-Level Multi-Prompt, NeurIPS 2025, arXiv:2505.21844) is a TTA framework for Open-Vocabulary Semantic Segmentation (OVSS) using NA-CLIP (ViT-L/14).

**Phase 1 (complete)**: Evaluate existing CTTA methods on ACDC (fog/night/rain/snow), 10 continual rounds, evaluate-before-adapt protocol.

**Phase 2 (current)**: Design a new CTTA method that is both stable (CoTTA-level) and high-quality (MLMP episodic-level). See `proposal.md` for full analysis and proposed directions.

---

## Key Experimental Results (ACDC, 10 rounds)

Results differ significantly by adaptation steps per sample. Two variants were run:

### Step=1 (stable dynamics, `acdc_table_step1.tex`)

| Method | R1 Mean | R10 Mean | Mean (all rounds) | Verdict |
|--------|---------|----------|-------------------|---------|
| No Adaptation | 23.3 | 23.3 | 23.3 | Stable baseline |
| TENT-continual | 23.9 | 30.9 | 28.0 | Steady improvement within 10 rounds (collapses ~R80) |
| MLMP-continual | 29.8 | 25.7 | 28.9 | Gradual decline |
| CoTTA | 23.4 | 23.4 | 23.4 | Stable, no improvement |
| MLMP episodic | 30.6 | 30.6 | **29.8** | Best overall, requires reset |

### Step=10 (fast adaptation, collapses earlier, `acdc_table_step10.tex`)

| Method | R1 Mean | R4 Mean | R10 Mean | Mean (all rounds) | Verdict |
|--------|---------|---------|----------|-------------------|---------|
| No Adaptation | 23.3 | 23.3 | 23.3 | 23.3 | Stable |
| TENT-continual | 28.3 | **35.3** (fog!) | 5.1 | 23.2 | Peaks R4, collapses by R10 |
| MLMP-continual | 28.7 | 3.2 | 1.5 | 6.0 | Collapses by R2 |
| CoTTA | 23.4 | 23.4 | 23.4 | 23.4 | Stable, no improvement |
| MLMP episodic | 33.4 | 33.4 | 33.4 | **30.6** | Best overall, requires reset |

**Core tension**: MLMP episodic (30.6 mIoU) requires per-sample reset — unrealistic in deployment. Entropy-based continual methods improve early but collapse catastrophically. CoTTA is stable but never improves past No Adaptation.

---

## Why Entropy-Based Methods Collapse (Root Cause)

Entropy minimization rewards **confidence**, not **correctness**. In the CTTA setting, LayerNorm (γ, β) parameters accumulate drift across samples. Collapse follows four phases:

```
Phase 1 — Genuine adaptation (R1–R20):
  Entropy ↓ → real performance gain (CLIP text anchors delay catastrophic drift)

Phase 2 — Gradual drift (R20–R80):
  LayerNorm parameters shift; visual features slowly leave text-compatible space

Phase 3 — Catastrophic collapse (R80–R110):
  1–2 dominant classes (sky, road) always win → error accumulation → runaway feedback

Phase 4 — Degenerate steady state (~7.9 mIoU):
  All pixels predicted as 1–2 classes; high confidence; gradient ≈ 0; locked
```

**Why OVSS delays (but doesn't prevent) collapse**: The frozen text encoder provides fixed semantic anchors that constrain drift. This is why TENT on OVSS initially improves (unlike on closed-set models), but LayerNorm updates can still push visual features out of the text-compatible regime over many rounds.

---

## Design Constraints for New Method

| Constraint | Value |
|-----------|-------|
| Backbone | NA-CLIP (ViT-L/14), LayerNorm only (no BatchNorm) |
| Text encoder | Frozen throughout |
| Adaptation scope | LayerNorm (γ, β) or visual prompt tokens |
| Source data | Small proxy available (fog condition used as source) |
| Labels | None (unsupervised TTA) |
| Reset policy | No reset — continual setting |
| **Excluded: entropy minimization** | Collapses to trivial solution |
| **Excluded: global batch statistics** | Cannot distinguish class-specific drift; inapplicable to ViT-L/14 |

---

## Proposed New Method (see `proposal.md` for full directions, `docs/cma_continual_spec.md` and `docs/cma_proto_continual_spec.md` for active implementations)

**Primary: Direction 1 + Direction 2 combined** (`cma_proto_continual`). D1 implemented first as `cma_continual`; it revealed that confirmation bias causes collapse even without entropy minimization, so D2 was implemented on top of D1.

### Direction 1 — Cross-modal Alignment TTA — **implemented as `cma_continual` (150-round result: collapsed at ~R35)**
Use frozen text embeddings as supervision. For top-K% confidence pixels, maximize cosine similarity between visual features and their predicted class text embedding:
```
L_CMA = -mean_{i ∈ S_conf} cos(v_i, t_{ĉ_i})
```
**Experimental finding**: peaked at R16 (26.82 mIoU), then collapsed to 1.20 by R40. Text embedding diversity *delays* but does not *prevent* collapse. Root cause: any loss whose target depends on current model predictions has a confirmation bias feedback loop that eventually locks the model into predicting 1–2 dominant classes. See `docs/cma_continual_spec.md` §9 (post-mortem).

### Direction 2 — Semantic Prototype Memory Bank (anti-forgetting) — **implemented as `cma_proto_continual`**
Per-class visual feature centroids. Two prototype sets: **source** (frozen forever, the confirmation-bias-immune anchor) and **target** (EMA-updated, slow adaptation). Combined loss with three cosine-alignment terms sharing the Top-K% mask:
```
L = λ_CMA · L_CMA(t_c)    + λ_src · L_src(p_src_c)   + λ_tgt · L_tgt(p_tgt_c)
     text anchor            fixed source anchor        EMA target anchor
```
Source prototypes initialized via pseudo-labels on fog (source proxy) with cross-prompt agreement filter; classes with zero confident pixels fall back to text embedding. `λ_tgt = 0` degrades to pure source-anchor variant. See `docs/cma_proto_continual_spec.md` for full design.

### Direction 3 — Pseudo-label Self-Training (ablation baseline) — *deferred*
Replace entropy with cross-entropy against EMA teacher predictions. Directional loss with an explicit target — more stable than entropy but lacks the OVSS-specific text geometry signal.

**DPCore (in progress)**: Instead of updating LayerNorm, learn visual prompt tokens. Maintain a coreset of (prompt, feature-stats) pairs — reuse nearest-match prompt for ID batches, learn a new prompt for OOD batches.

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
│   ├── cma_continual.py     # CMA-continual: cross-modal alignment loss (Direction 1)
│   ├── cma_proto_continual.py  # CMA + prototype bank (Direction 1+2, primary method)
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
No `CorruptTransform` added when `dataset == "ACDCDataset"`. Method-specific args added via `add_method_specific_args`: CoTTA uses `--mt/--rst/--ap/--aug_n`; DPCore uses `--temp_tau/--ema_alpha/--thr_rho/--prompt_num/--verbose_dpcore`; CMA-continual uses `--top_k_percent`; CMA-Proto-continual uses `--top_k_percent/--lambda_cma/--lambda_src/--lambda_tgt/--ema_alpha/--src_conf_threshold/--src_max_samples` plus shared `--src_corruption` (defaults to `conditions[0]` if unset).

### `adapt/cma_continual.py` — CMA Implementation Details
- **Loss**: `-mean(cos(v_i, t_{c_i}))` over top-K% confidence pixels per batch. See `docs/cma_continual_spec.md` §2 for tensor mechanics.
- **What's trained**: LayerNorm (γ, β) of visual encoder, identical to TENT/MLMP. Text encoder fully frozen.
- **Pseudo-labels are no_grad**: `pred_cls` and `confidence` are computed under `torch.no_grad()` so the mask/target lookup doesn't backpropagate into the prediction itself; gradients flow only through `vis_feat`.
- **Multi-prompt averaging**: Logits and text features are averaged over the 7 prompt templates before computing pseudo-labels and the alignment target. The averaged text vector is re-normalized to unit length (mean of unit vectors is not unit-norm).
- **CLS token handling**: `image_features` returned by `model.forward()` has shape `(B, w*h+1, D)` with CLS at index 0. CMA loss uses `image_features[:, 1:, :]` to align with `logits` (which already drops CLS).

### `adapt/cma_proto_continual.py` — CMA-Proto Implementation Details
- **Three-term loss**: `L = λ_CMA · L_CMA + λ_src · L_src + λ_tgt · L_tgt`, all cosine-alignment, shared Top-K mask. See `docs/cma_proto_continual_spec.md` §3.
- **Prototype init**: `obtain_src_prototypes(data_loader)` is called **once** before the stream begins (analogous to DPCore's `obtain_src_stat()`). Filters pixels by `confidence ≥ src_conf_threshold` AND cross-prompt agreement (all 7 templates predict same class); classes with zero pixels fall back to text embedding.
- **Source prototype is frozen forever**: `self.p_src` never mutates after init — this is the anti-confirmation-bias anchor. Target prototype `self.p_tgt` is EMA-updated under `torch.no_grad()` inside the loss function.
- **Source proxy for ACDC**: first condition (`fog`) by default; override via `--src_corruption`. Pseudo-label-only (no ground truth), matching realistic CTTA assumption.
- **Pre-stream dispatch in `main_continual.py`**: mirrors DPCore's block — if `args.method == 'cma_proto_continual'`, build a source `prepare_data()` loader with `corruption=args.src_corruption or conditions[0]` and call `adapt_method.obtain_src_prototypes(src_loader)`.
- **Degradation to simpler variants**: `λ_tgt=0` → source-anchor-only (hard Option A); `λ_src=λ_tgt=0` → equivalent to plain `cma_continual`.

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
| `cma_continual.sh` | cma_continual | main_continual.py | LR=1e-5, steps=1, batch=1, top_k_percent=0.2 |
| `cma_proto_continual.sh` | cma_proto_continual | main_continual.py | LR=1e-5, steps=1, batch=1, λ_cma=1.0, λ_src=1.0, λ_tgt=0.5, ema=0.999, src=fog |

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
