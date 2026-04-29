# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MLMP — Project Guide for Claude

> **🟢 Current status (2026-04-27)**: h_threshold sweep running (R63/150 as of 2026-04-27). Variants at h_thr=1.5/1.6/1.7/1.8 (all cau_rst=0.01, h_warn=1.4). Early leader @R63: **h_thr=1.7 → mean≈32.15** (vs 1.8→31.31, 1.5→31.92). Prior confirmed best: **h_thr=1.6, h_warn=1.4, cau_rst=0.01 → mean=31.59, peak=32.96@R27, R150=31.34** — beats MLMP-episodic (30.6) by +1.0 mIoU, stable to R150. Save dirs: `save/ACDCDataset/tent_divgate_continual_cau_threshold_{value}/`.
> **For the full research arc (every method tried, what we learned, current state), read [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md) first.** That file is the canonical entry point — this guide covers conventions and impl details, not narrative.

## Research Goal

**MLMP** (Multi-Level Multi-Prompt, NeurIPS 2025, arXiv:2505.21844) is a TTA framework for Open-Vocabulary Semantic Segmentation (OVSS) using NA-CLIP (ViT-L/14).

**Phase 1 (complete)**: Evaluate existing CTTA methods on ACDC (fog/night/rain/snow), continual rounds, evaluate-before-adapt protocol. Established the core tension: methods that improve always collapse; methods that are stable never improve.

**Phase 2 (in progress)**: Design a new CTTA method that is both stable (CoTTA-level) and high-quality (MLMP episodic-level). Six methods tried so far, summarized in [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md). The original analysis is in `proposal.md`; the post-CMA reframing (after the first three methods failed) is in `proposal_after_cma.md`.

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

## Methods Implemented in Phase 2 (chronological)

The full narrative — what each method tried, why it failed or partially worked, what we learned — is in [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md). Brief index here:

| Method | File | Spec | Status / R150 result |
|---|---|---|---|
| `cma_continual` | `adapt/cma_continual.py` | [cma_continual_spec.md](docs/cma_continual_spec.md) | **Collapsed**. Peak 26.82 R16 → dead 1.20 by R40. Post-mortem in §9 reframed the problem (confirmation bias is in the *class index*, not the target). |
| `cma_proto_continual` | `adapt/cma_proto_continual.py` | [cma_proto_continual_spec.md](docs/cma_proto_continual_spec.md) | **Collapsed**. Frozen source prototype delayed collapse to R54 but didn't prevent it — confirmation bias remains because pseudo-label `ĉ_i` still comes from the current model. Motivated `proposal_after_cma.md`. |
| `cma_layered_continual` | `adapt/cma_layered_continual.py` | [cma_layered_continual_spec.md](docs/cma_layered_continual_spec.md) | **Stable but flat**. Layer-stratified restoration (Direction A) caps at ~24 mIoU regardless of cutoff/rate combination. Negative result: defense moved to optimizer side cannot raise CMA's natural ceiling. |
| `cma_divgate_continual` | `adapt/cma_divgate_continual.py` | [cma_divgate_continual_spec.md](docs/cma_divgate_continual_spec.md) | **Partial success**. Buffer-N H_margin gate (Direction B) prevented dead state (mean 21.21 vs CMA's 5.54), but base loss ceiling (CMA peak 27.4) caps overall mean below No Adapt 23.34. |
| **`tent_divgate_continual`** | `adapt/tent_divgate_continual.py` | [tent_divgate_continual_spec.md](docs/tent_divgate_continual_spec.md) | **Best method. All 150R complete.** Baseline (h_thr=1.8): mean=30.14, R150=29.02. After cautious_rst sweep + threshold tune (h_thr=1.6, h_warn=1.4, cau_rst=0.01): **mean=31.59, peak=32.96@R27, R150=31.34**. Beats MLMP-episodic by +1.0 mIoU, stable to R150. **Next**: h_threshold sweep (1.7/1.8/2.0). |

The original three-direction reframing is in [proposal_after_cma.md](proposal_after_cma.md) (written after CMA-Proto failed). Direction A = layered restoration; Direction B = diversity gate; Direction C (two-timescale meta-adapt) is still deferred.

**Headline finding**: TENT-DivGate is the first method that **does** match and beat MLMP-episodic under continual conditions (mean 31.59 vs 30.6, stable to R150). The same gate that *partially* worked on CMA *fully* works on TENT because TENT's drift is slow enough for cautious-mode restoration to compensate (CMA collapsed in ~18 rounds; TENT drifts over ~60). Current work: h_threshold sweep to find the gate activation rate that maximises the mean.

### DPCore (parallel work, prompt-tuning approach)
Instead of LayerNorm, learn visual prompt tokens. Maintain a coreset of (prompt, feature-stats) pairs — reuse nearest-match prompt for ID batches, learn a new prompt for OOD batches. See `adapt/dpcore.py`. Less central to the current research arc.

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
│   ├── __init__.py                # get_method() factory (uses inspect.Parameter.empty for optional args)
│   ├── mlmp.py                    # Episodic MLMP
│   ├── mlmp_continual.py          # Naive continual MLMP (no reset, has continual_adapt() alias)
│   ├── tent_continual.py          # Naive continual TENT (has continual_adapt() alias)
│   ├── cma_continual.py           # Phase 2A — CMA loss, collapsed
│   ├── cma_proto_continual.py     # Phase 2B — CMA + frozen source prototype, collapsed
│   ├── cma_layered_continual.py   # Phase 2D — Direction A: layer-stratified restore (flat)
│   ├── cma_divgate_continual.py   # Phase 2E — Direction B on CMA: H_margin gate (partial)
│   ├── tent_divgate_continual.py  # Phase 2F — Direction B on TENT: H_margin gate (CURRENT BEST)
│   ├── cotta.py                   # CoTTA with NA-CLIP backbone (open-vocab)
│   ├── dpcore.py                  # DPCore: visual prompt coreset (see below)
│   ├── prompt_vit.py              # PromptVisualEncoder wrapper — injects learnable prompt tokens
│   ├── tent.py / tpt.py / clipartt.py / watt.py  # Other episodic baselines
├── docs/
│   ├── EXPERIMENT_STATUS.md       # FULL RESEARCH ARC — read first
│   ├── cma_*_spec.md              # one spec per method (design + post-mortem)
│   ├── tent_divgate_continual_spec.md
│   └── cma_*_plan.md              # implementation plans (writing-plans output)
├── proposal.md                    # Phase 1 analysis + original Direction 1+2+3 proposal
├── proposal_after_cma.md          # Post-CMA reframing → Direction A/B/C
├── ovss/clip/model.py             # Modified CLIP: multi-layer output, vision_out_type, CLS return
├── utils/
│   ├── segmentation_datasets.py   # All datasets incl. ACDCDataset
│   ├── metrics.py / misc.py / mm_transforms.py
├── bash/ACDC_10_round/            # one runner script per method (see table below)
├── parse_acdc_results.py          # Reads save/ACDCDataset/ → LaTeX table (acdc_table.tex)
└── prompts.yaml                   # 7 text prompt templates
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
No `CorruptTransform` added when `dataset == "ACDCDataset"`. Method-specific args added via `add_method_specific_args`:
- `cotta` → `--mt/--rst/--ap/--aug_n`
- `dpcore` → `--temp_tau/--ema_alpha/--thr_rho/--prompt_num/--verbose_dpcore`
- `cma_continual` → `--top_k_percent`
- `cma_proto_continual` → `--top_k_percent/--lambda_cma/--lambda_src/--lambda_tgt/--ema_alpha/--src_conf_threshold/--src_max_samples` plus shared `--src_corruption` (defaults to `conditions[0]` if unset)
- `cma_layered_continual` → `--top_k_percent/--early_rst/--mid_rst/--late_rst/--early_cutoff/--late_cutoff`
- `cma_divgate_continual` → `--top_k_percent/--h_threshold/--h_warning/--monitor_interval/--cautious_rst/--brake_rst`
- `tent_divgate_continual` → `--h_threshold/--h_warning/--monitor_interval/--cautious_rst/--brake_rst` (no `--top_k_percent` — pure TENT loss)

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

### `adapt/cma_layered_continual.py` — Layered Restoration (Direction A)
- **Loss**: identical to `cma_continual` (same top-K% CMA tensor ops, copied verbatim).
- **New mechanism**: after each `optimizer.step()`, every visual-encoder LN parameter is stochastically restored toward the source snapshot with a probability that depends on which transformer block it belongs to.
- **Layer classification**: regex `resblocks\.(\d+)\.` on the state-dict name. `ln_pre` → early; `ln_post` → late. Three groups: `[0, early_cutoff)` → `early_rst`; `[early_cutoff, late_cutoff)` → `mid_rst`; `[late_cutoff, num_blocks)` → `late_rst`.
- **`named_ln_params`**: precomputed list of `(state_dict_name, param)` pairs from `collect_ln_params(self.model.visual)`. For ViT-L/14 there are exactly 100 entries (24 blocks × 2 LN × 2 params + ln_pre × 2 + ln_post × 2).
- **Restoration mask is per-element**: `(torch.rand(p.shape) < rst).to(p.dtype)`, then `p.data.mul_(1-mask).add_(src*mask)`. Setting any rate to 0 short-circuits the loop. Setting to 1 fully freezes that group.
- **Degradation**: `early_rst = mid_rst = late_rst = 0` → equivalent to `cma_continual`.

### `adapt/cma_divgate_continual.py` and `adapt/tent_divgate_continual.py` — Diversity-Gated (Direction B)
Both share the same gate machinery; only the base loss differs. Spec docs are next to each other for direct comparison.

- **Buffer-N marginal aggregation**: every adapt step, `probs.mean(dim=[0, 2, 3])` (shape `(C,)`) is `.detach().float().cpu()`-pushed to `self.marginal_buf`. After `monitor_interval` (default 50) batches, `_update_mode()` aggregates `mean(buffer)`, normalizes, and computes `H_margin = -Σ p log p`.
- **Three-tier mode → flat rst**: `aggressive` (rst=0), `cautious` (rst=cautious_rst), `brake` (rst=brake_rst). Boundaries `h_threshold=1.8` (aggressive cutoff) and `h_warning=1.2` (cautious cutoff). **Empirical observation**: brake mode rarely fires at default `h_warning=1.2` — the gate effectively behaves as a 2-tier (aggressive ↔ cautious) controller. This is fine when paired with TENT (cautious-mode 0.005 rst is enough to compensate slow drift); it was the bottleneck for CMA-DivGate where drift was too fast.
- **Initial mode = aggressive**: `current_rst = 0` for the first `monitor_interval` batches, so the gate doesn't restrict initial adaptation.
- **Mode transitions are logged to stdout**: `[DivGate] B{total_batches}: H_margin={value:.3f}  {old} -> {new}` (CMA variant) or `[DivGate-T]` (TENT variant). Same-mode windows are silent.
- **Restoration is flat across all visual-encoder LN params** (single rate per gate-state, applied to all 100 LN params on ViT-L/14). This is intentionally distinct from Direction A's stratified rate.
- **CMA variant** (`cma_divgate_continual`): uses CMA loss with `--top_k_percent` mask. Marginals come from `avg_logits` (prompt-averaged).
- **TENT variant** (`tent_divgate_continual`): uses pure pixel-wise softmax entropy across all pixels (no top-K mask). Marginals come from `logits[0].softmax(dim=1)` — single prompt template (`tent_continual` convention).
- **Determinism note (important for hyperparameter sweeps)**: with seed=0, runs differing only in `brake_rst` are bit-identical until brake mode actually fires. We saw this in the CMA-DivGate brake-rate sweep (`save/ACDCDataset/cma_divgate_continual_brake_*`) — the four runs were identical for the first 33 rounds because brake never triggered with the default `h_warning=1.2`. Cautious_rst was the only restoration in effect.

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
| `cma_layered_continual.sh` | cma_layered_continual | main_continual.py | LR=1e-5, steps=1, top_k=0.2, early/mid/late_rst=0.001/0.01/0.05, cutoffs=8/16 (defaults; rates in the script have been tuned during ablation) |
| `cma_divgate_continual.sh` | cma_divgate_continual | main_continual.py | LR=1e-5, steps=1, top_k=0.2, h_threshold=1.8, h_warning=1.2, monitor_interval=50, cautious_rst=0.005, brake_rst=0.05 |
| `tent_divgate_continual.sh` | tent_divgate_continual | main_continual.py | LR=1e-5, steps=1, no top-K — pure TENT loss. **Best confirmed**: h_thr=1.6, h_warn=1.4, cau_rst=0.01, brake_rst=0.05. **Script currently set to h_thr=1.8, h_warn=1.4, cau_rst=0.01** for the h_threshold sweep (SAVE_DIR uses `_cau_threshold_${H_THRESHOLD}/` suffix). |

Results saved to `save/ACDCDataset/{method_name}/` (or custom `SAVE_DIR` in the script). Multiple runs of the same method with different hyperparameters use suffixes like `cma_layered_continual_rate__0.001_0.05` or `cma_divgate_continual_brake_0.005`.

---

## Running & Monitoring Experiments

Launch any bash script directly; GPU is set inside the script (`GPU_ID=N`):

```bash
bash bash/ACDC_10_round/tent_divgate_continual.sh
```

To run a variant with different hyperparameters, edit `H_THRESHOLD`, `CAUTIOUS_RST`, and `SAVE_DIR` inline before launching, or override on the fly:

```bash
H_THRESHOLD=1.7 CAUTIOUS_RST=0.01 SAVE_DIR="save/ACDCDataset/tent_divgate_continual_hthr_1.7/" \
  bash bash/ACDC_10_round/tent_divgate_continual.sh
```

Check progress mid-run (results written after every round):

```bash
tail -5 save/ACDCDataset/<save_dir>/results_all_rounds.txt
# Round N, fog, night, rain, snow, Mean_mIoU
```

Track gate behaviour (TENT-DivGate only — written every monitor_interval batches):

```bash
tail -20 save/ACDCDataset/<save_dir>/divgate_log.txt
# total_batches,h_margin,mode
```

Mode-transition lines are also printed to stdout: `[DivGate-T] B{N}: H_margin={v}  old -> new`.

---

## Result Parsing

`parse_acdc_results.py` auto-generates a LaTeX table:
- Reads `results_all_rounds.txt` for continual methods
- Reads per-condition `results.txt` for episodic MLMP
- Shows Rounds 1/4/7/10 × {fog, night, rain, snow}
- Extracts `Total Duration (s)` → time per round (seconds)
- Output: `save/ACDCDataset/acdc_table.tex`

Run: `python parse_acdc_results.py`
