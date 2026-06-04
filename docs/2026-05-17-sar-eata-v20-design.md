# Design: SAR-Continual and EATA-Continual on PascalVOC20

**Date**: 2026-05-17
**Status**: approved, pending implementation plan
**Goal**: Implement two entropy-based TTA methods (SAR, EATA) as continual variants and run them on PascalVOC20 to test whether they show upward performance trends where TENT-DivGate plateaus.

---

## Context

TENT-DivGate works on ACDC (mean=31.59, beats MLMP-episodic +1.0 mIoU) but on Cityscapes/VOC20 with synthetic ImageNet-C corruptions it cannot push performance upward — the adaptation headroom between source model and MLMP-episodic is too small for TENT's entropy minimization to find a positive gradient direction. The DivGate's role is to prevent collapse; it cannot create improvement that TENT does not generate.

We hypothesise that two recent TTA methods address this from different angles:
- **SAR** (ICLR 2023) finds flatter minima via SAM, which generalise better when headroom is small.
- **EATA** (ICML 2022) uses Fisher-weighted EWC regularisation — a *structured* anti-forgetting mechanism, in contrast to DivGate's *stochastic* restoration.

Both could plausibly produce upward trends where TENT-DivGate plateaus. This document specifies the implementation for testing both on VOC20.

---

## Scope

**In scope**:
- Two new methods: `sar_continual`, `eata_continual` (both TENT-loss base, single-prompt, continual mode)
- Bash scripts for VOC20 (all-15 and weather-5 reusing same script via `CORRUPTIONS_LIST` env-var)
- Integration with `main_continual.py` (method-specific args + EATA's pre-stream Fisher computation)

**Out of scope**:
- MLMP-base variants (would double the work; can be added later if TENT-base shows promise)
- ACDC or Cityscapes runs (only VOC20 in this iteration)
- SAR's augmented-view variance filtering (segmentation augmentation is complex; original paper mostly uses entropy-only filtering)
- EATA's condition-aware Fisher reweighting (used in multi-domain; single Fisher matrix sufficient here)

---

## Method 1: SAR-Continual

### Reference
Niu et al., *"Towards Stable Test-Time Adaptation in Dynamic Wild World"*, ICLR 2023.

### Mechanism

Three independent components composed at each adaptation step:

**A. Reliable sample filtering**
- Compute per-image mean pixel entropy from logits.
- If `mean_entropy > e_margin`, **skip the entire sample** (no gradient step, no MA update).
- Default `e_margin = 0.4 * ln(num_classes)` ≈ 1.198 for VOC20 (20 classes).

**B. SAM optimizer step** (replaces vanilla SGD/Adam)
- Forward pass → compute entropy loss.
- Compute gradient ∇L at current weights θ.
- Compute perturbation `ε = sam_rho * ∇L / ||∇L||₂`.
- Move weights to perturbed position θ + ε.
- Re-compute forward pass and gradient ∇L̃ at θ + ε.
- Restore weights to θ, then take gradient step with ∇L̃: `θ ← θ - lr * ∇L̃`.
- Default `sam_rho = 0.05`.

**C. Model recovery**
- Maintain exponential moving average of the loss: `loss_ma ← ema_factor * loss_ma + (1 - ema_factor) * current_loss`.
- If `loss_ma < e_0` after at least `recovery_warmup` batches, **reset all visual-encoder LN parameters to source snapshot** and clear `loss_ma`. Low EMA = entropy minimisation has driven the model to over-confident trivial predictions (SAR paper's collapse signal; matches `ema < 0.2` reset condition in the official `mr-eggplant/SAR` reference implementation).
- Default `e_0 = 0.1` for our 19/20-class setting (paper uses 0.2 for ImageNet's 1000 classes; scaled roughly by `ln(C)`). `ema_factor = 0.9`, `recovery_warmup = 50`.

### Loss
- Pure TENT loss (pixel-wise softmax entropy, no top-K filtering, single prompt template — mirrors `tent_divgate_continual`).

### What's trained
- Visual-encoder LayerNorm (γ, β) only — same as TENT/MLMP/DivGate.

### Required state
- Source weight snapshot (deepcopy at init, used for model recovery).
- `loss_ma` scalar.
- `total_batches` counter (for warmup gating).

### CLI args (added to `main_continual.py`)
`--e_margin`, `--sam_rho`, `--e_0`, `--ema_factor`, `--recovery_warmup`

---

## Method 2: EATA-Continual

### Reference
Niu et al., *"Efficient Test-Time Model Adaptation without Forgetting"*, ICML 2022.

### Mechanism

**A. Pre-stream Fisher computation** (one-time, before adaptation begins)
- Build a data loader over the clean VOC20 split (`corruption="original"`).
- Iterate up to `fisher_size` images (default 2000); for each:
  - Forward pass → pseudo-label `ŷ` (argmax of logits).
  - Compute cross-entropy with `ŷ` as target.
  - Backward; accumulate `(∂L/∂θ_i)²` for every visual-encoder LN parameter.
- After accumulation: `F_i = sum(grad²) / n_samples`. Store `self.fisher` as a dict mapping LN param name → Fisher tensor.
- Also store `self.src_params` = deepcopy of LN params at this point (anchor for EWC).

**B. Reliable + non-redundant sample filtering** (per adaptation step)
- *Reliable*: per-image mean entropy < `e_margin` (same default as SAR).
- *Non-redundant*: compute per-image average logit vector (mean over pixels), L2-normalise. Maintain `current_logit_ema` (EMA over recent reliable samples). If `cosine(current_sample_logits, current_logit_ema) > 1 - d_margin`, skip as redundant.
- If both filters pass: include in this batch's gradient update.

**C. Fisher-weighted EWC loss**
- Augment TENT loss: `L = L_TENT + fisher_alpha * Σ_i F_i * (θ_i - θ_i^src)²`.
- Sum is over all visual-encoder LN params with non-zero Fisher.
- Default `fisher_alpha = 2000`, `d_margin = 0.05`.

### Loss
- TENT loss (single prompt, all pixels) + Fisher-weighted EWC penalty.

### What's trained
- Visual-encoder LayerNorm (γ, β) only.

### Required state
- `self.fisher`: dict[str, Tensor] of Fisher values keyed by LN param name.
- `self.src_params`: dict[str, Tensor] of LN snapshot at Fisher computation time.
- `self.current_logit_ema`: running average for non-redundant filtering.
- `self.logit_ema_decay = 0.9` (internal constant).

### Pre-stream dispatch (in `main_continual.py`)
Mirror the existing DPCore pattern:
```python
if args.method == 'eata_continual':
    src_loader = prepare_data(args.dataset, args.data_dir, args.init_resize,
                              args.patch_size, args.patch_stride,
                              corruption="original",
                              batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=False)
    adapt_method.obtain_src_fisher(src_loader)
```
Use `corruption="original"` to read the clean VOC20 split (no `CorruptTransform` inserted).

### CLI args (added to `main_continual.py`)
`--e_margin`, `--d_margin`, `--fisher_alpha`, `--fisher_size`

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Create | `adapt/sam.py` | SAM optimizer wrapper (used by SAR) |
| Create | `adapt/sar_continual.py` | SAR-Continual main class |
| Create | `adapt/eata_continual.py` | EATA-Continual main class |
| Modify | `adapt/__init__.py` | Register `sar_continual` and `eata_continual` in METHOD_CLASSES |
| Modify | `main_continual.py` | (a) `add_method_specific_args` branches; (b) pre-stream dispatch for EATA Fisher computation |
| Create | `bash/v20/sar_continual.sh` | VOC20 runner for SAR |
| Create | `bash/v20/eata_continual.sh` | VOC20 runner for EATA |

No changes to `main.py`, `utils/`, `ovss/`, or any other adapt method.

---

## Bash Script Design

Both scripts follow the existing `bash/v20/tent_divgate_continual.sh` template:
- Same VOC20 patch convention (`INIT_RESIZE="224 224"`, `--patch_size 224 224 --patch_stride 112`)
- Same `CORRUPTIONS_ARRAY` bash array (commentable per-line, `CORRUPTIONS_LIST` env-var override)
- Same `SAVE_DIR` env-var override pattern
- Same shared hyperparams: `BATCH_SIZE=1`, `LR=0.00001`, `STEPS=1`, `CONTINUAL_ROUNDS=150`, `--seed 0`, `--class_extensions`

**SAR script (`sar_continual.sh`) additions**:
```bash
E_MARGIN=1.198          # 0.4 * ln(20) for VOC20
SAM_RHO=0.05
E_0=0.2
EMA_FACTOR=0.9
RECOVERY_WARMUP=50

SAVE_DIR="${SAVE_DIR:-save/${DATASET}/sar_continual/}"
```

**EATA script (`eata_continual.sh`) additions**:
```bash
E_MARGIN=1.198          # 0.4 * ln(20) for VOC20
D_MARGIN=0.05
FISHER_ALPHA=2000
FISHER_SIZE=2000

SAVE_DIR="${SAVE_DIR:-save/${DATASET}/eata_continual/}"
```

Override convention for subset runs (identical to existing scripts):
```bash
CORRUPTIONS_LIST="snow frost fog brightness contrast" \
SAVE_DIR="save/PascalVOC20Dataset/sar_continual_weather/" \
bash bash/v20/sar_continual.sh
```

---

## Output Format

Identical to existing continual methods:
- `save/PascalVOC20Dataset/{method}/results_all_rounds.txt` — header `Round, <c1>, <c2>, ..., Mean_mIoU`, one row per round.
- `save/PascalVOC20Dataset/{method}/round_NN/{corruption}/results.txt` — per-round per-condition mIoU/mDice/mAcc.
- `save/PascalVOC20Dataset/{method}/args.json` — full CLI args record.

**Method-specific logs** (lightweight, optional but recommended):
- SAR: `sar_log.txt` with columns `total_batches, mean_entropy, was_filtered, loss_ma, was_reset` (one row per batch).
- EATA: `eata_log.txt` with columns `total_batches, mean_entropy, was_reliable, was_non_redundant, ewc_loss, tent_loss` (one row per batch).

These logs allow post-hoc analysis of how often each filter activates and whether the recovery mechanism fires.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| SAM doubles compute per step (two forward+backward) | Document in script comment; expect ~2x wall time vs TENT |
| Fisher computation on 2000 images may be slow at full resolution | `fisher_size` is a hyperparameter; can lower to 500 for smoke testing |
| EATA's redundancy filter may starve adaptation if logits stable | Filter activation logged; if >50% samples filtered, lower `d_margin` |
| SAR model recovery never fires if `e_0` too high | Default `e_0=0.2` taken from original SAR paper; tunable per dataset |
| Fisher penalty may freeze adaptation if `fisher_alpha` too large | Default 2000 from EATA paper; if no adaptation observed, reduce by 10x |

---

## Success Criteria

The two methods are considered *successfully implemented* when:
1. Smoke test (`--debug` mode, 5 batches per condition, 2 conditions) completes without error and produces all expected output files.
2. Full 150-round run on weather-5 subset completes and produces a `results_all_rounds.txt` with monotonic Round numbers and finite mIoU values.
3. Method-specific logs show expected behaviour: SAR's filter activates on some fraction of samples; EATA's Fisher penalty is non-zero in loss output.

Whether either method *outperforms* TENT-DivGate is the **research question being tested**, not an implementation success criterion.

---

## What is NOT in scope

- No new visualisation/plotting scripts (existing `plot_voc20_methods_comparison.py` will auto-pick up new save_dirs after adding the method to `METHOD_STYLES`; that is a one-line change, deferred until results are in)
- No parser updates (no LaTeX table generation needed yet)
- No MLMP-base variants of SAR/EATA (deferred pending TENT-base results)
- No Cityscapes or ACDC versions (deferred pending VOC20 results)
- No hyperparameter sweep — use paper defaults for the initial run; sweep deferred until base behaviour is characterised
