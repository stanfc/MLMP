# Gradient-Norm Penalty in DeYO+MLMP+DivGate — Design Spec

**Date:** 2026-06-21
**Author:** tekai (design dialogue with Claude)
**Status:** Approved design, pending implementation
**Setting:** CTTA, no-reset, evaluate-before-adapt, 150 continual rounds, NA-CLIP
ViT-L/14, LayerNorm-only adaptation (`top_block_exclude=6`), frozen text encoder,
seed=0.

---

## 1. Motivation

學長's 0618 signal hunt ([docs/2026-06-18-contribution.md](../../2026-06-18-contribution.md)
§7) found that **`grad_norm` (the L2 norm of the adapt loss gradient over the
trainable LN params) is highly negatively correlated with mIoU** across all three
datasets — and is the *only* signal whose trend survives on VOC20's "uniform
degradation" regime (trend |Spearman| = 0.84; Pearson −0.92), where the H_margin
diversity gate is blind (H stays ~3.0, gate never fires, DivGate == no-gate).

That work used `grad_norm` only as a **passive monitoring / gating signal**
(GradSlope gate, composite gate). This spec proposes the next step: turn
`grad_norm` into an **active optimisation target** — add it as a penalty term to
the adaptation loss so the optimiser directly avoids the high-`grad_norm` states
that the correlation says are bad.

**Hypothesis:** `L' = L + λ·‖∇_θ L‖²` keeps the model in a flatter, lower-gradient
LayerNorm configuration, delaying/preventing the degradation that DivGate cannot
catch on VOC20, while not hurting the collapse-type datasets (ACDC, Cityscapes)
where DivGate already works.

### 1.1 Known conceptual risk (must be tracked, not ignored)

Penalising `grad_norm` pushes θ toward low-gradient regions. That includes
**flat minima** (good — this is the Sharpness-Aware Minimization story) but *also*
the **fully-collapsed degenerate state** (CLAUDE.md §collapse: "Phase 4 …
gradient ≈ 0; locked"). 學長's data shows `grad_norm` is *U-shaped*: high at R1,
**bottoms at the mIoU peak**, then rises during degradation — but the runs never
reached the fully-locked zero-gradient state because the gate protected them. So a
penalty could in principle either (a) help (flat-minimum seeking) or (b) accelerate
collapse. **Keeping the H_margin DivGate as a safety net is therefore part of the
design, not optional.** Resolving (a) vs (b) is exactly what this experiment tests.

---

## 2. Relationship to existing work (why this is novel, not SAR)

`min_θ [L(θ) + ρ‖∇L‖]` is the first-order objective of SAM (Sharpness-Aware
Minimization), which SAR uses as its optimiser. We deliberately do **NOT** use SAM
here, because:

- SAM only *implicitly* approximates the penalty via a perturb-then-descend step;
  the knob is the perturbation radius ρ, not a direct `grad_norm` coefficient.
- Wrapping DeYO+MLMP in SAM would reduce the contribution to "apply SAM to a new
  base", overlapping the existing SAR family and decoupling from the measured
  `grad_norm` signal.

This spec uses the **explicit second-order penalty** `L + λ‖∇_θ L‖²` computed by
double-backward (`create_graph=True`). The λ coefficient *is* the "grad_norm
strength" knob, directly tied to the empirical finding. (SAM remains available as
an optional sanity-check baseline but is out of scope for the primary experiment.)

---

## 3. Method

### 3.1 Base

Copy `adapt/deyo_mlmp_divgate_continual.py` →
`adapt/deyo_mlmp_gradpen_divgate_continual.py`, class
`DeYOMLMPGradPenDivGateContinual`. **The H_margin DivGate machinery
(`_update_mode`, `_stochastic_restore_flat`, gate thresholds, `gate_log.csv`) is
preserved verbatim.** Only the loss/backward block changes.

### 3.2 The penalty (the only code change)

In `adapt()`'s inner loop, replace the original
`loss.backward(); optimizer.step()` with:

```python
ln_params = [p for _, p in self.named_ln_params]
if self.grad_pen_lambda > 0.0:
    # explicit gradient-norm penalty (double backward)
    g = torch.autograd.grad(loss, ln_params, create_graph=True)
    grad_sq = sum((gi.float() ** 2).sum() for gi in g)        # ‖∇L‖²  (fp32)
    penalty = grad_sq if self.grad_pen_form == 'sq' \
              else grad_sq.clamp(min=1e-12).sqrt()             # ‖∇L‖² or ‖∇L‖
    total = loss + self.grad_pen_lambda * penalty
    total.backward()
    grad_norm_raw = float(grad_sq.detach().sqrt())
    penalty_val = float(penalty.detach())
else:
    # λ == 0: EXACT original path → provably bit-identical to divgate baseline
    loss.backward()
    with torch.no_grad():
        gsq = sum((p.grad.float() ** 2).sum()
                  for _, p in self.named_ln_params if p.grad is not None)
    grad_norm_raw = float(gsq.sqrt())
    penalty_val = 0.0
self._log_gradpen(grad_norm_raw, penalty_val)
self.optimizer.step()
self.optimizer.zero_grad()
if self.current_rst > 0.0:
    self._stochastic_restore_flat(self.current_rst)
```

### 3.3 Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Penalty form (primary) | **squared `‖∇L‖²`** | gradient `2λ·H·∇L` is smooth even as `‖∇L‖→0`; linear form's `H·∇L/‖∇L‖` diverges near zero. Penalising `‖∇L‖²` monotonically penalises `‖∇L‖`, so the narrative holds. |
| Penalty form (variant) | `--grad_pen_form linear` available | documented fallback if user wants to penalise the literally-measured `‖∇L‖`. |
| Timing | **always-on** (every adapt step) | most faithful to "add to the loss"; SAM-analogous. |
| Timing fallbacks (NOT implemented now) | gate-gated (only in cautious/brake mode); warmup-then-on | documented as the next things to try **if always-on suppresses the peak** (the mis-calibrated-SmoothAnchor failure mode). |
| Precision | penalty computed in **fp32** (`.float()`) | NA-CLIP is fp16; double-backward + squaring underflows in fp16 (cf. memory `fp16_clip_grad_gotchas`). |
| λ=0 path | **branch to the exact original `loss.backward()`** | guarantees the λ=0 control reproduces the divgate baseline bit-identically (no `create_graph` numerical drift). |
| Gate | **H_margin DivGate preserved** | safety net against the collapse risk in §1.1. |

### 3.4 New CLI args (method-specific, in `main_continual.py`)

All of `deyo_mlmp_divgate_continual`'s args, plus:
- `--grad_pen_lambda` (float, default `0.0`) — the strength knob.
- `--grad_pen_form` (str, default `sq`, choices `sq`/`linear`).

### 3.5 New logging

`<save_dir>/gradpen_log.csv`, header `total_batches,grad_norm,penalty,lambda`,
one row per adapt step that takes a gradient step. This is the **mechanism check**:
verify that λ>0 actually drives the raw `grad_norm` trajectory down vs λ=0.

---

## 4. Experiment grid

Per-dataset proven DivGate config (from
`adapt/deyo_mlmp_divgate_continual.py` docstring + the existing bash scripts):

| Dataset | h_threshold | h_warning | cautious_rst | brake_rst | corruptions | resize / subset |
|---|---|---|---|---|---|---|
| ACDC | 2.0 | 1.7 | 0.005 | 0.02 | fog/night/rain/snow | 1120×560 / — |
| Cityscapes | 2.1 | 1.8 | 0.005 | 0.02 | snow/frost/fog/brightness/contrast | 1120×560 / sub100 |
| VOC20 | 3.0 | 2.7 | 0.005 | 0.02 | snow/frost/fog/brightness/contrast | 224×224 / sub100 |

### Stage 1 — λ main effect (gate fixed at proven config), 150R

λ ∈ **{0, 0.003, 0.01, 0.03, 0.1}** (squared form) on each dataset → **15 runs**.
- λ=0 reproduces the divgate baseline (control + correctness anchor).
- λ scale is unknown a priori (`‖∇L‖²` ~ 25–36 at peak, `L` ~ O(1–3)); this is a
  log grid spanning ~1.5 orders. After the first run, recalibrate the grid from the
  observed `penalty / loss` ratio in `gradpen_log.csv` if needed.

### Stage 2 — gate interaction (only if Stage 1 shows a positive λ), ~6 runs

Take each dataset's best λ, sweep gate ∈ {looser, tighter}
(e.g. h_threshold ±0.1–0.2, or cautious_rst 0.005→0.01).

**VOC20 is the cleanest test of the hypothesis:** its DivGate never fires (H~3.0),
so on VOC20 the λ penalty is the *only* active defence — directly probing whether
`grad_norm` repairs the gate's uniform-degradation blind spot.

---

## 5. Files

| File | Action | Responsibility |
|---|---|---|
| `adapt/deyo_mlmp_gradpen_divgate_continual.py` | create | method = divgate base + grad-norm penalty + gradpen_log |
| `adapt/__init__.py` | modify | import + registry entry |
| `main_continual.py` | modify | method-specific args block (`--grad_pen_lambda`, `--grad_pen_form`) |
| `bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh` | create | ACDC runner, `GRAD_PEN_LAMBDA` env-overridable |
| `bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh` | create | VOC20 runner |
| `bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh` | create | Cityscapes runner |
| `bash/sweep_gradpen_divgate.sh` | create | Stage-1 λ sweep launcher (packed GPUs, staggered, `gradpen_l<λ>` save suffix) |
| `scripts/plot_gradpen_sweep.py` | create | per-λ mean/peak/last table + grad_norm-vs-λ + mIoU-vs-round figures |
| `docs/EXPERIMENT_STATUS.md` | modify | new Phase O section documenting the experiment |

---

## 6. Success criteria

1. **Correctness:** λ=0 run is bit-identical to the existing `deyo_mlmp_divgate_continual`
   baseline (`results_all_rounds.txt` matches).
2. **Mechanism:** λ>0 measurably lowers the raw `grad_norm` trajectory vs λ=0
   (`gradpen_log.csv`). If it does not, the penalty is not engaging — stop and debug.
3. **Main result:** some λ>0 beats λ=0 (divgate baseline) on 150R mean mIoU.
4. **Milestone:** beats MLMP-episodic (ACDC 29.84 step1 / 30.6 step10, VOC20 76.21,
   Cityscapes 20.0).
5. **Story:** on VOC20 (gate-blind regime) λ>0 reduces the post-peak drift that
   DivGate cannot, confirming `grad_norm` complements `H_margin`.

## 7. Out of scope (documented fallbacks, not built now)

- SAM-based variant (option B from brainstorming).
- Per-layer / curvature-weighted penalty.

---

## 8. REVISION (2026-06-22) — smoke calibration killed the always-on raw penalty; redesign to penalise grad_norm *rise*

### 8.1 What the smoke found
On VOC20 (subset 20, 2–3 rounds) the always-on raw penalty has **no usable
"effective + stable" band**:

| form | λ | behaviour |
|---|---|---|
| sq | 1e-4, 3e-4 | ≈ baseline (no effect) |
| sq | **1e-3** | **collapse** (mIoU 55→0.23 after one step) |
| linear | 3e-3, 1e-2, 3e-2 | stable but ≈ baseline |
| linear | **0.1, 0.3, 1.0** | **collapse** |

Verified separately that λ=0 is **bit-identical** to the divgate baseline (correctness
of the method holds) — the problem is the *strategy*, not the code.

### 8.2 Root cause (confirms the brainstorming risk)
`grad_norm` is high during **both** healthy adaptation (the U-shape left arm, R1→peak)
**and** degradation (right arm). An always-on penalty on the *raw* norm cannot tell
them apart: gentle → ignores everything; strong enough to bite degradation → also kills
the healthy gradient → one-step collapse. Penalising the *absolute* grad_norm is wrong.

### 8.3 Redesign: penalise the RISE of grad_norm above a healthy running baseline
Keep the same file + DivGate (λ=0 stays the bit-identical control). Add a mode knob;
the penalised quantity becomes `relu(‖∇L‖ − g_ref)` so it is **zero during the healthy
phase** (no interference) and only bites when grad_norm regresses upward (degradation).

New CLI args (supersede §3.4's two-arg list):
- `--grad_pen_lambda` (float, default 0.0) — strength.
- `--grad_pen_form` (sq/linear, default linear) — applies to the *penalised quantity*
  (raw norm in `raw` mode, the excess in `excess`/`ema`). **Default switched to
  `linear`** (sq is a knife-edge per §8.1).
- `--grad_pen_mode {raw, excess, ema}` (default `raw`) — `raw` = always-on (kept as the
  known-bad reference); `excess`/`ema` = penalise rise above a baseline.
- `--grad_pen_ema_decay` (float, default 0.99) — α for `ema` mode.
- `--grad_clip` (float, default 0.0 = off; recommend ~10) — `clip_grad_norm_` on LN
  params after `total.backward()`, **only when grad_pen_lambda>0** (keeps λ=0 pristine);
  clips the occasional grad_norm spike (~88) that can one-shot the model.

Baseline `g_ref` (a detached scalar) updated every `monitor_interval` window from the
window-mean per-step grad_norm `gw`:
- `excess`: `g_ref = min(g_ref, gw)`  (rise above best-ever-healthy floor)
- `ema`:    `g_ref = α·g_ref + (1−α)·gw`  (rise above recent smoothed level)
- both: `g_ref` starts undefined → penalty 0 for the first window (natural warmup).

Penalty term: `gnorm = sqrt(grad_sq); excess = relu(gnorm − g_ref);
penalty = excess**2 if form=='sq' else excess;  total = loss + λ·penalty`. When
`excess==0`, `total.backward()` equals `loss.backward()` for that step.

### 8.4 Why keep the DivGate (answer to "do we add the gate?")
Yes. λ=0 → proven divgate baseline (clean control, no attribution confound). The
rise-penalty only fires on degradation, so it is **complementary** to the gate, not
redundant. On VOC20 the gate never fires → the penalty stands alone there (the key
hypothesis test). A no-gate ablation is optional future work.

### 8.5 Revised experiment grid
Compare **both** `excess` and `ema` (user request). λ must be **recalibrated** for the
excess form on the 20-image subset (same method as §8.1) before the 150R sweep, because
`relu(‖∇L‖ − g_ref)` is smaller than the raw norm → needs a different λ scale. Sweep:
`grad_pen_mode ∈ {excess, ema} × λ ∈ {recalibrated grid}`, `--grad_pen_form linear`,
`--grad_clip 10`, gate at each dataset's proven config; plus λ=0 (divgate) control and
optionally one `raw` reference. VOC20 first (cleanest), then ACDC / Cityscapes.
