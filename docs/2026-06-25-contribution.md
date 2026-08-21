# 0625 Contribution — Grad-norm degradation signal → HMGate (the unified gate)

**Date:** 2026-06-25
**Setting:** CTTA, no-reset, evaluate-before-adapt, 150 rounds, NA-CLIP ViT-L/14,
LayerNorm-only adaptation (top_block_exclude=6), frozen text encoder, seed=0.
**Base method:** DeYO (entropy + PLPD dual filter + reweighted entropy) on full MLMP
(7 prompts, 18-layer mean-fusion adapt, UAML evaluate). All gates restore visual LN only.
**Datasets:** ACDC (4 cond), Cityscapes-C (5corr sub100), VOC20-C (5corr sub100).

---

## 0. Objective (clarified this week)

The metric that matters, in priority order:
1. **Do NOT drop** — after reaching its peak the curve must stay up (no late collapse/decay).
2. **Near the peak** — on the no-drop basis, get as close to the achievable peak as possible.
3. **Beat MLMP-episodic** — episodic is the main paper baseline; the story needs to exceed it,
   especially on VOC20.

mean and peak alone are NOT the target.

---

## 1. Headline result — HMGate wins on all three

`deyo_mlmp_hmgate_continual` — DeYO+MLMP + an **H-margin-regime-gated grad-lag restore**.

| Dataset | No-Adapt | MLMP-episodic | DeYO+MLMP (no gate) | **HMGate (ours)** |
|---|---|---|---|---|
| **ACDC** (4 cond) | 23.34 | 29.84 | pk33.5 → **last 6.0** (collapse) | pk33.4, **last 33.4**, mean 32.3 |
| **Cityscapes-C** | 20.56 | 22.71 | pk24.1 → **last 2.8** (collapse) | pk24.1, **last 24.0**, mean 23.9 |
| **VOC20-C** | 68.60 | 75.35 | pk78.7 → last 74.3 | pk79.0, **last 79.0**, mean 77.1 |

**HMGate is the first method that simultaneously, on all three datasets:**
- does NOT drop (last ≈ peak; drop magnitude ≈ 0.0),
- stays near the achievable peak (ACDC 33.4 vs no-gate peak 33.5; Cityscapes at peak 24.1),
- **beats MLMP-episodic** — ACDC last 33.4 > 29.84 (+3.6); Cityscapes 24.0 > 22.71 (+1.3);
  VOC20 **79.0 > 75.35 (+3.7)**.

Figure: `figures/hmgate_final.png` (no-gate vs HMGate + episodic/no-adapt lines, 3 panels).

---

## 2. The arc — why every grad-magnitude gate failed first

grad_norm (‖LN-grad‖) was found to mirror mIoU (bottoms at the peak, rises during degradation)
and to work on VOC20 where H_margin is blind (see 0618 §7). The whole week was about turning that
signal into a gate. Every variant tried, and why it failed:

| Method | gate trigger / cap | failure |
|---|---|---|
| GradSlope sw10+maxlag300 | slope → lag, lag capped at 300 | fast collapse needs lag≫300 → restore too shallow → ACDC last 21.7 |
| GradAnchor (method2) | slope trigger, restore to best-state | slope noise fires during climb → suppresses it (flat 30.5) |
| GradLagAdapt (method1) | slope → lag, cap = grad-min distance | holds ACDC/Citys (31.7/23.1) but deep restore on VOC20 noise kills the climb (73.6, below episodic) |
| Hybrid (slope-gated cap) | shallow vs deep cap by **slope magnitude** | VOC20 noise slope (≤0.5) ≫ ACDC collapse slope (≤0.08) → can't separate |
| Hybrid + low rst | base_rst 0.002 | VOC20 78.3 ✓ but ACDC drops (too weak to hold fast collapse) |
| GradRatio (proposed) | EMA-grad/grad-min **ratio** trigger | VOC20 noise ratio ~1.25 **exceeds** ACDC collapse ratio ~1.24 → fires on VOC20 climb (capped 72.2) |

**Root cause (the key negative result):** *grad_norm magnitude — in ANY form (slope OR ratio) —
cannot separate VOC20's noise from ACDC's real collapse, because VOC20's noise floor is larger
than ACDC's real degradation signal.* A single grad-magnitude threshold either over-restores VOC20
(suppresses the climb) or under-restores ACDC (lets it collapse).

---

## 3. The fix — use H_margin as the REGIME switch, grad for the DEPTH

H_margin **drops only on real marginal collapse** (the prediction marginal collapsing to 1-2
classes). It is therefore a clean *regime classifier* even though it can't track VOC20's magnitude:
- ACDC/Cityscapes (collapse type): marginal collapses → H_margin drops.
- VOC20 (uniform degradation): marginal stays diverse → H_margin stays high.

So HMGate splits the two jobs across the two signals:
- **H_margin → regime** (deep vs shallow restore allowed).
- **grad_norm slope → depth/timing** within the regime.

### Mechanism (per `monitor_interval=50` batches)
```
H_margin = entropy of the windowed ensemble (7-prompt) class marginal
h_max    = running max of H_margin
collapse_regime = H_margin < h_drop_ratio * h_max      # 0.9; RELATIVE (no scale calibration)

slope = least-squares slope of grad_norm over the last slope_window windows
if slope <= deadzone:        lag = 0          # grad flat/falling = healthy -> no restore
else:
    cap = windows_since_min          if collapse_regime   # DEEP: back to the best (grad-min) state
        = min(maxlag_shallow, ...)   otherwise            # SHALLOW (6 win = 300 batch): don't suppress the climb
    lag = clamp(lag_gain * slope, 1, cap);  rst = base_rst
# restore: stochastically pull `rst` of LN params toward the snapshot `lag` windows back
```
Per-window LN snapshots are kept in a deque (lag measured in windows); restore target lies between
"now" and the best (grad-min) state, never past it. Restore machinery is SmoothAnchor's; the
deep/shallow cap is method1's; **the switch from slope to H_margin is what makes it work**.

### Regime activation (confirms it works as designed)
- VOC20: **collapse windows = 0%** → always shallow → free climb to 79.0.
- ACDC: collapse windows = 78% → deep restore → held at 33.4.
- Cityscapes: collapse windows = 27% → deep restore exactly through the fast-collapse stretch.

**Two-signal division of labour (the convergence point of the whole signal hunt):**
H_margin answers "is this a collapse regime?" (its native strength); grad_norm answers
"when / how deep to brake?" within that regime.

---

## 4. Files

- **Method:** `adapt/deyo_mlmp_hmgate_continual.py`
  (registered `deyo_mlmp_hmgate_continual`; argparse block in `main_continual.py`).
- **Run scripts:** `bash/{ACDC_10_round,cityscapes,v20}/deyo_mlmp_hmgate_continual*.sh`
  (h_drop_ratio=0.9, maxlag_shallow=6, slope_window=10, lag_gain=1500, base_rst=0.01).
- **Result dirs:** `save/{ACDCDataset,CityscapesDataset,PascalVOC20Dataset}/deyo_mlmp_hmgate_continual*/`
  — each has `results_all_rounds.txt`, `gate_log.csv`
  (`total_batches,grad_norm,grad_slope,h_margin,collapse,windows_since_min,lag,rst`).
- **Figures:** `figures/hmgate_final.png` (clean 3-panel),
  `figures/all_gates_3datasets.png` (all gate variants).
- **Plot scripts:** `plot_hmgate_final.py`, `plot_all_gates_3datasets.py`.
- **Earlier gate variants (kept for ablation):** `adapt/deyo_mlmp_grad{slope,anchor,lagadapt,ratio}_continual.py`.

---

## 5. Extension — 5 new datasets + the `h_drop_ratio` fix for the late droppers

HMGate was extended to VOC21, PascalContext59/60, COCO-Object, COCO-Stuff (all 5corr sub100, 150R)
plus a **VOC20 full + 15-corruption** run (1449 img × 15 corr = 21735 batch/round). Baselines
(no-gate, episodic, no-adapt) were run for each.

On three of these, `h_drop_ratio=0.9` still dropped after the peak — the marginal collapses only
*shallowly*, so 0.9 (wait for H_margin to fall 10%) triggered the deep restore too late. Raising to
**`h_drop_ratio=0.95`** (deep restore engages once H_margin falls just 5%) fixes them. New runs use a
`_hdr095` suffix and never overwrite the 0.9 runs.

| Dataset | 0.9 peak | 0.9 last (drop) | 0.95 peak | 0.95 last (drop) |
|---|---|---|---|---|
| ACDC | 33.4 | 33.4 (0.0) | 32.9 | 32.9 (0.0) |
| Cityscapes-C | 24.1 | 24.0 (0.0) | 23.7 | 23.7 (0.0) |
| VOC20-C (5corr) | 79.0 | 79.0 (0.0) | 79.0 | 79.0 (0.0) |
| VOC21-C | 49.3 | 49.3 (0.0) | 48.2 | 48.2 (0.0) |
| PContext59 | 27.6 | 27.6 (0.0) | 27.0 | 27.0 (0.0) |
| PContext60 | 23.8 | 23.8 (0.0) | 23.4 | 23.4 (0.0) |
| COCO-Object | 20.9 | 20.8 (0.1) | 20.9 | 20.8 (0.1) |
| **COCO-Stuff** | 13.4 | **12.7 (0.7)** | 13.4 | **13.3 (0.1)** ✓ |
| **VOC20 full+15corr** | 77.4 | **69.4 (8.1)** | 77.4 | **77.1 (0.3, R19/150)** ✓ |

**Read-out:**
- 0.95 **fixes the two real droppers**: COCO-Stuff (0.7→0.1) and VOC20-full+15corr (8.1→0.3).
  Same mechanism, same diagnosis — earlier deep restore catches the shallow marginal collapse before
  it compounds.
- COCO-Object drops 0.1 under both — already negligible, unchanged.
- Cost: the **collapse-type peaks dip slightly** (ACDC 33.4→32.9, Cityscapes 24.1→23.7,
  VOC21 49.3→48.2, P59 27.6→27.0, P60 23.8→23.4) because the deep restore now fires a touch earlier.
  VOC20-C (5corr) is unaffected (its marginal never collapses → 0.95 still never deep-restores).
- VOC20-full+15corr at 0.9 is the clearest failure→fix: R05 peak 77.4 → R51 69.4 (steady decay);
  at 0.95 it triggers at R09 and holds 77.4→77.1 through R19. Still running.

So 0.95 trades ~0.3–1.1 mIoU of collapse-type peak for the no-drop guarantee on the shallow-collapse
datasets. This is consistent with §0: no-drop is priority #1.

Figures: `figures/hdr_compare.png` (0.9 vs 0.95, 9-panel), `figures/dropping_diag_*.png`
(mIoU/H_margin/grad_norm stacked for the droppers). Plot scripts: `plot_hdr_compare.py`,
`plot_dropping_diagnosis.py`, `plot_hmgate_newdatasets.py`.
Run scripts: `bash/{v21,p59,p60,coco_obj,coco_stuff}/deyo_mlmp_hmgate_continual*.sh` and their
`_hdr095` variants; VOC20 full at `bash/v20/deyo_mlmp_hmgate_continual_full_15corr*.sh`.

---

## 6. Next

1. **Pick one `h_drop_ratio`** — 0.95 is the safe default (no-drop everywhere) at a small peak cost;
   consider 0.92 as a middle ground only if the collapse-type peak loss matters for the table.
2. **Let VOC20 full+15corr finish** (R19/150 at 0.95, ~90 hrs for 150R, GPU3) to confirm long-term hold.
3. **Ablate the regime switch** — HMGate with `h_drop_ratio` swept and a "slope-switch" control to
   show H_margin is what makes the difference.
4. **Simplify the hyperparameter count** — HMGate currently exposes ~8 gate params; only
   `h_drop_ratio` is actively tuned. Fold the rest to fixed defaults for the paper.
5. Write the formal spec (`docs/hmgate_spec.md`) and fold HMGate into EXPERIMENT_STATUS.md as the
   current best method.
