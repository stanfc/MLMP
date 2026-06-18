# 0618 Contribution — DeYO + MLMP + SmoothAnchor (gate ablation, 3 datasets)

**Date:** 2026-06-18
**Setting:** CTTA, no-reset, evaluate-before-adapt, 150 continual rounds, NA-CLIP ViT-L/14,
LayerNorm-only adaptation (top_block_exclude=6), frozen text encoder, seed=0.
**Datasets:** ACDC (4 cond, fog/night/rain/snow), Cityscapes-C (5corr sub100), VOC20-C (5corr sub100).

---

## 1. What was done

Built and ran the full **DeYO+MLMP gate ablation** on all three datasets, 150 rounds each:

- **DeYO+MLMP (no gate)** — DeYO adapt loss (entropy + PLPD dual filter + reweighted entropy)
  on top of the full MLMP machinery (multi-prompt + multi-layer UAML eval). No restoration.
- **DeYO+MLMP + DivGate** — discrete 3-tier H_margin gate (aggressive / cautious / brake),
  stochastic restoration toward source.
- **DeYO+MLMP + SmoothAnchor** — continuous `lag(H)` gate: restore toward a lagged LN snapshot,
  depth of lag scales with how far H_margin has dropped.

Figure: `figures/deyo_mlmp_smooth_full.png` (3 panels, 5 lines each: the three above +
MLMP-episodic and No-Adapt reference lines).

---

## 2. Headline results (mean mIoU, 150R)

| Dataset | No-Adapt | DeYO+MLMP (no gate) | + DivGate | + SmoothAnchor (orig) | + SmoothAnchor (recal) |
|---|---|---|---|---|---|
| **ACDC** (4 cond) | 23.3 | peak 33.5 → **last 6.0** (mean 20.6) | peak 33.1, last 30.1, **mean 31.8** | peak 31.4, last 31.3, mean 31.1 | **peak 33.6**, last 28.2, mean 31.4 |
| **Cityscapes-C** | 20.6 | peak 24.1 → **last 2.8** (mean 11.7) | peak 24.1, last 23.7, **mean 23.6** | peak 23.9, last 23.6, mean 23.6 | peak 24.1, last 23.2, mean 23.3 |
| **VOC20-C** | 68.6 | peak 78.7, last 74.3, mean 77.4 | (identical to no-gate) 77.4 | peak 78.7, **last 76.7, mean 77.7** | — (not recalibrated) |

ACDC baselines are proper 4-condition means (fog/night/rain/snow), from
`save_tekai/save/ACDCDataset/{No_Adaptation, mlmp_episodic_step_1}`:
No-Adapt mean **23.34**, MLMP-episodic mean **29.84**.

**Takeaways:**
- **Both gates defeat collapse on the collapse-type datasets (ACDC, Cityscapes).** No-gate crashes
  (ACDC 33.5→6.0, Cityscapes 24.1→2.8); DivGate and SmoothAnchor hold to R150.
- **DivGate is the best overall on ACDC** (mean 31.8, last 30.1): reaches near-peak and holds.
- **VOC20 is a different regime** (see §4): no collapse, gates barely matter; SmoothAnchor's
  always-on mild restore happens to edge out no-gate (77.7 vs 77.4).

---

## 3. Key finding — SmoothAnchor H-scale miscalibration (and fix)

The **original SmoothAnchor underperformed because its H window was set on the wrong scale.**

- Constructor assumed healthy H_margin ≈ 3.0, so `h_ceil=2.9, h_floor=2.2`.
- But DeYO+MLMP's **actual internal-H range is only [2.24, 2.51]** (gate_log) — the whole window
  sat *above* the observed H. Result: **restore was active 100% of the time, from batch 50**,
  dragging the model back during the healthy climb → capped the peak (ACDC stuck at 31.4, never
  reaching the ~33.5 that no-gate/DivGate hit).

**Calibration method (transferable):** the evaluate-based H (entropy_log) maps cleanly onto the
gate's internal-H via a fixed offset:

```
internal_H ≈ eval_H + 0.24   (measured on ACDC and Cityscapes; same method, same offset)
```

Using the no-gate eval-H trajectory, the critical points map to:

| Phase | eval_H | internal_H |
|---|---|---|
| Peak (ACDC R34, mIoU 33.5) | ~1.80 | **~2.05** |
| Collapse onset | ~1.6–1.7 | ~1.9–2.0 |
| Deep collapse | <1.4 | <1.65 |

This lines up exactly with the **proven DivGate threshold** (`h_threshold≈2.0`): restore should
stay OFF while H ≥ ~2.0 (free climb to peak), and engage only as H drops below it.

**Recalibrated windows:**

| Dataset | DivGate h_threshold (ref) | SmoothAnchor h_ceil | h_floor | lag_scale |
|---|---|---|---|---|
| ACDC | 2.0 | **2.0** | **1.6** | 90 |
| Cityscapes | 2.1 | **2.1** | **1.7** | 90 |
| VOC20 | 1.9 | *unchanged* (2.9) | 2.2 | 150 |

**Effect of recalibration (ACDC):** peak **31.4 → 33.6** (now reaches the true peak, matching
no-gate/DivGate), mean 31.1 → 31.4. **Remaining trade-off:** the recalibrated run reaches peak
but then drifts down to last 28.2 (vs DivGate's 30.1) — late-stage restore (`rst=0.005`) is a bit
too weak to fully hold once H sinks to ~1.7. Candidate next tweaks: `rst 0.005→0.01`, or
`h_floor 1.6→1.7` (engage deep restore earlier).

---

## 4. Two collapse regimes (why VOC20 is different)

| Regime | Datasets | Signature | Does H-gate help? |
|---|---|---|---|
| **Collapse (塌縮型)** | ACDC, Cityscapes | marginal collapses to 1–2 classes; **H_margin drops** (ACDC eval-H 2.2→0.75) | **Yes** — H is a clean collapse detector; gate restores in time |
| **Uniform degradation (均勻退化)** | VOC20 | mild drift (78.7→74.3) with **H_margin staying high ~3.0** (diverse marginal) | **No** — H never drops below threshold, so the gate never fires; DivGate result is *bit-identical* to no-gate |

On VOC20, applying the collapse-style recalibration (`h_ceil≈1.9`) would disable restore entirely
(H is always > 1.9) → regress to no-gate (last 74.3, worse than the current 76.7). VOC20's
degradation is invisible to an H_margin gate and needs a different signal (治本).

---

## 5. Files

- **Figure:** `figures/deyo_mlmp_smooth_full.png`, plot script `plot_deyo_mlmp_smooth_full.py`
- **Method impls:** `adapt/deyo_mlmp_continual.py`, `adapt/deyo_mlmp_divgate_continual.py`,
  `adapt/deyo_mlmp_smooth_anchor_continual.py`
- **Run scripts (recalibrated):**
  - `bash/ACDC_10_round/deyo_mlmp_smooth_anchor_continual.sh` (h_ceil=2.0, h_floor=1.6, lag_scale=90)
  - `bash/cityscapes/deyo_mlmp_smooth_anchor_continual_5corr_sub100.sh` (h_ceil=2.1, h_floor=1.7, lag_scale=90)
  - `bash/v20/deyo_mlmp_smooth_anchor_continual_5corr_sub100.sh` (unchanged)
- **Result dirs:**
  - `save/ACDCDataset/deyo_mlmp_smooth_anchor_continual_recal_h2.0_1.6/`
  - `save/CityscapesDataset/deyo_mlmp_smooth_anchor_continual_recal_h2.1_1.7_5corr_sub100/`
  - originals (pre-recal) kept alongside for comparison.
- Each result dir has `results_all_rounds.txt`, `entropy_log.csv` (evaluate-based H),
  `gate_log.csv` (gate internal-H, lag, rst).

---

## 6. Open items

1. ACDC SmoothAnchor: close the peak-vs-hold gap (try `rst=0.01` or `h_floor=1.7`) so it both
   reaches 33.6 **and** holds like DivGate (~30).
2. VOC20 (治本): find a degradation signal that isn't H_margin (H is blind to uniform drift).
   → **Addressed in §7 (signal hunt).**
3. Decide headline method for the paper: **DivGate** currently wins on ACDC mean (31.8); SmoothAnchor
   is competitive and reaches a higher peak but needs the hold fix.

---

## 7. Degradation-signal hunt (beyond H_margin)

**Motivation:** H_margin gates collapse-type degradation (ACDC, Cityscapes) but is blind to
VOC20's *uniform degradation*. Goal: instrument many candidate signals, run no-gate DeYO+MLMP on
all three datasets, and find which signal(s) track / flag the mIoU drop — especially on VOC20.

### 7.1 Instrumentation (side-effect-free)

Built a 23-signal monitor, opt-in via `--log_signals` (off by default), writing per-batch
`signals_log.csv`. **Verified it does NOT perturb adaptation:** with vs without `--log_signals`,
`results_all_rounds.txt` is **bit-identical** (the model forward has `dropout_p=0` / no RNG, so the
extra diagnostic forward does not shift DeYO's `torch.randperm` RNG; the per-batch adapt-internal
signals are just stashed from already-computed tensors).

- Monitor: [utils/collapse_signals.py](utils/collapse_signals.py)
- Method diagnostics (`diagnose()` + `self.diag`): [adapt/deyo_mlmp_continual.py](adapt/deyo_mlmp_continual.py)
- Flag wiring: [main_continual.py](main_continual.py) (`--log_signals`)
- Run scripts: `bash/{ACDC_10_round,cityscapes,v20}/deyo_mlmp_continual_monitor*.sh`
- Output dirs: `save/{ACDCDataset,CityscapesDataset,PascalVOC20Dataset}/deyo_mlmp_continual_monitor*/signals_log.csv`

23 signals across 4 families: marginal-distribution (h_margin, max_marginal, n_active_classes,
gini, kl_marg_ref…), per-pixel confidence (mean_conf, frac_conf_high/low, logit_gap…), temporal/
spatial (pred_hist_drift, conn_components), and model-side (**ln_param_drift, grad_norm,
prompt_disagree, layer_disagree, plpd_mean, feat_norm, feat_text_align, filter_pass_rate**).

### 7.2 Two-facet analysis (key reframing)

A signal can help a gate in **two different ways**, and they rank signals differently:

- **Facet 1 — TREND:** does the signal move with mIoU? `|Spearman(signal, mIoU)|`. High → tracks the
  *magnitude* of degradation (good for a continuous brake).
- **Facet 2 — THRESHOLD usability:** is the signal **monotone in time** so a single threshold is
  crossed once = a clean one-shot trigger? `|Spearman(signal, round)|`. (We rejected a pre/post-peak
  AUC metric: it is ~1 for ANY monotone signal regardless of mIoU, so it doesn't discriminate.)

A gate needs Facet 2 (a usable trigger level); Facet 1 tells it how hard to brake.

### 7.3 Findings

| Signal | ACDC trend | Citys trend | **VOC20 trend** | VOC20 monotone | value@peak (ACDC/Citys/VOC20) |
|---|---|---|---|---|---|
| **grad_norm** | 0.90 | 0.88 | **0.84** | 0.01 (U-shape) | 5.65 / 6.24 / 6.00 |
| **h_margin** | 0.94 | 0.98 | 0.14 | **1.00** | 1.77 / 2.05 / 1.61 |
| **mean_conf** | 0.93 | 0.93 | 0.14 | 1.00 | 0.85 / 0.71 / 0.73 |
| ln_param_drift | 0.94 | 0.99 | 0.14 | 1.00 | 6.16 / 2.72 / 4.55 |

1. **grad_norm is the only signal whose *trend* survives on VOC20** (Pearson −0.92, Spearman −0.84):
   as mIoU drifts down, the self-supervised gradient on LN shrinks — a dynamics signal that does
   **not** rely on marginal collapse, so it sees the uniform drift that H_margin cannot. But it is
   **U-shaped** (bottoms near the peak), so a single threshold is crossed twice → poor *trigger*.
2. **For thresholding, the picture inverts** (user's insight): H_margin / mean_conf / most marginal
   signals are **perfectly monotone in time on all three datasets** (monotone = 1.00) with a
   reasonably consistent value at the peak. So a threshold like "restore when h_margin < ~1.7" **does
   fire near the peak on VOC20 too** (≈R45–59), even though its *trend* correlation is only 0.14.
   ⇒ "low trend correlation" ≠ "useless for a gate". This corrects the earlier "H is blind to VOC20"
   claim: H is blind to the *magnitude*, not to the *peak crossing*.
3. **`mean_conf` (mean max-softmax) has the most consistent peak value across datasets**
   (0.71–0.85) → best candidate for a single *universal* threshold; h_margin (1.6–2.05) is next;
   ln_param_drift is inconsistent (model-scale dependent).

**Gate-design implication:** single trigger → use a monotone signal (mean_conf / h_margin) with the
threshold at its peak value; continuous brake strength → use grad_norm (magnitude). A composite
(mean_conf threshold fires the gate; grad_norm sets restore depth) should generalize to all three.

### 7.4 Figures

- **[figures/signal_corr_heatmap.png](figures/signal_corr_heatmap.png)** — 23 signals × 3 datasets,
  Spearman(signal, mIoU) trend heatmap. VOC20 column is flat except grad_norm.
- **[figures/signal_two_facets.png](figures/signal_two_facets.png)** — side-by-side TREND (green) vs
  THRESHOLD-monotonicity (purple) heatmaps; the two columns are complementary on VOC20.
- **[figures/signal_trends_ACDC.png](figures/signal_trends_ACDC.png)**,
  **[figures/signal_trends_Cityscapes.png](figures/signal_trends_Cityscapes.png)**,
  **[figures/signal_trends_VOC20.png](figures/signal_trends_VOC20.png)** — each signal (blue) twin-axed
  with mIoU (red) per round.
- Analysis scripts: [plot_signal_correlation.py](plot_signal_correlation.py) (trend),
  [plot_signal_two_facets.py](plot_signal_two_facets.py) (trend vs threshold).

### 7.5 The two H_margin computations (important — they are different signals)

Two different "H_margin" numbers appear in this project and they are **not comparable**:

| | **evaluate-based H_margin** | **gate-internal H_margin** |
|---|---|---|
| Where | `entropy_log.csv`, `signals_log.csv` (col `h_margin`) | `gate_log.csv` (DivGate / SmoothAnchor) |
| Forward path | single **averaged-prompt** evaluate forward, `adaptive_weighted_mean` layer fusion | **adapt ensemble**: 7 individual prompts + `mean` fusion over 18 layers |
| Smoothing | **per-batch, no window** | **windowed mean** over `monitor_interval=50` batches |
| VOC20 range | min 0.07 / max 2.85 / **mean 1.57** (spread 2.78) | min 2.50 / max 3.07 / **mean 2.96** (spread 0.57) |
| Look | noisy, spread | tight, concentrated ~3.0 |

The figure `v20_divgate_gate_hmargin.png` plots the **gate-internal** value (windowed ensemble) —
that's why it sits tightly around ~3.0. The new signal panel uses the **evaluate-based per-batch**
value — naturally spread 0.07–2.85. Same name, different signal; the offset is roughly
`internal_H ≈ eval_H + 0.24` on ACDC/Cityscapes and `≈ eval_H + 1.4` on VOC20. Both nonetheless
agree that H_margin does not *track* VOC20's mIoU (gate-H stays flat ~3.0; eval-H trend corr 0.14).

### 7.6 Next

- Prototype a **composite gate** (mean_conf trigger + grad_norm-scaled restore) and test on all three.
- Sanity-check that mean_conf's ~0.7–0.85 peak band gives a usable fixed threshold across datasets.
