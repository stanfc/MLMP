# 0702 Contribution — GradDivGate-PA: a permanent best-state anchor fixes the full-dataset drift

**Date:** 2026-07-02
**Setting:** CTTA, no-reset, evaluate-before-adapt, NA-CLIP ViT-L/14, LayerNorm-only
(top_block_exclude=6), frozen text encoder, seed=0. sub100 runs = 150 rounds;
full-dataset runs = 50 rounds.
**Base:** DeYO (entropy + PLPD dual filter + reweighted entropy) on full MLMP
(7 prompts, 18-layer mean-fusion adapt, UAML evaluate). All restores touch visual LN only.

---

## 0. Naming (settled this week)
- **GradDivGate (GDG)** = last week's "HMGate". Grad slope (trigger + depth) + Diversity/H_margin
  (regime cap). Renamed to disambiguate from `divgate` (which is H_margin only).
- **GradDivGate-PA (GDG-PA)** = GDG with a **permanent best-state anchor** (this week's headline).
- Code names unchanged: `deyo_mlmp_hmgate_continual` (GDG), `deyo_mlmp_hmgate2_continual` (GDG-PA).

---

## 1. The problem: last week's `h_drop_ratio=0.95` "fix" was illusory on full datasets

0625 §5 reported `h_drop_ratio=0.95` fixing VOC20-full+15corr (77.1 at R19). **Run to completion it
also drops** — the early reading was just a slower decay caught before it fell. Full-dataset, run to
the end:

| VOC20 full+15corr | last (R150) | drop from peak 77.4 |
|---|---|---|
| GDG 0.9 (rolling anchor) | 59.5 | 17.9 |
| GDG 0.95 (rolling anchor) | 63.5 | 13.9 |
| no-gate | 34.0 | 43.3 |

So **`h_drop_ratio` does not solve the full-dataset drift** — both 0.9 and 0.95 decay to ~60.
The sub100 "no-drop" property did **not** transfer to the full benchmark.

---

## 2. Diagnosis: buffer eviction (the rolling anchor slides down)

GDG keeps per-window LN snapshots in a deque of `max_windows=2000` windows (= 100k batches). The
deep restore targets the snapshot `windows_since_min` back (the grad-min = peak state). But the
deque only spans a fixed **batch** budget:

| run | batch/round | deque coverage |
|---|---|---|
| sub100 5corr | 500 | **200 rounds** (whole run fits) |
| VOC20 full+15corr | 21,735 | **4.6 rounds** |

On full datasets the peak snapshot is **evicted by ~R13**; after that the deep restore falls back to
`_win_buf[0]` — the oldest *still-held* (already-degraded) window — which itself slides forward and
down. So the reachable anchor drifts → residual −0.05/round decay. Confirmed in `gate_log`:
`windows_since_min` on VOC20-full hits 8669 (≫ 2000). **The sub100 "flatness" was partly an artifact
of the whole run fitting in the buffer, not a true fixed point.**

---

## 3. The fix: GradDivGate-PA — a permanent, never-evicted best anchor

Keep a dedicated `best_snapshot` updated **only** when a new grad-norm minimum appears, **never
evicted**. In the collapse regime (deep restore) restore toward this permanent best instead of the
rolling deque. Healthy regime still uses the lag-deque shallow restore (so the climb isn't
suppressed). Everything else identical to GDG.

```python
if g < self.g_min:                 # new best state
    self.best_snapshot = current_LN_snapshot     # pinned, never dropped
# deep restore: pull rst of LN params toward best_snapshot (not the deque)
```
Cost: one extra ~150 KB snapshot. File: `adapt/deyo_mlmp_hmgate2_continual.py`.

---

## 4. Headline result — PA fixes the full-dataset drift (2 full benchmarks)

| VOC20 full+15corr (R150) | last | drop | | Cityscapes full+15corr | last | drop |
|---|---|---|---|---|---|---|
| **GDG-PA** | **76.5** | **1.0** | | **GDG-PA** (R44) | **23.9** | **0.1** |
| GDG 0.95 | 63.5 | 13.9 | | GDG 0.95 | 15.2 | 8.6 |
| GDG 0.9 | 59.5 | 17.9 | | no-gate | 3.1 | 20.5 |
| no-gate | 34.0 | 43.3 | | episodic / no-adapt | 23.0 / 21.6 | — |
| episodic / no-adapt | 77.6 / 69.0 | — | | | | |

- **VOC20-full**: PA is a flat plateau at 76.5 (drop 1.0, post-peak slope ≈ −0.006/round), basically
  tying episodic 77.6, where the rolling-anchor GDG decays to ~60 (below no-adapt 69) and no-gate
  collapses to 34.
- **Cityscapes-full**: PA holds 23.9 (above episodic 23.0, no-adapt 21.6) through R44 while GDG 0.95
  slides to 15.2 (below both) and no-gate collapses to 3.1.
- **The permanent anchor is the thing that converts the slope into a plateau.** Two heavy full
  benchmarks now confirm it.

Figures: `figures/v20full_compare.png`, `figures/cityscapes_full_compare.png`,
`figures/pa_vs_gdg_grid.png`, `figures/pa_{v20full,acdc}_miou_vs_{hmargin,gradnorm}.png`.

### Where PA does NOT change anything (and its cost)
On the **sub100** runs the deque already covers the whole run, so PA ≡ GDG (identical curves on
Cityscapes/VOC20/VOC21/P59/P60/COCO-obj). The only sub100 difference is **ACDC: PA peak 31.8 vs GDG
33.4** — the stronger (always-true-best) anchor suppresses ACDC's climb by ~1.6. **PA's benefit is
specific to heavy/long runs where eviction bites; on light runs it's neutral-to-slightly-worse.**
COCO-Stuff sub100 still drops ~0.8 under PA (a base-ceiling issue, not eviction — see §6).

---

## 5. Analysis experiments — is GDG-PA really *adapting* (not inflating / memorizing)?

Two probes on GDG-PA, VOC20-C 5corr sub100, 150R (`--ood_corruptions`, `--resample_subset` flags
added to `main_continual.py`):

1. **OOD generalization probe** — each round, after adapting on the 5 train corruptions, also run a
   **frozen-weight** eval on 5 held-out corruptions (noise+blur). ID 72.2→**79.0**, OOD 68.4→**73.7**:
   the two curves rise **in parallel**. The model improves on corruptions it never adapts to ⇒
   **GDG-PA is improving the representation overall, not fitting the training corruptions.**
2. **Subset-resample** — re-draw a fresh random 100-image subset every (round, corruption). Still
   climbs and holds (mean 75.3, peak 79.9), staying far above No-Adapt 68.6 and around episodic 75.3
   ⇒ **not overfitting to the specific 100 images** — the adaptation is genuinely generalizable.

Figure: `figures/analysis_exps.png`.

---

## 6. Negative result — LCoTTA+MLMP (subspace constraint does not transfer)

Ported LCoTTA ("Lifelong TTA via Online Learning in Tracked Low-Dim Subspace", NeurIPS 2025) onto
MLMP: project the LN gradient onto the PCA principal subspace of recent gradients before stepping
(`adapt/lcotta_mlmp_continual.py`, torch-SVD, single backward). Result on sub100 5corr:

| | V20 | Cityscapes | ACDC |
|---|---|---|---|
| LCoTTA+MLMP | pk74.2 / last74.0 (low, no drop) | **collapse → 1.0** | **collapse → 1.5** |

Subspace projection **caps V20 below episodic and fails to prevent collapse** on Cityscapes/ACDC —
because the collapse direction *is* the dominant gradient mode, so projecting onto the principal
subspace preserves it. **Restore-based defense (GDG/PA) holds where subspace-based (LCoTTA) collapses**
— a clean contrast / negative baseline for the paper.

---

## 7. New direction started — put the constraint in the LOSS (proactive, not reactive)

Instead of a gate (detect degradation → restore), write the anti-collapse constraint into the loss.
Constraint: must be **forward-computable + differentiable** (single backward) — rules out grad_norm
(a gradient → 2nd backward, and unstable; the failed teammate attempt). Correlation analysis of 26
signals vs mIoU on ACDC/Cityscapes/VOC20 (`signals_log.csv`):

- **Only grad_norm correlates with mIoU on all 3** (−0.95/−0.78/−0.92) — but it's the un-usable one.
- Every **forward-differentiable** signal is strong on collapse-type (ACDC/Citys |corr| 0.9+) but
  **flips/dies on VOC20** (uniform degradation, signal-blind). h_margin: +0.97/+0.96/−0.20.

⇒ A loss-regularizer can only defend the **collapse** mode (acceptable: collapse is the catastrophic
one). Implemented **marginal-diversity term**: `L = L_DeYO+MLMP − λ · H(mean-pixel class softmax)`
(`adapt/deyo_mlmp_divloss_continual.py`, `--lambda_div`). This is the loss-form of GDG's H_margin
gate (SHOT/EATA InfoMax family).

**Result on ACDC (150R): NEGATIVE — the loss term does not prevent collapse.**

| λ | peak | last | drop |
|---|---|---|---|
| 0.1 | 33.5 | **9.9** | 23.6 |
| 0.5 | 33.2 | **8.6** | 24.6 |
| 1.0 | 32.8 | **5.0** | 27.8 |
| (no-gate ref) | 33.5 | 6.0 | 27.5 |
| (GDG-PA ref) | 31.8 | **31.8** | 0.1 |

All three λ collapse toward no-gate (last 5–10, **all below No-Adapt 23.3**); larger λ collapses
*worse* AND lowers the peak. Figure: `figures/divloss_acdc.png`.

**Why it fails (hypothesis):** (1) at batch_size=1 the per-step marginal is a *single image*'s class
histogram — maximizing its entropy optimizes the wrong quantity; the collapse is a *cross-stream*
marginal phenomenon the gate captured by aggregating over a 50-batch window, which a per-step loss
cannot see. (2) A forward regularizer only adds per-step pressure; it cannot *undo the accumulated
LN drift* the way restore actively pulls parameters back. So a proper loss version would need a
cross-batch EMA marginal (SHOT/EATA-style) — which re-introduces cross-batch state, no longer a clean
stateless loss.

**Takeaway (reinforces the paper's spine):** anti-collapse defense — restore-based **GDG-PA holds**,
subspace-based **LCoTTA collapses** (§6), loss-regularizer **divloss collapses**. Restore toward a
healthy anchor is the mechanism that works.

---

## 8. Other housekeeping
- **Cityscapes episodic baseline corrected**: the old `.save/CityscapesDataset/mlmp` used lr=0.01
  (10× too high) → degraded to 10.86 (below No-Adapt). Reran at lr=0.001 → **23.0** (`mlmp_lr1e-3`).
- **Full-benchmark batch queued** (`docs/running_experiments.md`): no-adapt / episodic(reuse) /
  no-gate / GDG-PA on V20+Cityscapes full+5corr and V21/P59/P60/COCO-obj/COCO-stuff full+15/5corr,
  full val, 50R. episodic & no-adapt deduped per-dataset.

---

## 9. Next
1. **Let Cityscapes-full PA finish** (R44/50) and run **PA on all full benchmarks** (queued) — fill
   the paper's full-dataset table. The competitor is **MLMP-episodic only** (OVSS-TTA is new).
2. **divloss sweep** — if the diversity loss term holds ACDC without a gate, it's a much simpler
   method; compare to GDG-PA. If λ must trade off the benign concentration, characterize it.
3. **COCO-Stuff residual drop** (~0.8, not eviction) — base-ceiling / many-class pseudo-label noise;
   decide whether to address or scope out.
4. **Settle the headline method**: GDG-PA (h_drop_ratio 0.9, permanent anchor) is the current best;
   write the formal spec and fold into EXPERIMENT_STATUS.md.
