# Gate Mechanisms Catalog — DeYO+MLMP CTTA (for the report)

**Last updated:** 2026-06-22
**Purpose:** plain-language reference for *every* gate/defense mechanism tried on the
DeYO+MLMP base, what signal each watches, when it acts, what it restores toward, and
its result so far. Built so the mechanisms can be explained quickly when writing the
report. Companion data: each run's `save/.../results_all_rounds.txt` + `gate_log.csv`.

---

## 0. The shared base — DeYO + MLMP (what is being adapted)

- **Backbone:** NA-CLIP ViT-L/14, frozen text encoder. Only the **LayerNorm (γ,β)** of
  the visual encoder are updated (`top_block_exclude=6` → upper blocks only). LR 5e-6.
- **MLMP machinery:** 7 text prompts + 18-layer UAML feature fusion (eval uses
  adaptive-weighted-mean; adapt uses mean). This is the "multi-level multi-prompt" eval.
- **DeYO adapt loss:** entropy minimisation on pixels that survive a **dual reliability
  filter** — keep a (prompt,pixel) only if (a) entropy < margin AND (b) PLPD
  (patch-shuffle prediction disagreement) > threshold — then a reweighted-entropy loss.
- **Protocol:** continual, no per-sample reset, evaluate-before-adapt, 150 rounds, seed 0.

**Without any defense, this base COLLAPSES** (ACDC 33.5→6.0, Cityscapes 24.1→2.8): the
classic "improves then collapses" TTA failure. Every mechanism below is a *defense* that
tries to keep it both **high** (reach the peak) and **stable** (hold it to R150).

### The two degradation regimes (why one defense rarely fits all)
- **Collapse type (ACDC, Cityscapes):** the class marginal collapses onto 1–2 classes;
  **H_margin (class-diversity entropy) DROPS**. Easy to detect.
- **Uniform-degradation type (VOC20):** mIoU drifts down mildly (78.7→74.3) but the
  marginal stays diverse, so **H_margin stays high (~3.0) and never flags it.**

---

## 1. The three signals (what we can watch per batch)

| Signal | What it is | Behaviour vs mIoU | Good for |
|---|---|---|---|
| **H_margin** | entropy of the (windowed) mean class distribution | DROPS on collapse; **flat/high on VOC20 (blind)** | collapse TRIGGER (ACDC/Cityscapes) |
| **mean_conf** | mean max-softmax over pixels | rises monotonically toward the peak on **all** datasets (incl VOC20) | universal peak-crossing TRIGGER |
| **grad_norm** | L2 norm of the LN-param gradient of the adapt loss | U-shaped: high at R1, **bottoms at the mIoU peak, RISES as mIoU degrades** on all datasets (incl VOC20) | degradation magnitude / restore DEPTH |

Two distinct "facets" of a signal (學長, `docs/2026-06-18-contribution.md` §7.2):
- **TREND** = does it move with mIoU magnitude (good for a continuous brake). grad_norm wins (works on VOC20).
- **THRESHOLD** = is it monotone-in-time so one crossing = a clean one-shot trigger. H_margin / mean_conf win.

> Key calibration fact: grad_norm is **noisy and high even when healthy** (per-step spikes
> to ~88 at the healthy baseline). So it is usable as a *trend/depth* signal but NOT as a
> raw magnitude to threshold or to penalise directly (see §6).

---

## 2. `deyo_mlmp_divgate_continual` — H_margin gate → restore to SOURCE  *(current best baseline)*
- **Watch:** windowed H_margin. **Trigger:** 3-tier — H≥h_threshold → aggressive (no restore);
  h_warning≤H<h_threshold → cautious (rst=cautious_rst); H<h_warning → brake (rst=brake_rst).
- **Restore target:** the frozen **source** snapshot, stochastically (a fraction `rst` of LN
  params reset to source each step).
- **Per-dataset thresholds:** ACDC h2.0/1.7, Cityscapes h2.1/1.8, VOC20 h3.0/2.7; rst 0.005/0.02.
- **Result (學長, 150R):** ACDC mean **31.8**, Cityscapes **23.6**, VOC20 **77.4**. Beats episodic
  on ACDC/Cityscapes. **This is the bar to beat.**
- **Weakness:** blind to VOC20's uniform drift (H stays >threshold → gate never fires → = no-gate).

## 3. `deyo_mlmp_gradslope_continual` — grad_norm SLOPE → restore to LAGGED snapshot
- **Watch:** least-squares **slope** of the windowed grad_norm over `slope_window` windows.
- **Trigger:** slope ≤ slope_deadzone → no restore; slope > deadzone → restore.
- **Restore target:** a **lagged LN snapshot** `lag` batches ago, `lag = lag_gain*slope`
  (capped at max_lag); prob `base_rst`. Steeper rise → restore further back.
- **Result (ours, 150R, 2026-06-22):** stable, **beats episodic on all 3**, but **does NOT beat
  divgate**: V20 73.6 (vs 77.4), ACDC ~31.4 (vs 31.8), Cityscapes ~23.2 (vs 23.6).
  `slope_deadzone` was **insensitive** (0.001–0.008 all the same); `base_rst=0.01 > 0.02`.
- **Why it loses on V20:** there is no collapse to prevent; restoring on the (real) grad_norm
  rise just **caps the peak** the base model would otherwise hold.

## 4. `deyo_mlmp_composite_gate_continual` — mean_conf TRIGGER + grad_norm DEPTH → restore to SOURCE  *(NEW BET, running)*
- **Watch:** windowed mean_conf (timing) + windowed grad_norm (intensity).
- **Trigger:** restore turns ON once windowed mean_conf rises **past `conf_ceil`** (i.e. past the
  mIoU peak). Because mean_conf is monotone on **all** datasets, this fires on VOC20 too —
  the case divgate misses.
- **Restore target:** **source** (DivGate-style flat stochastic restore), with
  `rst = base_rst` scaled UP to `base_rst*grad_mult_max` as grad_norm rises above its trigger level.
- **Hypothesis:** combines the universal trigger (mean_conf) with the magnitude signal
  (grad_norm) → should hold the VOC20 peak that divgate cannot, while still catching ACDC/Cityscapes.
- **Sweep (running 2026-06-22):** `conf_ceil` ≈ each dataset's peak mean_conf
  (ACDC {0.78,0.83,0.86}, V20 {0.68,0.71,0.74}, Cityscapes {0.66,0.69,0.72}) × base_rst {0.005,0.01}.
  **Results: TODO.**

## 5. `deyo_mlmp_smooth_anchor_continual` — continuous lag(H) → restore to snapshot/source  *(available, from 學長/Phase M)*
- **Watch:** windowed H_margin. **Restore:** toward a snapshot `lag(H)=lag_scale/(H−h_floor)`
  batches back — H≥h_ceil → no restore; H≤h_floor → all the way to source; in between → a
  recent-ish snapshot. "Raise h_floor → retreat to source sooner" (Phase M D-principle).
- **Status:** recalibrated windows exist (ACDC h_ceil2.0/floor1.6); not in this batch yet.
  Phase M finding: smooth-anchor ties a *tuned* source-reset (no clear win). Lower priority.

## 6. `deyo_mlmp_gradpen_divgate_continual` — grad_norm as a LOSS PENALTY  ❌ FAILED (negative result)
- **Idea:** add `λ·‖∇L‖²` (or excess over a baseline) to the loss to directly optimise grad_norm down.
- **Outcome (smoke-tested 2026-06-22):** **no usable operating point** — small λ does nothing
  (flat = baseline), larger λ **collapses** the model; `excess`/`ema` rise-penalties collapse too;
  grad-clipping does NOT save it.
- **Why:** (a) grad_norm is high even when healthy (so penalising it fights healthy learning),
  and (b) the penalty's gradient direction (Hessian·∇L of the entropy loss) is destabilising.
- **Lesson:** grad_norm is a good degradation **detector/restoration-trigger** (→ §3,§4) but a
  **bad loss target**. This negative result is what motivated the pivot to restoration-based use.
- Code kept (λ=0 is bit-identical to divgate); spec `docs/superpowers/specs/2026-06-21-gradnorm-penalty-design.md` §8.

---

## 7. Reference numbers (150R mean mIoU)
| Dataset | No-Adapt | MLMP-episodic | DeYO+MLMP no-gate | **divgate (bar)** |
|---|---|---|---|---|
| ACDC (4 cond) | 23.34 | 29.84 (step1) / 30.6 (step10) | 33.5→6.0 collapse | **31.8** |
| Cityscapes (5corr sub100) | 20.6 | 20.0 | 24.1→2.8 collapse | **23.6** |
| VOC20 (5corr sub100) | 68.6 | 76.21 | peak78.7→last74.3, mean 77.4 | **77.4** (=no-gate) |

## 8. Current experiment status (2026-06-22)
- **gradslope sweep** (18 runs): V20 done; ACDC ~95%; Cityscapes ~73%. Results in §3.
- **composite sweep** (wave-1, 15 runs): launched, all 3 datasets. Results: TODO.
- **Deferred (wave-2):** remaining Cityscapes composite (rst0.01 ×3); same-machine divgate
  baselines (currently comparing to 學長's numbers, not same-machine).
- Launchers: `bash/sweep_gradslope.sh`, `bash/sweep_composite.sh` (both: per-dataset, concurrent, env-overridable).
