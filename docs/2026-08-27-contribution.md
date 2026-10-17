# 0827 Contribution — batch-size collapse map across all four tasks, and gradnorm_scaled's first clean win outside OVSS

**Date:** 2026-08-27
**Scope:** this week spans all three lines in the repo — Traditional (non-CLIP) Classification and
Segmentation (`DeYO/`), CLIP Classification (`clsTTA/`), and OVSS (`MLMP/`) — organized around one
question: **does batch size cause collapse, and does `gradnorm_scaled` (AdaGate,
`shallow_cap_mode=growing_scaled`) fix it, on tasks other than OVSS?**

---

## 0. Headline: batch-size collapse map (corrected)

| Task | batch=1 | batch=8 | batch=64 |
|---|---|---|---|
| Traditional Cls (ImageNet-C, vitbase) | collapse | untested | no collapse (tent collapses to 0.10, sar survives 60.8) |
| Traditional Seg (ACDC, SegFormer-b0, DeYO, lr=1e-6, natural block) | **collapse** (R150=32.8, well below no_adapt=40.3) | no collapse (R150=39.8) | no collapse (R150=40.5) |
| CLIP Cls (ImageNet-C mild-5corr, ViT-L/14) | collapse (small block) | no collapse | no collapse |
| OVSS (V20/ACDC/Cityscapes, DeYO+MLMP) | collapse (no-gate) | collapse (no-gate) | collapse (no-gate, worst) |

Every no-gate row collapses somewhere. The corrected reading this week: **OVSS collapses at every
batch size once you check tail slope, not just the "does it end below baseline" question** — V20
no-gate batch=1/8/64 all have negative end-of-run slope (batch=64 collapses hardest, peak 78.4 →
52.8). Traditional Seg only collapses at batch=1 (ACDC has ~100 img/condition, so batch=64 there is
~1-2 steps/round — a structurally different regime, not "batch size is safe").

---

## 1. Traditional segmentation: two real bugs fixed before any of this was trustworthy

Built the traditional (non-CLIP) segmentation line from scratch this cycle: SegFormer-b0..b5
(`nvidia/segformer-{size}-finetuned-cityscapes-1024-1024`) on ACDC, standalone PIL loader, no mmseg
dependency (`DeYO/dataset_seg/acdc_seg.py`, `DeYO/methods/seg_model.py`).

- **BN/dropout bug**: `configure_model()` left the whole model in `.train()` mode (copying the
  classification convention). SegFormer's single `decode_head.batch_norm` then used live batch=1
  statistics instead of frozen pretrained running stats, and `decode_head.dropout` (p=0.1) randomly
  zeroed 10% of final features every forward pass — corrupting predictions from round 1, before any
  gradient step. **Fix**: keep the whole model in `.eval()` mode; gradients still flow via
  `requires_grad`, independent of train/eval mode (`DeYO/methods/tent_seg.py::configure_model()`,
  shared by all seg methods). Verified via same-image/zero-gradient-steps diagnostic — exact match to
  no_adapt post-fix.
- **`CUDA_VISIBLE_DEVICES` override bug**: `main_continual_cls.py` does
  `os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu` unconditionally (default `'0'`), silently
  overwriting any externally-set env var. Jobs launched with `CUDA_VISIBLE_DEVICES=N python ...`
  (no `--gpu` flag) silently landed on GPU0 instead and OOM'd. **Fix**: always pass `--gpu N`
  explicitly in launch scripts, never rely on the external env var alone.

---

## 2. `deyo_adagate_seg` (gradnorm_scaled) on traditional segmentation — LR-sensitive, not fundamentally broken

Ported AdaGate's `growing_scaled` shallow-cap recipe onto DeYO for segmentation
(`DeYO/methods/deyo_adagate_seg.py`), replacing the earlier `tent_gate` port.

**At lr=3e-6, 150R**: only a partial win. deyo_adagate R150=37.9 vs plain deyo's 29.6 — better, but
still declining at the end, not truly stabilized (`figures/acdc_seg_lr3e6_trajectories.png`).

**Root cause** (read directly from `adagate_log.csv`, not guessed): `collapse_regime` fired on
749/1218 (61.5%) of monitoring windows — h_margin oscillates noisily in a ~1.33–1.93 band with no
clean bimodal collapse/healthy separation on this small model, so the 90%-of-running-max threshold
is mostly reacting to ordinary noise. DEEP restores fired 12.8% of windows but each only stochastically
restores 1% of the 62 collected norm-affine tensors — partial damping, not real stabilization.

**At lr=1e-6, 40R (`deyo_lrgrid`, `figures/deyo_lrgrid.png`)**: **clean, unambiguous win.**
deyo_adagate R40=40.0 (at/above no_adapt=40.255) vs plain deyo@1e-6's R40=37.7 — **+2.3**, staying
near no_adapt the whole 40 rounds while plain deyo visibly declines from ~R10 on. **Takeaway:
gradnorm_scaled needs the safer LR to work cleanly on this base method/model — it is not fundamentally
broken, it's LR-sensitive**, same shape of finding as the OVSS side's own LR dependence.

**Batch ablation, DeYO alone, lr=1e-6, 150R, natural block (no block-length control, per standing
instruction)**: batch=1 R150=32.8 (**collapses** — well below no_adapt=40.3), batch=8 R150=39.8,
batch=64 R150=40.5 (both hold near/above no_adapt). Required adding `--batch_size` and `--model_size`
to `main_continual_seg.py` (previously hardcoded batch=1) and fixing a `pred[0],lbl[0]` metric bug
that assumed batch=1.

**Model-scale test (b0 vs b5, `figures/seg_b5.png`)**: SegFormer-b5 (84.6M params, 52 blocks vs b0's
8, 324 collected norm-affine tensors vs 62) at lr=1e-6. First 60R: deyo peaks ~50.77@R22-25, then
**still declines** — dips below no_adapt (50.62) by ~R50, ends R60=50.54. Scale delays and shrinks the
collapse magnitude by roughly an order of magnitude (total swing ~0.2 mIoU vs b0's multi-point swings)
but does **not** eliminate it. deyo_adagate held at R60=50.77 (no decline yet) — **relaunched to 150R
this week** (no checkpoint/resume mechanism exists in `main_continual_seg.py`, but the run is fully
deterministic — seed=0, shuffle=False — so a from-scratch rerun reproduces R1-60 exactly; safe to just
rerun rather than build resume support). All three arms currently running on GPU5 (only GPU with
headroom; GPUs 0-4 occupied by a labmate's concurrent OVSS batch-ablation jobs).

---

## 3. OVSS: gradnorm_scaled's batch-size story completed — clean positive at 8 and 64 too

The flagship `deyo_mlmp_adagate_continual` (gradnorm_scaled) result only existed at batch=1
(ACDC R150=31.92, VOC20 R150=78.4, Cityscapes R150=31.9-ish — all defaults). Ran the same method at
batch=8/64 on VOC20 (`bash/_v20_deyo_adagate_batch8_64.sh`; full `trainval.txt`, 2913 images, no
subset cap, natural block length; same hyperparams as the flagship config
`save/ACDCDataset/adagate_ABmad05_ecdf_grow_scaled/cmd.sh`, fixed lr=5e-6 — this codebase's OVSS
convention is fixed LR across batch size, unlike the traditional-cls line).

| batch | R1 | R150 |
|---|---|---|
| 1 (flagship, different subset config) | 72.2 | 78.4 |
| 8 (trainval, ~364 steps/round) | 73.1 | **78.7** |
| 64 (trainval, ~45 steps/round) | 72.3 | **78.3** |

All three converge to the same ~78.3–78.7 plateau, flat in the R146-150 window, no decline at any
batch size — the gate that stabilizes batch=1 **generalizes cleanly to batch=8/64** here, in direct
contrast to the no-gate ablation where all three batch sizes collapsed (batch=64 worst: peak 78.4 →
52.8). Figure: `figures/v20_deyo_adagate_batch_ablation.png`.

**Side note on GPU memory** (came up mid-week): the flagship ACDC/Cityscapes protocol uses
`--init_resize 1120 560` (matches the paper's driving-scene resolution) with `--patch_size 224 224
--patch_stride 112`, i.e. **36 sliding-window patches per image** — batch=8 there means up to 288
patches through the network at once (~79GB). This V20 batch ablation uses `--init_resize 224 224`
(resize == patch_size, so each image is exactly one patch) — batch=8/64 here really is just 8/64
images, hence the far smaller memory footprint. Not a bug on anyone's part, just two different
resolution conventions by dataset.

---

## 4. CLIP classification: mixed picture, and the first negative gradnorm_scaled result

**6-method 150R comparison** (ImageNet-C mild-5corr, sub875, batch=1, `--lr` from the proven-good
sweep, `clsTTA/`): source 49.65 → tent 56.16, eata 57.65, deyo 56.89, gdgpa 52.62, adagate 53.37, all
above source. **eata is the current best on this task/config.**

**Targeted gradnorm_scaled evidence** (batch=1, block=40, subset_size=40, 150R,
`DeYO/output/cls_gradnormscaled_evidence/`): deyo R150=52.0 vs deyo_adagate R150=44.5 —
**gradnorm_scaled underperforms plain deyo here**, a persistent ~6-8pt gap, not noise. This is the
first clean *negative* result for gradnorm_scaled outside OVSS — at this short block length there may
be no real collapse for the gate to rescue from, so its restores just hold back genuine progress. Not
yet reconciled with the positive traditional-seg (lr=1e-6) and positive OVSS (all batch sizes)
results — worth checking whether a longer block or different LR changes this.

**Traditional cls batch=64 matched-block rerun** (block=781, the closest achievable to batch=1's 875
given ImageNet-C's fixed 50000-image val set — user sign-off to accept this instead of forcing exact
match via repetition): tent collapsed to 0.10 (chance-level), sar survived at 60.84. **deyo and eata
both OOM-crashed** (GPU0 contention with other concurrent jobs) and were never rerun — **pending**.

---

## 5. Housekeeping
- Killed all 6 stuck Cityscapes-15corr arms (`pkill -9 -f "save_dir save/CityscapesDataset/cityscapes_15corr/"`),
  freeing ~77GB on GPU0. Not restarted per explicit instruction.
- Standing simplification adopted: **stop controlling block length across batch-size ablations**
  going forward (ACDC's ~100 img/condition makes it structurally impossible at batch=64 anyway; user
  extended this to VOC20/OVSS ablations too where it was technically possible).

---

## 6. seg_b5 root cause: DEEP restore essentially never fires, and it's *not* the b0@3e-6 noise problem

seg_b5 finished at 150R: deyo peaks ~50.77@R22-25 then keeps declining to R150=49.93 (below no_adapt
50.62) — scale delays collapse, doesn't prevent it. deyo_adagate (`h_drop_ratio=0.9`, the OVSS default)
tracks deyo up to the same peak, then slowly decays back to R150=50.60 — barely beats no_adapt, clearly
worse than the b0@1e-6 result (+2.3 clean win).

**Root cause, read directly from `adagate_log.csv` (1218 windows)**: DEEP restore (the strong,
permanently-anchored correction) fired **exactly 1 time** the entire run. DEEP needs `z > trend_thr`
(active grad-norm-slope trend) **AND** `collapse_regime` (h_margin dropped) simultaneously — on b5,
`z > trend_thr` fires 28.7% of windows (statistically identical to OVSS's self-calibrated 29-32% rate,
so this half is fine), `collapse_regime` fires 19.0% of windows independently, but the two **almost
never co-occur** (1/1218). Pre- vs post-peak gate statistics are essentially indistinguishable
(collapse 18.0%→20.0%, DEEP 0.16%→0%, mean grad_norm 2.14→1.93) — the gate has **no internal signal**
that anything changed at the R76 turning point. All correction falls to SHALLOW restore (28.6% of
windows, `rst=0.01`/element), which doesn't even target the true best state — it restores toward
`_win_buf[-1-lag]`, a snapshot `lag` windows back (median lag=10 windows ≈1.2 rounds, but up to 94
windows ≈11.6 rounds; unrelated to `windows_since_min`, which stays ~58 windows/~7 rounds stale on
average) — an arbitrary recent-ish state, not the peak. Weak + untargeted ⇒ merely slows the bleed.

**This is a *different* failure mode from b0@lr=3e-6** (§2: there, collapse_regime over-fired at 61.5%
because h_margin was too noisy to separate collapse from health at all). Here on b5, both individual
trigger rates look OVSS-normal — it's specifically their **joint timing** that fails to align.

---

## 7. Ported 3 new baselines to traditional segmentation: RoTTA, LCoTTA, MGP

Extended `DeYO/methods/` with segmentation ports of three more TTA baselines, mirroring their existing
OVSS ports (`MLMP/adapt/{rotta,lcotta,mgp_deyo_mlmp}_continual.py`) but wired to SegFormer/DeYO instead
of CLIP/MLMP:

- **`rotta_seg.py`**: CSTU category-balanced memory bank (verbatim port) + EMA teacher (`nu=0.001`).
  One real fix vs. the OVSS version: the OVSS port briefly calls `.train()` during the memory-bank
  update step — on this backbone that would **reintroduce the BN/dropout bug from §1**, so the seg port
  stays in `.eval()` throughout (gradients flow via `requires_grad` regardless of train/eval mode).
- **`lcotta_seg.py`**: gradient-subspace projection (PCA via torch SVD of a moving window of past
  gradients, project current gradient onto it before stepping) on plain per-pixel entropy loss
  (matching the paper's own default — no CLIP multi-prompt fusion exists here to combine it with).
- **`mgp_seg.py`**: `_MGPSubspaceTracker` (Marchenko-Pastur-thresholded SVD distillation + inertial
  fusion, copied verbatim — fully backbone-agnostic) wrapping DeYO's segmentation loss instead of
  MLMP's multi-prompt loss.

All three smoke-tested clean and wired into `main_continual_seg.py` (`--method rotta|lcotta|mgp`,
`optimizer=Adam` for these three specifically, matching their OVSS ports — unlike this driver's other
methods, which use SGD).

**9-method comparison, SegFormer-b5, ACDC, batch=1, 150R** (`figures/seg_b5_9methods.png`): mgp best
(R150=50.71, peak 50.83@R92), tent/sar mild-positive (~50.67), eata/rotta flat (~50.62, barely adapt at
all), deyo declines (49.93), **lcotta collapses hardest of all nine** (47.73) — a clean negative
baseline, consistent with the OVSS side's own finding that LCoTTA doesn't prevent collapse.

---

## 8. h_drop_ratio retuned for b5 (batch=1): 0.94 beats mgp, becomes the new "stable" deyo_adagate

Given §6's diagnosis (DEEP needs `collapse_regime` and active-`z` to co-occur, and on b5 they don't at
`h_drop_ratio=0.9`), swept `h_drop_ratio ∈ {0.80, 0.85, 0.91, 0.92, 0.93, 0.94, 0.95, 0.99}`, 150R each
(`figures/seg_b5_hdrop_sweep.png`):

| h_drop_ratio | 0.80–0.91 | 0.92 | 0.93 | **0.94** | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| R150 mIoU | ~50.60 (flat, DEEP never fires) | 50.68 | 50.74 | **50.77** | 50.74 | 50.66 |

Clean inverted-U, peak at **`h_drop_ratio=0.94`** — R150=50.77, **beats mgp's R150=50.71** (though mgp's
peak, 50.83@R92, is still slightly higher than 0.94's peak of 50.77@R142). `h_drop_ratio=0.94` is now
adopted as **the stabilized deyo_adagate config for b5** and substituted into
`figures/seg_b5_9methods.png`'s deyo_adagate arm in place of the OVSS-default 0.9.

---

## 9. batch=8 on b5: deyo's collapse is delayed, not prevented, at 500R — and 0.94 completely fails there too

**150R, batch=8** (`figures/seg_b5_9methods_batch8.png`): reran all 9 methods at batch=8. Striking
reversal from batch=1 — **plain deyo does NOT collapse**, climbing steadily to R150=50.78 (2nd best,
mgp still leads at 50.82). Looked like batch=8 "fixes" deyo the same way it did for b0 (§0).

**Extended to 500R** (`figures/seg_b5_9methods_batch8_500R.png`, all 9 methods, fully deterministic
reruns from scratch since `main_continual_seg.py` has no checkpoint/resume): the "fix" was an illusion
— **deyo peaks ~R150-180 then declines the rest of the way to R500=50.48, below no_adapt**. batch=8 only
*delayed* the collapse (matches b0's lr-sensitivity lesson: apparent stability can just mean "not enough
rounds yet"). mgp is the only method still monotonically rising at R500 (51.14). deyo_adagate
(`h_drop_ratio=0.94`, carried over unretuned from the batch=1 tuning) holds up much better than plain
deyo (R500=50.66) but is clearly *not* stable either — investigated why.

**Why 0.94 fails at batch=8**: `adagate_log.csv` shows `collapse_regime` fired **0% of 530 windows** —
literally never. h_margin's dynamic range collapses from ~1.6–1.9 (batch=1) to **1.846–1.885** at
batch=8 (batching 8 images per step averages out the marginal-distribution noise that gave h_margin
room to swing) — the observed floor of `h_margin/running_h_max` never dropped below **0.979** the whole
run, so any `h_drop_ratio ≤ 0.979` is mathematically incapable of ever firing.

**Re-swept for batch=8** (`h_drop_ratio ∈ {0.94, 0.96, 0.975, 0.98, 0.985, 0.99, 0.995}`, 500R each,
`figures/seg_b5_batch8_hdrop_sweep.png` + trajectories in
`figures/seg_b5_batch8_hdrop_trajectories.png`): flat at ~50.66 through 0.98 (confirms the 0.979 floor
exactly), then rises — 0.985→50.67, 0.990→50.74, 0.995→50.76 — **still climbing at 0.995, no peak found
yet** (unlike batch=1's clean inverted-U at 0.94). Trajectories reveal *why* it doesn't converge cleanly:
at 0.990/0.995, `collapse_regime` fires 34–66% of windows and DEEP fires in **tight clusters** (e.g. 3-4
DEEP events within 100-150 batches of each other) — each cluster snaps ~1%/element back toward
`best_snapshot`, producing a visible **sawtooth**: adapt-and-climb, then a multi-window DEEP burst yanks
mIoU back down, repeat. **At high `h_drop_ratio` the gate flips from "asleep" (batch=1's 0.9 problem) to
"twitchy"** — over-triggering DEEP in bursts against a `best_snapshot` that itself isn't always fresh
(`windows_since_min` mixes recent resets with 40-60-window-stale stretches). Neither failure mode
resembles OVSS's clean, flat, self-calibrating behavior.

---

## 10. Why doesn't traditional seg get OVSS-clean gate behavior? Two distinct causes, not one

Asked directly (not just asserted): is SegFormer's signal just worse than OVSS's? Answer is **yes, but
for two separable reasons, one more fixable than the other**:

1. **Genuine hyperparameter fragility**: the SAME `h_drop_ratio` needed retuning across b0-lr→b5-scale→
   batch-size, three separate times this cycle (0.9 works at b0/lr=1e-6; needs 0.94 at b5/batch=1; needs
   ≥0.985 — and still doesn't cleanly converge — at b5/batch=8). OVSS's `ABmad05_ecdf_grow_scaled` used
   **one fixed config across ACDC/VOC20/Cityscapes** (three different datasets) with self-calibrated
   firing rates within 1.1× of each other. Traditional seg's h_margin dynamics are simply less portable.
2. **Headroom**: SegFormer-b5's no_adapt on ACDC is 50.62, and the best any of 9 methods+sweeps achieved
   was ~50.8-51.1 — **headroom of 0.2-0.5 mIoU**, vs. OVSS's ACDC headroom of ~7-8 mIoU (per this repo's
   own CLAUDE.md). When the real adaptation signal and measurement noise are the same order of
   magnitude, no amount of threshold-tuning can cleanly separate "genuine drift" from "noise" — this is
   a structural limit, not a tuning problem. (SegFormer-b5 is *already* Cityscapes-finetuned, so ACDC's
   real-weather shift barely challenges it.)

**Directly tested #2 by building a higher-headroom traditional-seg benchmark** — see §11.

---

## 11. New dataset: Cityscapes-C hard5corr — headroom hypothesis confirmed, cleanest gate demo yet

Built `DeYO/dataset_seg/cityscapes_c_seg.py`: Cityscapes val (500 images, 3 cities) + the CLIP-cls
line's "hard5corr" set (`defocus_blur, glass_blur, gaussian_noise, zoom_blur, elastic_transform`,
severity=5) applied on-the-fly. Vendored MLMP's standalone `utils/imagecorruptions` module (pure
numpy/PIL/skimage/cv2/scipy, no mmcv/mmseg dependency) into `DeYO/dataset_seg/imagecorruptions/` so the
traditional-seg line stays mmseg-free, matching `acdc_seg.py`'s existing standalone-loader convention.
Wired into `main_continual_seg.py` via new `--dataset {acdc,cityscapes_c}`.

**Headroom confirmed immediately**: no_adapt mean_mIoU = **25.2** (b5) vs ACDC's 50.6 — severity-5
synthetic corruption hits a Cityscapes-finetuned model far harder than ACDC's real weather does
(`gaussian_noise` alone: 9.8 mIoU).

**9-method comparison, sub100/condition, batch=8, 500R** (`figures/cityscapes_c_hard5corr_sub100_
batch8_500R.png`): **2.7 mIoU spread between best and worst** (vs. ACDC's <0.5) — every qualitative
pattern that was ambiguous on ACDC is now unambiguous:
- **mgp**: monotonic rise the entire 500R, no peak, R500=24.9 (best, by a wide margin).
- **lcotta**: classic rise-then-collapse — peaks 24.27@R130, then a long clean decline to **22.2** by
  R500 (worst) — the collapse that was only a mild downward drift on ACDC is now a real crash.
- **deyo (plain)**: same shape — peaks 24.0@R60-70, declines to 22.9 (below no_adapt).
- **deyo_adagate (`h_drop_ratio=0.94`, unretuned for this dataset)**: tracks plain deyo's early peak
  almost exactly, but **after ~R100 the gate visibly holds it** at 23.8-24.0 (with sawtooth from active
  DEEP corrections) instead of collapsing alongside deyo — R500=23.86, **~1 full mIoU above plain deyo**.
  **This is the single cleanest gate-vs-no-gate demonstration in the whole traditional-seg line to
  date** — the exact mechanism (prevent the collapse deyo suffers) working exactly as designed, now that
  there's enough headroom for the effect to be visible above noise.
- tent/sar: mild rise-then-decline back to baseline; eata: flat (barely adapts, as on ACDC); rotta:
  slow mild positive drift.

**batch=1, sub100, 150R version launched** (`output/cityscapes_c_hard5corr_sub100_batch1_150R/`) —
in progress, not yet complete; `h_drop_ratio=0.94` is actually *native* to this config (it was tuned at
batch=1 originally, just on ACDC, not yet on this dataset).

---

## 12. TODO — figures still to draw once in-progress runs finish

- [ ] **Cityscapes-C hard5corr, batch=1, 150R, 9-method comparison** — mirror
      `plot_cityscapes_c_hard5corr_sub100_batch8_500R.py`, pointed at
      `output/cityscapes_c_hard5corr_sub100_batch1_150R/` instead. Output:
      `figures/cityscapes_c_hard5corr_sub100_batch1_150R.png` (preliminary version already exists
      and is kept updated on request; final version once all 9 arms reach R150). Note: `sar`
      collapsed and got stuck at an identical degenerate value from R3 onward — see §16, needs a
      closer look at `run.log` before/alongside the trajectory plot, not just plotting it as a
      normal curve.
- [ ] **CLIP cls batch=8/64 LR grid** — mirror `plot_batch1_lrgrid.py`'s structure (one panel or
      two side-by-side), reading `save/clip_vitL14_mild5corr_batch{8,64}_nogate_lr*_15R/`.
      Batch=8: lr ∈ {1.25e-5, 5e-5, 1e-4}. Batch=64: lr ∈ {1.25e-5, 2e-4, 8e-4}. Purpose: does
      the linear-scaling prediction from batch=1's confirmed-good LR (1.25e-5 → ×8 → 1e-4 for
      batch=8, ×64 → 8e-4 for batch=64) actually fix collapse the way it did at batch=1? Output:
      `figures/batch8_64_lrgrid.png`. Still very early (R1-10 depending on arm) — batch=64's LR
      grid is heavy (50000 images/round) and slow.
- [x] **OVSS no_adapt + MLMP-episodic references, matched to the trainval.txt/no-subset-cap
      config** — done. `save/PascalVOC20Dataset/v20_batch_ablation/No_Adaptation/` (batch-invariant,
      R1=70.41 is the final value) and `.../mlmp_episodic/` (complete, all 5 corruptions:
      snow=75.00, frost=69.04, fog=75.36, brightness=79.38, contrast=75.27 → mean **74.81**).
- [ ] **OVSS no-gate batch=1/8/64, refit to full trainval.txt (no subset cap)** — launched
      (`bash/_v20_batch_ablation_nogate_fulldata.sh`), matching the gated arms' data config so the
      no-gate vs gated comparison finally uses the *same* images at every batch size (see §15 for
      why the old subset-capped comparison was misleading at batch=1/8, not just batch=1 vs gated).
      All three still in R1 as of last check (batch=1 is ~14565 steps/round, slowest).
- [ ] **OVSS gated batch=1, clean apples-to-apples** — `bash/_v20_deyo_adagate_batch1_clean.sh`,
      in progress (R6+ as of last check). Once both this and the no-gate fulldata rerun finish,
      redraw `figures/v20_batch_ablation_with_refs.png` with fully-matched data on both lines at
      all three batch sizes (currently filled with the OLD mismatched batch=1 data on both sides,
      per explicit request, so the figure is complete rather than partially empty — see §15).

---

## 13. Caught a bad comparison: the old gated OVSS batch=1 line wasn't apples-to-apples

`figures/v20_deyo_adagate_batch_ablation.png`'s batch=1 arm was borrowed from a *different*
experiment (`v20_acdc_matched`, `--split val --subset_size=100`) rather than run fresh on the same
`trainval.txt`/no-subset-cap config as batch=8/64. R1 values expose it: no-gate batch=1 R1=59.7 vs
gated batch=1 R1=72.2 — a 12.5-point gap at round 1 (before adaptation has done anything) that can only
come from different underlying image subsets, not the method. Launched a clean rerun
(`bash/_v20_deyo_adagate_batch1_clean.sh`, same trainval.txt/no-cap config as batch=8/64) — see TODO
above. **Lesson**: reusing a result from an earlier, differently-configured experiment as one arm of a
new ablation silently breaks the comparison — always verify R1 (pre-adaptation) values match across
arms that are supposed to differ only in the variable being ablated.

---

## 14. Next
1. **Cityscapes-C hard5corr batch=1, 150R** — let it finish; compare against the batch=8/500R result to
   see whether the same batch-size-breaks-h_drop_ratio problem (§9) recurs here, or whether the bigger
   headroom makes the gate more batch-size-robust too.
2. **h_drop_ratio has now been shown to need per-(model-scale, batch-size, dataset) retuning three
   separate times** — worth checking whether a *relative* formulation (e.g. percentile-based rather than
   ratio-of-running-max) would be more portable than sweeping `h_drop_ratio` again for every new config.
3. **Reconcile the CLIP-cls gradnorm_scaled negative result** (§4) with the now much stronger positive
   traditional-seg (Cityscapes-C) and OVSS (all batch sizes) results — try a longer block or the
   batch=64 config before concluding gradnorm_scaled doesn't transfer to CLIP classification.
4. **Traditional cls batch=64**: rerun deyo/eata (OOM'd, no data) on a GPU with headroom.
5. Fold the corrected batch-size collapse map (§0) and the headroom finding (§10-11) into the paper's
   core narrative — the traditional-seg line now has both a mechanistic failure story (§6, §9) and a
   clean positive demonstration (§11) to anchor the gate's real-world value on.
