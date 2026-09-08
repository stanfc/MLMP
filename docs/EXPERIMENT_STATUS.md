# Experiment Status — Research Arc

**Last updated**: 2026-06-03
**Newest result (2026-06-03, §15)**: **H1 validation experiments RUN & analyzed.** H1 ("CLIP saw synthetic → no adaptation direction") is **refuted**: Cityscapes *synthetic* corruptions have the same gradient geometry (cos ≈ 0.39) and headroom as *native* shifts (cos ≈ 0.49). The real axis is **VOC vs everything else**, and the 3 synthetic failures split into two mechanisms — VOC = genuine no-headroom; Cityscapes = temporal-stream collapse *despite* headroom + correct gradient. New testable hypothesis **H2** (stream length/heterogeneity, not synthetic-ness). See §15.
**Prior state**: SAR phase COMPLETE on ACDC + VOC20 (after **direction-bug fix**, see §7). SAR alone hits ACDC mean 30.32 (matches MLMP-episodic 30.60 but with periodic W-shape dips); on VOC20 weather it hits ~67 (below No-Adapt 70.79, headroom problem unchanged). **SAR-DivGate hybrid implemented (Phase J)** — combines SAR's SAM + reliable filter with DivGate's 3-tier stochastic restore (replaces SAR's hard recovery). Implemented, not yet run. **New `bash/v20_acdc_matched/` folder** built for direct ACDC↔VOC20 trajectory comparison (4 corruptions × 101 imgs = 404/round ≈ ACDC 406). 8 method scripts ready; no runs yet.
**Active runs**: none currently. Next priority: `bash/ACDC_10_round/sar_divgate_continual.sh` (Phase J test), then `bash/v20_acdc_matched/*` for cross-dataset trajectories.
**Open data**: CMA-Layered loose-rate variants in `save/ACDCDataset/cma_layered_continual_rate_*` paused at R47-48 (lower priority).

This document is the **entry point** for anyone picking up the work — read this first, then drill into the referenced specs for detail. The full timeline below covers every experiment from the original CMA hypothesis through the current TENT-DivGate breakthrough.

---

## 1. The Big Picture

**Project**: MLMP — Test-Time Adaptation for Open-Vocabulary Semantic Segmentation on NA-CLIP (ViT-L/14), evaluated on ACDC (fog/night/rain/snow × 10 conditions, 150 continual rounds, evaluate-before-adapt protocol).

**Phase 1 (complete, in `proposal.md`)**: Benchmark existing CTTA methods. Established the core tension:

| Baseline | R10 mean | R150 / final | Verdict |
|----------|----------|---|---|
| No Adaptation | 23.3 | 23.3 | Stable, no learning |
| TENT-continual step=1 | 30.92 (peak R10) | 7.90 | Improves quickly, **collapses by R80** |
| TENT-continual step=10 | 32.44 (peak R2) | 5.10 | Even faster peak, faster crash |
| MLMP-continual step=1 | 25.7 | crashes | Slow degradation |
| CoTTA | 23.4 | 23.4 | **Stable but never improves** |
| **MLMP episodic step=10** | **30.6** | **30.6** | **Best quality, requires per-sample reset (unrealistic)** |

> **The core tension**: every CTTA method that *improves* eventually *collapses*; every method that is *stable* never *improves*. MLMP-episodic gets 30.6 mIoU but only works because it resets per sample — useless for deployment.

**Phase 2 (this work)**: Design a new CTTA method that is both **stable (CoTTA-level)** and **high-quality (MLMP-episodic-level or better)**.

The current best result — **TENT-DivGate at mean 30.49 over the first 26 rounds, no collapse** — is the first method we have that *matches* the episodic upper bound under realistic continual conditions. The remaining test is whether it survives the R30-R80 window where natural TENT loses 11+ mIoU.

---

## 2. Research Arc — Six Methods, Three Failures, One Success (so far)

The work proceeded in three phases. Each phase ended with a hard-won negative result that reframed the problem for the next phase.

### Phase A: CMA-continual — directional loss instead of entropy

**Hypothesis** (`proposal.md`): Entropy-based losses (TENT, MLMP) collapse because `L = -Σ p log p` rewards *confidence* not *correctness*. The frozen CLIP text encoder provides 19 fixed semantic anchors. Replacing entropy with cosine alignment to the predicted class's text embedding should give a directional signal that breaks the trivial solution.

**Loss**: `L_CMA = -mean_{i ∈ top-K%} cos(v_i, t_{ĉ_i})` over top-K% confident pixels.

**Implementation**: `adapt/cma_continual.py`. Spec: [docs/cma_continual_spec.md](cma_continual_spec.md).

**Result** (`save/ACDCDataset/cma_continual_k_0.2/`, 145 rounds): peak **26.82 @R16**, then dropped to **1.20 by R34, dead state forever**. Mean = 5.54. The k=0.5 variant peaked slightly higher (27.40 @R18) and crashed slightly later (R40), but qualitatively identical.

**Diagnosis** (post-mortem in [cma_continual_spec.md](cma_continual_spec.md) §9):
> The hypothesis was wrong at the deepest level. The problem is NOT the specific form of entropy. It's that **any loss whose target depends on the current model's predictions creates a confirmation bias feedback loop**. CMA's directional pull is *more* aggressive than entropy's confidence sharpening, so it actually collapses *faster* (R34 vs TENT's R80).

### Phase B: CMA-Proto-continual — frozen external anchor

**Refined hypothesis**: To prevent the feedback loop, the loss must include at least one term whose target is **completely independent of the current model's predictions** — an external anchor.

**Loss**: three cosine-alignment terms sharing the same Top-K% mask:
```
L = λ_CMA · cos(v_i, t_{ĉ_i})        # text anchor (Phase A)
  + λ_src · cos(v_i, p^src_{ĉ_i})    # frozen source visual prototype
  + λ_tgt · cos(v_i, p^tgt_{ĉ_i})    # EMA target visual prototype
```

The source prototype `p_src` is computed once before the stream begins (using ACDC fog as source proxy with cross-prompt-agreement filter) and **never updated** during adaptation.

**Implementation**: `adapt/cma_proto_continual.py`. Spec: [docs/cma_proto_continual_spec.md](cma_proto_continual_spec.md).

**Result** (`save/ACDCDataset/cma_proto_continual_step_1/`, 150 rounds): peak ~26.1 @R16, **collapsed to dead state by R54**. Better than plain CMA (R34 → R54, +20 rounds) but qualitatively identical end state.

**Diagnosis** ([proposal_after_cma.md](../proposal_after_cma.md) §0.3) — the **deepest** insight of the project so far:
> Even with `p_src` literally frozen, the confirmation-bias loop remains. The problem is **NOT in the target vector** — it's in the **class index selection**. All three terms have the form `-cos(v_i, target_X[ĉ_i])` where `ĉ_i = argmax(current_model(x_i))`. Whether `target_X` is text, frozen-source-proto, or EMA-target-proto, the "which prototype to pull toward" decision is made by the very model we're updating. So the model can hallucinate a class assignment, get pulled toward that class's anchor (regardless of which anchor type), and the next pseudo-label leans even harder in that direction.
>
> **Any loss whose pseudo-label `ĉ_i` comes from the current model has confirmation bias, period.**

### Phase C: Reframe — `proposal_after_cma.md` and three new directions

After CMA-Proto failed, we wrote [proposal_after_cma.md](../proposal_after_cma.md) which acknowledged that **loss-level external anchors don't escape the feedback loop** and proposed three structurally different directions:

- **Direction A (Layer-Stratified Restoration)**: protect late layers (semantic) with CoTTA-style stochastic restoration toward source weights, let early layers (low-level) adapt freely. Loss stays as CMA; defense moves to the optimizer side.
- **Direction B (Diversity-Gated CMA)**: monitor batch-level marginal class entropy `H_margin`. When `H_margin` is high (healthy diversity) run aggressive (no restore); when it falls (collapse predictor), apply CoTTA restoration. Label-free monitoring as a contribution.
- **Direction C (Two-Timescale Meta-Adaptation)**: separate "fast" per-sample adapter from "slow" cross-sample model. Most direct break of the feedback loop, but most complex. Deferred.

We implemented A and B in parallel.

### Phase D: CMA-Layered (Direction A)

**Implementation**: `adapt/cma_layered_continual.py`. Spec: [docs/cma_layered_continual_spec.md](cma_layered_continual_spec.md). Plan: [docs/cma_layered_continual_plan.md](cma_layered_continual_plan.md).

**Mechanism**: after each `optimizer.step()`, every visual-encoder LN parameter is stochastically restored toward source with a probability that depends on which transformer block it belongs to:
- early ([0, early_cutoff) + ln_pre): `early_rst` (default 0.001)
- mid ([early_cutoff, late_cutoff)): `mid_rst` (default 0.01)
- late ([late_cutoff, 24) + ln_post): `late_rst` (default 0.05)

**Six runs spanning rate × cutoff sweep** (`save/ACDCDataset/cma_layered_continual_*/`):

| Run | rates | cutoffs | rounds | peak / mean | verdict |
|---|---|---|---|---|---|
| step_1 (proposal default) | 0.001/0.01/0.05 | 8/16 | 150 | 23.38 / 23.21 | flat — like CoTTA |
| cutoff_12_16 | 0.001/0.01/0.05 | 12/16 | 150 | 23.49 / 23.28 | flat |
| cutoff_12_20 | 0.001/0.01/0.05 | 12/20 | 150 | 23.48 / 23.26 | flat |
| rate_0.001_0.01 | 0.0001/0.001/0.01 | 8/16 | 47 | 23.73 / 23.08 | gentle rise |
| rate_0.001_0.05 | 0/0.001/0.05 | 8/16 | 47 | 23.75 / 23.09 | already past peak |
| rate_0.005_0.01 | 0/0.005/0.01 | 8/16 | 48 | 23.75 / 23.13 | already past peak |

**Diagnosis**:
1. Default rates (proposal §1.4 values) are too strong → method becomes essentially CoTTA.
2. Loose rates allow some adaptation, but **cap at ~23.7 mIoU** — far below CMA's natural peak (27.4) and miles below TENT's peak (32.4). The CMA loss with top-K=0.2 mask has a relatively low ceiling; layer-stratified restoration cannot raise that ceiling, only stabilize it.
3. **Cutoff position barely matters** within 8-12 / 16-20 range — the bottleneck is the rate, not the boundary.
4. **Conclusion: Direction A alone is insufficient** for breaking through 24 mIoU. Negative result, but a clean one.

### Phase E: CMA-DivGate (Direction B on CMA base)

**Implementation**: `adapt/cma_divgate_continual.py`. Spec: [docs/cma_divgate_continual_spec.md](cma_divgate_continual_spec.md). Plan: [docs/cma_divgate_continual_plan.md](cma_divgate_continual_plan.md).

**Mechanism**:
- Buffer per-batch marginal class probs `probs.mean(dim=[0, 2, 3])` for `monitor_interval=50` batches.
- After N batches: aggregate, compute `H_margin = -Σ p log p`, pick mode.
- Three-tier mode → flat stochastic restore rate: `aggressive` (rst=0), `cautious` (rst=0.005), `brake` (rst=0.05).
- Loss: identical to CMA-continual.

**Result** (`save/ACDCDataset/cma_divgate_continual/`, 150 rounds): peak **26.82 @R16**, R150 = 19.32, mean = 21.21.

**Sub-experiments** (`save/ACDCDataset/cma_divgate_continual_brake_*`): the 4 brake_rst variants (0.001, 0.005, 0.02, 0.05) were **bit-identical for the first 33 rounds**. This was investigated and turned out to be expected: H_margin never dropped below `h_warning = 1.2` in early rounds, so brake mode never fired and the only effective restoration was cautious_rst (which was constant at 0.005 across all four runs). The gate behaves as a 2-tier (aggressive ↔ cautious) controller, not 3-tier.

**Diagnosis**:
1. **The gate works**: mean 21.21 vs CMA's 5.54 = **+15.7 mIoU lifted from dead state**. Definitive proof that H_margin is a usable label-free collapse predictor.
2. **But the absolute level (21.21) is still BELOW No Adapt baseline 23.34**: the gate prevents catastrophic collapse but the brake is sticky enough that mean never recovers to source level.
3. **Most importantly**: even an *ideal* CMA-DivGate is bounded above by CMA's natural peak (27.4). **The base loss is the ceiling**.

### Phase F: TENT-DivGate (Direction B on TENT base) — **COMPLETE, MAIN RESULT**

**Implementation**: `adapt/tent_divgate_continual.py`. Spec: [docs/tent_divgate_continual_spec.md](tent_divgate_continual_spec.md).

**Motivation**: TENT-continual's natural peak (32.4 step=10, 30.9 step=1) **exceeds MLMP-episodic (30.6)**. If the same gate that successfully prevented the dead state for CMA can do the same for TENT, the result might match or beat episodic. This was the crucial methodological correction — Direction B's brake mechanism wasn't broken; we'd just been pairing it with the wrong base loss.

**Mechanism**: identical gate to CMA-DivGate, but base loss is pure TENT pixel-wise entropy (no Top-K mask, matching `tent_continual.py`).

#### Phase F-1: Initial baseline run (h_thr=1.8, h_warn=1.2, cau_rst=0.005)

`save/ACDCDataset/tent_divgate_continual/` — 150 rounds complete.

| Round | TENT-natural | TENT-DivGate | Δ |
|---|---|---|---|
| R1 | 23.89 | 23.89 | 0 (bit-identical, gate in aggressive/rst=0) |
| R10 | 30.92 | 30.92 | 0 (still aggressive) |
| R20 | 32.90 | 32.50 | −0.40 (gate has fired, cautious mode) |
| R80 | ~8 | ~29 | **+21 (TENT collapsed; DivGate stable)** |
| R150 | 7.90 | 29.02 | **+21.1** |

**Overall**: mean=30.14, peak=32.50@R20, R150=29.02. **First method to match MLMP-episodic (30.6) over the full 150 rounds.** Natural TENT crashes by R80; DivGate holds above 29 to R150.

Key observation: at h_warning=1.2, brake mode **never fires**. Gate behaves as 2-tier (aggressive ↔ cautious), same as CMA-DivGate. Also added `divgate_log.txt` file logging (H_margin trajectory, one row per 50-batch window).

#### Phase F-2: Threshold adjustment + cautious_rst sweep

**Change**: lowered h_threshold 1.8→1.6 (extends aggressive window, slightly higher peak) and raised h_warning 1.2→1.4 (gives brake mode a realistic trigger region). Then swept cautious_rst across 5 values.

All 5 new variants use h_threshold=1.6, h_warning=1.4. Results (150 rounds each):

| cautious_rst | mean (150R) | peak | R150 | gate: agg/cau/brake |
|---|---|---|---|---|
| 0.001 | 27.13 | 32.90@R19 | 22.79 | still drifts late |
| 0.003 | 28.32 | 32.90@R19 | 23.61 | drifts late |
| 0.005 | 31.25 | 32.90@R19 | 29.92 | stable |
| 0.008 | 31.49 | 32.95@R26 | **31.49** | very stable |
| **0.010** | **31.59** | **32.96@R27** | **31.34** | **best overall** |

Save dirs: `save/ACDCDataset/tent_divgate_continual_cau_rst_{value}/`

**Key findings from the sweep**:
1. **cau_rst=0.01 is the new best**: mean=31.59 beats MLMP-episodic by +1.0 mIoU; R150=31.34 stays stable far above No Adapt (23.3).
2. **The threshold change alone helped**: new variants peak at 32.90-32.96 vs baseline's 32.50 (h_threshold=1.6 allows slightly more aggressive early).
3. **R150 stability jump**: 29.02 (baseline) → 31.34-31.49 (cau_rst≥0.008). The method is genuinely stable through 150 rounds.
4. **Brake mode still never fires**: even with h_warning raised to 1.4. H_margin distribution (from cau_rst=0.01 log) shows 1204 aggressive / 14 cautious / 0 brake windows. H_margin min=1.576, max=2.646, mean=1.788, median≈1.751.

#### Phase F-3: H_margin distribution analysis (basis for next sweep)

H_margin histogram from cau_rst=0.01 run (1218 total 50-batch windows):

| H_margin range | window count | % of time |
|---|---|---|
| [1.5, 1.6) | 14 | 1.1% ← current cautious zone |
| [1.6, 1.7) | 308 | 25.3% |
| [1.7, 1.8) | 511 | 41.9% ← median here |
| [1.8, 1.9) | 234 | 19.2% |
| [1.9+) | 151 | 12.4% (early rounds ~2.4-2.6) |

At h_threshold=1.6, gate is in aggressive mode 98.9% of the time — the cautious mechanism barely activates. **Raising h_threshold** moves the threshold into the dense part of the distribution, triggering cautious restoration far more often.

---

## 3. Why TENT-DivGate Works Where CMA-DivGate Failed

Two factors compound:

| Factor | CMA-DivGate | TENT-DivGate |
|---|---|---|
| Natural peak | 27.4 | 32.9 |
| Drift speed | R16 → R34 dead (18 rounds) | R20 → R80 down 12 mIoU (60 rounds) |
| Per-batch signal | top-K 20% pixels | 100% pixels |
| Gate reaction time | insufficient | sufficient |

CMA's drift was too fast for cautious-mode rst=0.005 to compensate; brake fires too late. TENT's drift is slow enough that the same cautious mode accumulates enough restoration per monitor window to keep the model in its high-plasticity regime. The gate didn't change between the two methods — only the speed of the underlying drift.

This is a useful general lesson: **the same anti-collapse mechanism can succeed or fail depending on whether its reaction time matches the base loss's drift time-scale**.

---

## 4. Phase G: Cityscapes Generalisation — COMPLETE (2026-05-03 → 2026-05-10)

**Motivation**: TENT-DivGate has been validated on ACDC (real adverse weather, 4 conditions). To support a generalisation claim in the paper, we need to show it works on a different dataset.

**Setup**: CityscapesDataset val (500 clean images), 15 ImageNet-C corruptions on-the-fly via `CorruptTransform`, 150 rounds, evaluate-before-adapt. All scripts under `bash/cityscapes_continual/`. Design spec: [2026-05-03-cityscapes-continual-divgate-design.md](2026-05-03-cityscapes-continual-divgate-design.md).

**Scripts written** (all use 15-corruption bash array, commentable per-line, `CORRUPTIONS_LIST` env-var override):
- `no_adapt.sh`, `tent_continual.sh`, `cotta.sh`, `mlmp_continual.sh`, `mlmp_episodic.sh`, `tent_divgate_continual.sh`

**Cache note**: `CorruptTransform` caches generated corruptions to `data/.cache/corruptions/<md5>_<name>_s5.npy`. First round is slow (glass_blur: ~30-60s/image at 2048×1024, ~3-5h total for 500 images). Rounds 2+ load from cache and are fast. Total cache size ≈ 45 GB for all 15 corruptions × 500 images.

---

### G-1: Final Experiments and Results

All save dirs under `save/CityscapesDataset/`.

| Experiment | Rounds | Trend |
|---|---|---|
| tent_divgate_continual all-15 h_thr=1.6 | 150 | flat, no improvement |
| tent_divgate_continual all-15 h_thr=1.7 | 150 | flat, no improvement |
| tent_divgate_continual weather-5 h_thr=1.6 | 150 | declining |
| tent_divgate_continual weather-5 h_thr=1.7 | 150 | declining |
| tent_divgate_continual weather-5 h_thr=2.0 | 150 | (diagnostic for high-restoration test) |
| tent_continual weather-5 | 150 | (diagnostic — baseline for DivGate vs no-gate) |
| tent_continual single-corruption fog | 150 | (diagnostic — pure TENT under no inter-corruption drift) |
| mlmp_continual all-15 | 150 | catastrophic collapse |
| mlmp_divgate_continual weather-5 h_thr=1.6 | 150 | partial stability |
| **MLMP episodic weather-5** | — | upper bound: mean=22.60 |

MLMP episodic weather-5 per condition: snow=21.86, frost=17.17, fog=24.56, brightness=32.35, contrast=17.04.

---

### G-2: Key Finding — TENT Does Not Create Positive Adaptation Signal on Cityscapes

**Root cause: adaptation headroom gap.**

| Dataset | Source model | MLMP episodic | Headroom | TENT-DivGate |
|---|---|---|---|---|
| ACDC | 23.3 | 30.6 | **+7.3 mIoU** | 31.59 ✅ beats episodic |
| Cityscapes weather-5 | 21.32 | 22.60 | **+1.27 mIoU** | 18.11 ❌ worse than source |

The fundamental problem: **TENT's entropy minimization requires a large performance gap to exploit**. On Cityscapes weather corruptions (brightness, contrast, fog, snow, frost at severity=5), NA-CLIP is already near-optimal. With only 1.27 mIoU of headroom and 500 samples per condition, TENT cannot find a consistent gradient direction; instead it drifts away from the source optimum (R35 = 18.11, **−3.21 below source model**).

**DivGate analysis**: H_margin on Cityscapes (mean=2.03) is +0.25 nats higher than ACDC (mean=1.79). With h_thr=1.6/1.7, the gate fires only 2.5–3.8% of windows — almost identical firing rate to ACDC's 1.1%. The gate is NOT miscalibrated. The problem is upstream: DivGate maintains a plateau that TENT never builds. When TENT produces no plateau, DivGate has nothing to maintain.

**Cross-group contamination in all-15 setting** (separate issue): Noise corruptions (gaussian, shot, impulse) collapse R1→R2 by −3 to −5 mIoU while weather/digital stay flat or improve. The 15-condition sequence creates cross-group forgetting: adaptation toward the majority (weather+digital, 10/15 conditions) degrades the minority (noise, 3/15). DivGate does not solve this because it monitors aggregate H_margin, not per-group diversity.

**Weather-5 worse than all-15** (counterintuitive): with fewer conditions, the model is pushed harder toward the weather-optimal state, but that state is WORSE than source (pure TENT harm with no recovery). In all-15, diverse corruptions partly buffer each other → higher mean H_margin → more windows in aggressive mode → less over-restoration.

---

### G-3: What the R20/R26 Spikes Tell Us (weather-5 h_thr=1.7)

R20=20.56 and R26=20.88 are anomalous peaks in the otherwise flat-declining weather-5 trajectory. These coincide with windows where the gate fired cautious mode. The model briefly recovered toward source-like performance. **This validates the gate mechanism** — restoration toward source IS helpful on Cityscapes weather, but it fires too infrequently to maintain improvement. A higher h_threshold would trigger restoration more often and might sustain performance at 20+.

---

### G-4: Open Questions for Cityscapes

**Highest priority** (determines paper framing):

1. **Run `tent_continual` (no gate) on weather-5**: Does pure TENT eventually collapse (< 10 mIoU) like ACDC? If yes, TENT-DivGate provides stability benefit even without improvement. If no (both flat), DivGate provides no benefit on Cityscapes. Script ready: `bash bash/cityscapes_continual/tent_continual.sh` with `CORRUPTIONS_LIST="fog snow frost brightness contrast"`.

2. **Run `no_adapt` on weather-5 for 10 rounds**: Confirm source model baseline = constant 21.32. Any method worse than this is harmful. Script ready: `bash bash/cityscapes_continual/no_adapt.sh` with weather-5 subset.

3. **Try h_thr=2.2–2.3 on weather-5**: This would put Cityscapes firing rate at ~20-30% (analogous to "frequent but not constant" cautious mode). The R20/R26 spikes suggest restoration helps; we need it to fire more often. Hypothesis: TENT-DivGate with high h_thr would maintain ~20+ mIoU by frequently restoring away from TENT's harmful drift.

4. **Run MLMP episodic on all-15**: Check whether noise/blur corruptions have significantly more headroom. If so, all-15 might provide enough headroom in aggregate for a positive Cityscapes result.

---

### G-5: Paper Framing Options

| Option | Framing | Risk |
|---|---|---|
| **A (recommended)** | ACDC = primary result. Cityscapes = limitation/analysis section. Show headroom dependency as finding. | Low — honest, ACDC result is strong enough |
| **B** | Cityscapes weather-5 with high h_thr (2.2+). If this works (20+), adds a generalization claim. | Medium — needs new experiments |
| **C** | Characterise Cityscapes all-15 as harder benchmark; show TENT-DivGate is stable even if not improving (vs mlmp_continual collapse). | Low — valid if tent_continual also eventually collapses |

**Do NOT** report weather-5 R35=18.11 as a positive result — it is below source model (21.32).

---

## 5. Phase H: VOC20 Generalisation — COMPLETE (2026-05-08 → 2026-05-18)

**Motivation**: After Cityscapes failed to generalise, VOC20 was the second test of whether the methods transfer to another synthetic-corruption dataset.

**Setup**: PascalVOC20Dataset val (1449 images), 15 ImageNet-C corruptions on-the-fly, 150 rounds, evaluate-before-adapt. Patch convention `INIT_RESIZE=224x224, patch=224, stride=112` → 1 patch per image (matches MLMP paper). Scripts under `bash/v20/`.

### H-1: Final Results (mean over rounds)

| Method | all-15 mean (R_last) | weather-5 mean (R_last) | Notes |
|---|---|---|---|
| **No-Adapt** | 69.00 (constant) | 70.79 (constant) | very high source baseline |
| **MLMP-episodic** | **74.11** (single-pass) | **75.35** (single-pass) | upper bound |
| CoTTA | 68.82 | 70.67 | stable, no improvement (R22 / R65) |
| TENT-continual | 1.12 (R42→0.26) | 0.98 (R150→0.26) | **catastrophic collapse** |
| MLMP-continual | 2.89 (R40→0.25) | 2.51 (R120→0.17) | **catastrophic collapse** |
| TENT-DivGate (h_thr=1.6) | 58.12 (R35→57.85, peak 62.54@R20) | 59.24 (R109→57.73) | degrades but doesn't crash |
| MLMP-DivGate (h_thr=1.6) | 66.92 (R36→66.91) | 67.89 (R145→67.65) | **best continual**, but still below source |

### H-2: Key Findings

**Finding 1 — Even larger headroom problem than Cityscapes.**
On VOC20, source = 70.79 (weather), MLMP-episodic = 75.35. Headroom = 4.56 mIoU. Cityscapes weather-5 had 1.27. ACDC had 7.3. VOC20 sits in between, but **no continual method even reaches the source baseline**, let alone the episodic upper bound. The best continual result (MLMP-DivGate, 67.89) is still 2.9 mIoU below No-Adapt.

**Finding 2 — DivGate is necessary but not sufficient on VOC20.**
- Without gate: TENT/MLMP collapse to ~0.2 mIoU (worse than random).
- With gate: TENT-DivGate plateaus at 58, MLMP-DivGate at 67 — better than collapsed but still below source.
- DivGate's job (prevent collapse) succeeds. It cannot, however, produce positive adaptation.

**Finding 3 — MLMP loss beats TENT loss on VOC20 in the continual setting.**
MLMP-DivGate 67.89 vs TENT-DivGate 59.24 (weather-5). This is the inverse of ACDC, where TENT loss was strictly better. Hypothesis: VOC20's 1-patch-per-image setting makes per-sample entropy noisier; MLMP's multi-prompt averaging stabilizes the signal.

**Finding 4 — VOC20 is a more brittle setting than Cityscapes.**
- Cityscapes TENT-continual reached round 80+ before collapsing.
- VOC20 TENT-continual collapses to single digits within 10–20 rounds.
- Likely cause: 1 patch/image × 1449 images = sparse gradient updates per condition.

### H-3: Generated Figures

All saved under `save/PascalVOC20Dataset/`:
- `voc20_methods_all15.{png,svg}` — methods comparison, 15 corruptions
- `voc20_methods_weather.{png,svg}` — methods comparison, weather-5 subset
- `hmargin_voc20_all15.{png,svg}` — H_margin trajectories, TENT-DivGate vs MLMP-DivGate
- `hmargin_voc20_weather.{png,svg}` — H_margin trajectories, weather-5

Plot scripts: `plot_voc20_methods_comparison.py`, `plot_hmargin_voc20.py`.

### H-4: What VOC20 Adds to the Story

VOC20 confirms that the headroom dependency is **real and dataset-independent** — it's not a Cityscapes quirk. The pattern is now:

| Dataset | Source | Episodic | Headroom | Best continual | Outcome |
|---|---|---|---|---|---|
| ACDC | 23.3 | 30.6 | 7.3 | **TENT-DivGate 31.59** | ✅ beats episodic |
| Cityscapes weather-5 | 21.3 | 22.6 | 1.3 | TENT-DivGate 18.1 | ❌ below source |
| VOC20 weather-5 | 70.8 | 75.4 | 4.6 | MLMP-DivGate 67.9 | ❌ below source |
| VOC20 all-15 | 69.0 | 74.1 | 5.1 | MLMP-DivGate 66.9 | ❌ below source |

The transition is sharp: somewhere between 5 and 7 mIoU headroom, continual TTA stops working with current methods.

---

## 6. Phase I: SAR-Continual and EATA-Continual — IMPLEMENTED, NOT YET RUN (2026-05-17~19)

**Motivation**: Both Cityscapes and VOC20 results show that TENT-loss-based methods cannot find positive gradients in low-headroom regimes. SAR and EATA were chosen as candidate fixes (see [docs/2026-05-17-sar-eata-v20-design.md](2026-05-17-sar-eata-v20-design.md) for survey reasoning):

- **SAR** (ICLR 2023): SAM optimizer finds flatter loss minima that generalize better when headroom is small; reliable sample filtering reduces noise; model recovery resets when entropy spikes.
- **EATA** (ICML 2022): Pre-stream Fisher information defines a *structured* anti-forgetting penalty (Fisher-weighted EWC), in contrast to DivGate's *stochastic* restoration. Reliable + non-redundant sample filtering.

### I-1: Implementation Status

| File | Status |
|---|---|
| `adapt/sam.py` | SAM optimizer wrapper for SAR |
| `adapt/sar_continual.py` | SAR-Continual main class |
| `adapt/eata_continual.py` | EATA-Continual main class (with `obtain_src_fisher()` pre-stream hook) |
| `adapt/__init__.py` | both registered |
| `main_continual.py` | CLI args added; EATA pre-stream Fisher dispatch (mirrors DPCore's `obtain_src_stat`) |
| `bash/v20/sar_continual.sh` | VOC20 runner |
| `bash/v20/eata_continual.sh` | VOC20 runner |
| `bash/ACDC_10_round/sar_continual.sh` | ACDC runner |
| `bash/ACDC_10_round/eata_continual.sh` | ACDC runner (uses `--src_corruption fog` as proxy, no clean ACDC split) |

Both methods smoke-tested (`--debug`, 2 corruptions × 5 batches) on VOC20 and ACDC; produce expected outputs (`results_all_rounds.txt`, per-batch `sar_log.txt` / `eata_log.txt`).

### I-2: Deviations from Original Papers

Documented in detail in the design spec. Highlights:
- LayerNorm-only updates (not BatchNorm) — paper used BN
- Per-pixel entropy averaged per image (not image-level) — segmentation adaptation
- SAR augmented-view variance filtering NOT implemented (only entropy-based filtering)
- EATA condition-aware Fisher NOT implemented (single Fisher matrix)
- `e_margin = 0.4 × ln(num_classes)` — kept as paper formula (1.198 for VOC20, 1.178 for ACDC)

### I-3: Next Experiments (in priority order)

1. **SAR-Continual on VOC20 weather-5** (150 rounds, default hyperparams). Compare vs MLMP-DivGate 67.89.
2. **EATA-Continual on VOC20 weather-5** (150 rounds, fisher_alpha=2000). Same comparison.
3. **SAR + EATA on ACDC** (150 rounds, 4 conditions). If they match or beat TENT-DivGate (31.59), the methods generalise.
4. **Diagnostic on logs**: after Phase I-1/2, count `was_filtered` (SAR) and `non_redundant` (EATA) rates from `*_log.txt`. If non_redundant < 20%, increase `d_margin` from 0.05 to 0.1-0.2 (smoke test showed it's too strict).
5. **If results show promise**: sweep `sam_rho ∈ {0.01, 0.05, 0.1}` for SAR, `fisher_alpha ∈ {0, 200, 2000, 20000}` for EATA.

### I-4: Concrete Run Commands

```bash
# VOC20 weather-5
CORRUPTIONS_LIST="snow frost fog brightness contrast" \
SAVE_DIR="save/PascalVOC20Dataset/sar_continual_weather/" \
bash bash/v20/sar_continual.sh

CORRUPTIONS_LIST="snow frost fog brightness contrast" \
SAVE_DIR="save/PascalVOC20Dataset/eata_continual_weather/" \
bash bash/v20/eata_continual.sh

# ACDC
bash bash/ACDC_10_round/sar_continual.sh
bash bash/ACDC_10_round/eata_continual.sh
```

---

## 7. Phase I Results — SAR Ran on ACDC + VOC20 (post-bug-fix, 2026-05-21)

### I-5: Direction Bug in SAR Recovery — found and fixed

The original implementation had **the model-recovery comparison reversed**:

```python
# WRONG (initial implementation, was in adapt/sar_continual.py)
if self.loss_ma > self.e_0:
    self._model_recovery()

# CORRECT (per SAR paper Algorithm 1 + official mr-eggplant/SAR/sar.py)
if self.loss_ma < self.e_0:
    self._model_recovery()
```

The intent: when entropy EMA *drops* (model collapsed to over-confident trivial predictions), reset. The reversed `>` made recovery fire when entropy was *high* — opposite of the paper. With pre-fix `E_0=0.7` on ACDC, recovery fired only 3 times in 60,900 batches (loss_ma usually stayed < 0.7 after warmup, so the wrong condition didn't trigger) — the run was effectively "SAR without recovery" (SAM + reliable filter only).

**Fix landed in [adapt/sar_continual.py:209](adapt/sar_continual.py#L209)**: direction flipped, threshold rescaled to `E_0=0.1` (paper uses 0.2 for ImageNet's 1000 classes; we scale by `0.2 × ln(19)/ln(1000) ≈ 0.085 → 0.1` for our 19-20 class setting). Updated:
- [docs/2026-05-17-sar-eata-v20-design.md:61](2026-05-17-sar-eata-v20-design.md) (spec)
- [bash/ACDC_10_round/sar_continual.sh](../bash/ACDC_10_round/sar_continual.sh), [bash/v20/sar_continual.sh](../bash/v20/sar_continual.sh), [bash/cityscapes_continual/sar_continual.sh](../bash/cityscapes_continual/sar_continual.sh) (E_0 updated to 0.1)

### I-6: ACDC Results (`save/ACDCDataset/sar_continual_weather/`, 150R)

Pre-fix (the data file is what's there now — `>` direction with `E_0=0.7`):

| Method | Mean | Peak | R150 | Pattern |
|---|---|---|---|---|
| **TENT-DivGate** (h_thr=1.6, cau_rst=0.01) | **31.59** | 32.96 @ R27 | **31.34** | Stable 31-33 entire 150R |
| **SAR-continual** (pre-fix) | 30.32 | **33.38** @ ~R15 | 25.31 | "W-shape": dips at R50/R100/R150 |
| MLMP episodic | 30.60 | — | — | Per-sample reset baseline |
| No Adapt | 23.34 | — | — | |

**Key observation**: SAR's peak (33.38) beats TENT-DivGate's peak (32.96), but SAR has 3 periodic deep dips down to ~24-25 mIoU — characteristic of "drift → catastrophic state → recovery → re-adapt" cycles. TENT-DivGate's continuous DivGate brake prevents the dips entirely. **This is what motivates Phase J (SAR-DivGate).**

Plot: [save/ACDCDataset/acdc_4methods_comparison.png](../save/ACDCDataset/acdc_4methods_comparison.png) (script: [plot_acdc_4methods.py](../plot_acdc_4methods.py)).

### I-7: VOC20 Results (`save/PascalVOC20Dataset/sar_continual_weather*/`)

Two runs exist:
- `sar_continual_weather/` — pre-fix, 42 rounds, mean ~67, declining trajectory
- `sar_continual_weather_v2/` — post-fix, 113 rounds, mean ~71, flat trajectory (R1 70.95, R2 70.93 — essentially flat from start because correct `<` direction + small `E_0=0.1` rarely triggers reset on healthy data)

| Method | Mean | Peak | Last R |
|---|---|---|---|
| MLMP episodic | **75.35** | — | — |
| No Adapt | 70.79 | — | — |
| SAR-continual (post-fix v2) | ~70.9 | ~70.95 | R113=70.94 |
| SAR-continual (pre-fix) | ~67-68 | 70.90 | R41=66.94 |
| TENT-DivGate h_thr=1.6 | ~57 | — | R109=57.73 |

Plot: [save/PascalVOC20Dataset/voc20_4methods_comparison.png](../save/PascalVOC20Dataset/voc20_4methods_comparison.png) (script: [plot_voc20_4methods.py](../plot_voc20_4methods.py); uses pre-fix `sar_continual_weather/` per user choice).

**Headroom verdict unchanged**: even with SAR's better optimizer, no continual method beats VOC20's source (No-Adapt 70.79 → MLMP-ep 75.35 = 4.6 absolute / 6.4% relative headroom is too small for entropy-based gradient signals to find improvement direction).

### I-8: Cityscapes — Script Only, Not Run

`bash/cityscapes_continual/sar_continual.sh` exists with the bug fix applied (`E_0=0.1`), but no run yet. SAR for Cityscapes is lower priority since the headroom there (1.3) is the smallest of all three datasets — least likely to show useful adaptation.

---

## 8. Phase J: SAR-DivGate-Continual — IMPLEMENTED, NOT YET RUN (2026-05-21)

### J-1: Hypothesis

SAR's W-shape dips on ACDC (§I-6) are exactly the failure mode DivGate's 3-tier graded restoration was designed to prevent. The two mechanisms address **disjoint failure modes** and use **disjoint signals**:

| | Signal | Defense |
|---|---|---|
| SAR recovery | Per-pixel entropy EMA (sample-level confidence) | Binary hard reset, fires only at extreme collapse |
| DivGate | Marginal class entropy H_margin (population-level diversity) | Continuous 3-tier graded stochastic restore |

**Proposed combination** ([docs/sar_divgate_continual_spec.md](sar_divgate_continual_spec.md)):
- **Keep** from SAR: reliability filter + SAM optimizer + pixel-level filter (SAM is the only mechanism that *raises* the peak — found flatter LN minima)
- **Drop** from SAR: loss_ma EMA tracking + hard model_recovery (replaced by DivGate)
- **Add** from TENT-DivGate: H_margin marginal buffer + 3-tier stochastic restore

Targets (ACDC): basic mean ≥ 28 (DivGate prevents W-shape), target ≥ 31.59 (matches TENT-DivGate, confirms SAM doesn't hurt), **ideal ≥ 32.5 + peak ≥ 33** (SAM lifts peak, DivGate holds it — research win).

### J-2: Implementation Status

| File | Purpose | Status |
|---|---|---|
| [adapt/sar_divgate_continual.py](../adapt/sar_divgate_continual.py) | `SARDivGateContinual` class | ✅ written, parses |
| [adapt/__init__.py](../adapt/__init__.py) | Registers `'sar_divgate_continual'` | ✅ done |
| [main_continual.py](../main_continual.py) | `add_method_specific_args` branch | ✅ done (7 CLI args) |
| [bash/ACDC_10_round/sar_divgate_continual.sh](../bash/ACDC_10_round/sar_divgate_continual.sh) | ACDC runner | ✅ |
| [bash/v20/sar_divgate_continual.sh](../bash/v20/sar_divgate_continual.sh) | VOC20 weather runner | ✅ |
| [bash/cityscapes_continual/sar_divgate_continual.sh](../bash/cityscapes_continual/sar_divgate_continual.sh) | Cityscapes runner | ✅ |
| [docs/sar_divgate_continual_spec.md](sar_divgate_continual_spec.md) | Full design spec (14 sections) | ✅ |

**Key implementation invariants** (from spec):
- H_margin push happens on **first forward only** (before SAM perturbation) — perturbed-weight forward is a virtual position, signal would be noisy
- H_margin buffer is updated **even when reliability filter skips the sample** — keeps gate responsive under heavy filtering
- Stochastic restore fires **only after a real SAM `second_step`** — no parameters changed, nothing to restore
- `loss_ma` field **does not exist** on the instance (regression guard against accidentally keeping SAR's recovery state)
- Log tag `[DivGate-S]` (S for SAR) distinguishes from `[DivGate-T]` (TENT) and `[DivGate]` (CMA) in mixed logs

### J-3: Defaults (day-1 commands)

```
SAR side:    e_margin=1.8, sam_rho=0.05         (same as sar_continual.sh)
DivGate:     h_threshold=1.6, h_warning=1.4     (TENT-DivGate ACDC best)
             monitor_interval=50
             cautious_rst=0.01, brake_rst=0.05
Shared:      lr=1e-5, steps=1, batch_size=1
```

```bash
# ACDC (highest priority — directly tests the hypothesis)
bash bash/ACDC_10_round/sar_divgate_continual.sh

# VOC20 weather (secondary — headroom likely still wins)
bash bash/v20/sar_divgate_continual.sh

# Cityscapes weather (tertiary)
bash bash/cityscapes_continual/sar_divgate_continual.sh
```

---

## 9. Phase K: `bash/v20_acdc_matched/` — Cross-Dataset Comparability Folder (2026-05-21)

### K-1: Motivation

ACDC and VOC20 trajectories are not directly overlay-comparable because they see **different amounts of data per round**:

| Dataset | Per round | Notes |
|---|---|---|
| ACDC | 406 imgs (fog 100 + night 106 + rain 100 + snow 100) | Natural conditions |
| VOC20 weather (`bash/v20/`) | 7245 imgs (5 corruptions × 1449 val) | Synthetic, full set |
| **VOC20 ACDC-matched (`bash/v20_acdc_matched/`)** | **404 imgs (4 × 101)** | New — matches ACDC's 406 |

VOC20 currently sees ~18× more data per round → a method "stable to R150 on ACDC" may collapse at R8 on VOC20 just from 18× more gradient steps. Cross-dataset stability claims become meaningless.

### K-2: Mechanism

**Deterministic VOC20 subset** via mmseg's existing `ann_file` machinery:

1. `scripts/make_voc_subset.py --n 101 --seed 0` → writes `data/VOC/VOC2012/ImageSets/Segmentation/val_subset_101_seed0.txt` (deterministic: `random.Random(seed) + sorted()`)
2. `prepare_data(..., ann_file=...)` kwarg overrides `mm_config['ann_file']`; auto-strips `data_root` prefix so both relative and full paths work
3. `--ann_file` CLI added to **both** `main.py` and `main_continual.py`, plumbed to **stream loaders only** (not src-stat loaders for EATA / DPCore / CMA-Proto — those still see full source distribution)

Every bash script in the folder auto-generates the subset file if missing — no manual setup needed.

### K-3: Default Corruption Choice

4 closest to ACDC's natural conditions (rationale documented in each script header):

| VOC20 corruption | ACDC condition | Why |
|---|---|---|
| `snow` | snow | direct match |
| `fog` | fog | direct match |
| `frost` | rain | both wet/icy outdoor weather, surface coverage |
| `contrast` | night | low contrast ≈ poor night visibility |

`CORRUPTIONS_ARRAY` has all 15 listed; comment / uncomment to pick a different 4-set.

### K-4: Scripts (8 methods)

```
bash/v20_acdc_matched/
├── no_adapt.sh                    (tent_continual without --adapt)
├── tent_continual.sh
├── mlmp_continual.sh
├── cotta.sh                       (mt=0.999, rst=0.00, ap=0.92, aug_n=32)
├── mlmp_episodic.sh               (uses main.py)
├── mlmp_divgate_continual.sh      (h_thr=1.6)
├── tent_divgate_continual.sh      (h_thr=1.6, cau_rst=0.01)
└── sar_continual.sh               (e_margin=1.8, sam_rho=0.05, E_0=0.1 post-fix)
```

All saved to `save/PascalVOC20Dataset/v20_acdc_matched/{method}/`.

### K-5: Status

✅ Smoke-tested (no_adapt config, 1 round, --debug) — pipeline works end-to-end, subset loaded (101 imgs/corruption confirmed via progress bar `1/101 → 5/101`), output format correct.

**Not yet run for real**. Once you launch them, results will be directly overlay-comparable to ACDC's same-method trajectories.

---

## 10. Open Questions (lower priority)

### Phase F-4 (planned): h_threshold sweep — isolate gate activation rate

**Hypothesis**: at h_threshold=1.6, gate barely activates (1.1% cautious). Raising h_threshold into the dense region of the H_margin distribution (1.7-2.0) would make cautious restoration apply more frequently, potentially further improving stability.

| h_threshold | predicted cautious rate | interpretation |
|---|---|---|
| 1.7 | ~25% | light gate activation |
| 1.8 | ~67% | moderate — majority of time in cautious |
| 2.0 | ~92% | heavy — almost always restoring; only very-early aggressive |

**Plan**: fix cautious_rst=0.01 (established best from F-2), vary h_threshold ∈ {1.7, 1.8, 2.0}. h_warning stays at 1.4.

**Prediction**: 1.7 or 1.8 may improve further over 1.6; 2.0 probably trades peak for stability.

**Save dirs** (to be created):
- `save/ACDCDataset/tent_divgate_continual_hthr_1.7/`
- `save/ACDCDataset/tent_divgate_continual_hthr_1.8/`
- `save/ACDCDataset/tent_divgate_continual_hthr_2.0/`

### Other open questions (lower priority)

- **Night condition lag**: night consistently lower than fog/rain/snow (~24-26 vs 33-34 at R150). Could try condition-specific gate or monitor threshold, but that complicates the method.
- **step=10 variant**: TENT step=10 peaks higher (32.4 vs 30.9 at step=1). If gate works at step=1, step=10 with DivGate might yield even higher mean — but historically step=10 collapses faster too.
- **CMA-Layered loose-rate runs**: paused at R47-48, academic completeness only.

---

## 11. Concrete Next Actions (in order)

**ACDC + VOC20 SAR runs complete (post-bug-fix). Verdict: SAR matches MLMP-episodic on ACDC but with W-shape instability; VOC20 still hits headroom ceiling.**

**Next research move: test SAR-DivGate (Phase J) on ACDC. This is the experiment that directly tests the hypothesis "DivGate's graded restore prevents SAR's W-shape collapse."**

1. **Phase J: SAR-DivGate on ACDC** (highest priority — direct hypothesis test):
   ```bash
   bash bash/ACDC_10_round/sar_divgate_continual.sh
   ```
   Defaults: `e_margin=1.8, sam_rho=0.05, h_thr=1.6, h_warn=1.4, cau_rst=0.01, brake_rst=0.05`.
   Success criteria (§J-1):
   - Basic: R150 mean ≥ 28, no W-shape dip below 25 → DivGate replaced SAR's recovery cleanly
   - Target: mean ≥ 31.59 → SAM didn't hurt
   - **Ideal**: mean ≥ 32.5 + peak ≥ 33 → SAM lifts peak, DivGate holds it

2. **Phase K: launch v20_acdc_matched bash folder** (parallel — produces cross-dataset trajectories):
   ```bash
   bash bash/v20_acdc_matched/no_adapt.sh
   bash bash/v20_acdc_matched/tent_continual.sh
   bash bash/v20_acdc_matched/mlmp_continual.sh
   bash bash/v20_acdc_matched/cotta.sh
   bash bash/v20_acdc_matched/mlmp_episodic.sh
   bash bash/v20_acdc_matched/mlmp_divgate_continual.sh
   bash bash/v20_acdc_matched/tent_divgate_continual.sh
   bash bash/v20_acdc_matched/sar_continual.sh
   ```
   Each runs on **404 imgs/round** (vs ACDC's 406) — same workload, directly overlay-compatible with ACDC trajectories.

3. **SAR-DivGate on VOC20 + Cityscapes** (secondary — once ACDC J-result is in):
   ```bash
   bash bash/v20/sar_divgate_continual.sh
   bash bash/cityscapes_continual/sar_divgate_continual.sh
   ```
   VOC20 likely still hits headroom; goal is to confirm SAR-DivGate doesn't make things worse, and ideally beats SAR's ~70.9.

4. **If SAR-DivGate ACDC ideal tier passes**: confirm reproducibility across seeds {0, 1, 2}. Then write up. Log `divgate_log.txt` mode-transition timeline to support the story.

5. **If SAR-DivGate basic tier fails (W-shape returns)**: raise `h_warning` to 1.5, or drop `monitor_interval` to 25 for faster gate reaction. See §J-1 for the interpretation paths.

6. **EATA** (deferred — implementation complete but never run, eclipsed by SAR results; revisit if SAR-family results stall): `bash/v20/eata_continual.sh`, `bash/ACDC_10_round/eata_continual.sh`. Run command + diagnostic pattern preserved in git history (this section pre-2026-05-25).

---

## 12. Codebase State — All Methods Currently Registered

| Method name (CLI) | Class | File | Bash scripts | Spec |
|---|---|---|---|---|
| `cma_continual` | `CMAContinual` | `adapt/cma_continual.py` | `bash/ACDC_10_round/cma_continual.sh` | [cma_continual_spec.md](cma_continual_spec.md) |
| `cma_proto_continual` | `CMAProtoContinual` | `adapt/cma_proto_continual.py` | `bash/ACDC_10_round/cma_proto_continual.sh` | [cma_proto_continual_spec.md](cma_proto_continual_spec.md) |
| `cma_layered_continual` | `CMALayeredContinual` | `adapt/cma_layered_continual.py` | `bash/ACDC_10_round/cma_layered_continual.sh` | [cma_layered_continual_spec.md](cma_layered_continual_spec.md) |
| `cma_divgate_continual` | `CMADivGateContinual` | `adapt/cma_divgate_continual.py` | `bash/ACDC_10_round/cma_divgate_continual.sh` | [cma_divgate_continual_spec.md](cma_divgate_continual_spec.md) |
| `tent_divgate_continual` | `TENTDivGateContinual` | `adapt/tent_divgate_continual.py` | `bash/{ACDC_10_round,v20,cityscapes_continual,v20_acdc_matched}/tent_divgate_continual.sh` | [tent_divgate_continual_spec.md](tent_divgate_continual_spec.md) |
| `mlmp_divgate_continual` | `MLMPDivGateContinual` | `adapt/mlmp_divgate_continual.py` | `bash/{ACDC_10_round,v20,cityscapes_continual,v20_acdc_matched}/mlmp_divgate_continual.sh` | (uses same DivGate spec) |
| `sar_continual` | `SARContinual` | `adapt/sar_continual.py` (+ `adapt/sam.py`) | `bash/{v20,ACDC_10_round,cityscapes_continual,v20_acdc_matched}/sar_continual.sh` | [2026-05-17-sar-eata-v20-design.md](2026-05-17-sar-eata-v20-design.md) |
| **`sar_divgate_continual`** | **`SARDivGateContinual`** | **`adapt/sar_divgate_continual.py`** | **`bash/{ACDC_10_round,v20,cityscapes_continual}/sar_divgate_continual.sh`** | **[sar_divgate_continual_spec.md](sar_divgate_continual_spec.md)** |
| `eata_continual` | `EATAContinual` | `adapt/eata_continual.py` | `bash/{v20,ACDC_10_round}/eata_continual.sh` | [2026-05-17-sar-eata-v20-design.md](2026-05-17-sar-eata-v20-design.md) |

Plus the original baselines (`tent_continual`, `mlmp_continual`, `cotta`, `dpcore`, `mlmp` episodic) which are unchanged.

**Bash script folders**:
- `bash/ACDC_10_round/` — 4 ACDC conditions, 1120×560, ~100 imgs/condition (~406/round natural)
- `bash/cityscapes_continual/` — 15 ImageNet-C corruptions, 1120×560, 500 imgs/condition (~7500/round full)
- `bash/v20/` — 15 ImageNet-C, 224×224 single-patch, 1449 imgs (~21,735 if all 15; ~7245 weather-5)
- **`bash/v20_acdc_matched/`** — **VOC20 with deterministic 101-img subset × 4 corruptions = 404/round; designed for cross-dataset overlay with ACDC**

**Cross-dataset comparability infrastructure**:
- `scripts/make_voc_subset.py` — generates deterministic VOC subset split files
- `--ann_file` CLI arg + `prepare_data(ann_file=...)` kwarg — overrides mmseg dataset split, VOC20/VOC21 only, ignored elsewhere

All methods registered in `adapt/__init__.py::METHOD_CLASSES` and dispatched via `main_continual.py::add_method_specific_args`. Results are saved under `save/{DATASET}/{save_dir}/results_all_rounds.txt` and parsed by `parse_acdc_results.py` into `acdc_table.tex`.

**Plotting scripts** (relevant):
- `plot_acdc_4methods.py` — ACDC 4-method overlay (No-Adapt / MLMP-ep / TENT-DivGate / SAR)
- `plot_voc20_4methods.py` — same 4-method overlay, VOC20 weather subset
- `plot_voc20_methods_comparison.py` — 6-method continual comparison on VOC20 (TENT/MLMP × {-continual, -DivGate}, plus CoTTA)
- `plot_all_methods_comparison.py` — ACDC all-methods overlay (important + faded ablation)

---

## 13. Reading Map — Where to Find What

| Question | Read |
|----------|------|
| What's the overall research goal and Phase 1 baseline? | [proposal.md](../proposal.md) |
| Why did entropy/CMA collapse? Original hypothesis. | [docs/cma_continual_spec.md](cma_continual_spec.md) §9 (post-mortem) |
| Why did frozen external anchors not solve it? Deepest insight. | [docs/cma_proto_continual_spec.md](cma_proto_continual_spec.md), then [proposal_after_cma.md](../proposal_after_cma.md) §0.3 |
| What are the three new directions A/B/C? | [proposal_after_cma.md](../proposal_after_cma.md) |
| Direction A spec / Direction A experimental results | [docs/cma_layered_continual_spec.md](cma_layered_continual_spec.md), this file §2 Phase D |
| Direction B (CMA base) spec / experimental results | [docs/cma_divgate_continual_spec.md](cma_divgate_continual_spec.md), this file §2 Phase E |
| Direction B (TENT base) spec / current results | [docs/tent_divgate_continual_spec.md](tent_divgate_continual_spec.md), this file §2 Phase F |
| SAR/EATA design (Phase I, papers + deviations) | [docs/2026-05-17-sar-eata-v20-design.md](2026-05-17-sar-eata-v20-design.md), this file §6 |
| **SAR ACDC + VOC20 results, direction bug + fix** | **this file §7** |
| **SAR-DivGate (Phase J) design** | **[docs/sar_divgate_continual_spec.md](sar_divgate_continual_spec.md), this file §8** |
| **v20_acdc_matched (Phase K) folder design** | **[docs/v20_acdc_matched_spec.md](v20_acdc_matched_spec.md), this file §9** |
| Codebase architecture, implementation gotchas | [CLAUDE.md](../CLAUDE.md) |
| Implementation plan format / step-by-step recipes | `docs/*_plan.md` |

---

## 14. Key Design Decisions (Quick Reference)

Made across the project, recorded here for continuity:

1. **Update target = visual-encoder LayerNorm γ, β** in every method. Same as TENT/MLMP baselines, makes any difference attributable to loss + restoration choice, not parameterization.
2. **Top-K% confidence mask = 0.2** for CMA-family (per-batch top 20%). Robust to overall confidence shifts. Pure TENT does NOT use this — covers all pixels — for clean comparison.
3. **Source prototype init via pseudo-labels, not GT** (CMA-Proto): matches realistic CTTA. Bias mitigated via cross-prompt agreement filter.
4. **External anchors do NOT solve confirmation bias** (lesson from CMA-Proto failure): the bias is in the class-index selection, which depends on the model's current state regardless of how anchor vectors are chosen.
5. **Restoration is flat across LN params** in DivGate variants (not layer-stratified): keeps the gate as the sole experimental variable.
6. **Gate's initial mode = aggressive (rst=0)**: starts free, brakes only after observing data. The first `monitor_interval` batches always have rst=0.
7. **Default thresholds** come from `proposal_after_cma.md §2.3`. After F-2 sweep, current best: `h_threshold=1.6, h_warning=1.4, cautious_rst=0.01, brake_rst=0.05, monitor_interval=50`. Brake mode never fires even at h_warning=1.4 (H_margin min observed = 1.576, brake triggers below 1.4 which never happens). Gate effectively 2-tier. Next: explore h_threshold 1.7-2.0 (Phase F-4).
8. **Bash script convention**: `save/ACDCDataset/{method_name}_step_{steps}/` (or method-specific suffix for hyperparameter sweeps).
9. **TENT-DivGate uses pure TENT loss (no top-K mask)** to keep a clean comparison with both pure TENT and CMA-DivGate.

---

## 15. Phase L: H1 Validation Experiments — RUN & ANALYZED (2026-06-03)

**H1 (user's hypothesis)**: NA-CLIP (LAION-pretrained) has likely seen images visually similar to ImageNet-C *synthetic* corruptions during pretraining → on synthetic shifts the source weights are already near-optimal → no coherent adaptation direction exists → DivGate's stochastic-restoration equilibrium gets pinned to source. *Native* shifts (real night / adverse weather) are genuinely OOD → a real adaptation direction exists → DivGate can climb. This was meant to explain the **4/4 native win vs 3/3 synthetic loss** pattern (see §2 cross-dataset table / memory).

Two experiments ran on GPU 3, both complete:
- **Exp 1 — gradient cosine**: per-image `cos(∂L_TENT/∂{γ,β}, ∂L_CE/∂{γ,β})` at source weights. H1 predicts native cos > 0, synthetic cos ≈ 0. (3500 images: 6 native conditions × 500 + 30 synthetic conditions × ~100.)
- **Exp 3 — severity sweep**: source mIoU at corruption severity 1→5 on VOC20 and Cityscapes, all 15 ImageNet-C corruptions × 5 severities × 2 datasets = 150 cells. H1 predicts flat decay (CLIP robust) on synthetic. **Both datasets now complete (75 cells each).**

Results / figures live in `experiments/h1_validation/results/{exp1,exp3}/` (gitignored). Plot scripts: `experiments/h1_validation/{exp1_gradient_cosine,exp3_severity_sweep}/plot.py`.

### L-1: Exp 1 verdict — H1's "no direction" claim is REFUTED

| Group | median cos | %>0 | median ‖g_sup‖ |
|---|---|---|---|
| **Native** (ACDC/DZ/ND) | **+0.494** | 98% | 0.838 |
| **Cityscapes** synthetic | **+0.385** | — | 0.852 |
| **VOC20** synthetic | **+0.197** | — | 0.547 |

**The "native vs synthetic" axis collapses into "VOC vs everything else."** Cityscapes synthetic corruptions have nearly the same gradient geometry as native real shifts (cos 0.39 vs 0.49; ‖g_sup‖ 0.85 vs 0.84) — the TENT entropy gradient points the *correct* way on synthetic Cityscapes, directly contradicting H1's "cos ≈ 0 on synthetic" prediction. Even VOC's 0.20 is > 0. What dragged the old 4-corruption "synthetic ≈ 0.30" snapshot down was **VOC alone**.

Within Cityscapes the corruption *type* matters: digital (jpeg 0.48, pixelate 0.47, glass_blur 0.48, motion 0.46, fog 0.45) ≈ native; noise/zoom lower (gaussian 0.28, zoom 0.26); `frost` an outlier at 0.14 **despite** ‖g_sup‖ 0.99 (real headroom the entropy gradient can't capture).

### L-2: Exp 3 verdict — H1's "near-optimal source" claim holds for VOC only

Full 15-corruption mean relative drop sev1→5: **Cityscapes 19.3%** (steep) vs **VOC20 11.0%** (flat); Cityscapes source mIoU 20–30 (same regime as native: ACDC 23.3 / DZ 20.2 / ND 31.0), VOC source 73–78 (near-ceiling).

The four cells backfilled 2026-06-03 (`elastic_transform`, `pixelate`, `jpeg_compression`, `zoom_blur` sev3-5) revealed the digital category is **Cityscapes' robust zone** and produced one clean reversal:

| Corruption | Cityscapes drop | VOC20 drop | note |
|---|---|---|---|
| **elastic_transform** | **0.6%** (flat) | **16.6%** (steep) | only corruption where City > VOC robustness |
| pixelate | 3.1% | 2.0% | both flat |
| jpeg_compression | 11.8% | 5.3% | City steeper (normal) |
| zoom_blur | 28.2% | 22.5% | both steep |

So even *within* Cityscapes it is not uniform: noise + contrast are genuinely OOD (gaussian 42%, contrast 41.6%, shot 28.8%); digital (elastic/pixelate/jpeg) sits in H1's "CLIP already robust" regime — but only digital. This matches Exp 1 (Cityscapes digital had the highest cosines). The elastic reversal is interpretable: VOC's fine object boundaries are sensitive to local warps; Cityscapes' large-region semantics (road/building/sky) are not.

### L-3: Joint conclusion — what actually discriminates DivGate success

| | per-image grad direction (cos) | headroom (‖g_sup‖ / source mIoU) | DivGate result |
|---|---|---|---|
| Native (ACDC/DZ/ND) | high ~0.49 | large | **✅ beats episodic** |
| **Cityscapes** synthetic | **high ~0.39** | **large** | ❌ fails |
| VOC20 synthetic | low ~0.20 | small (near-optimal) | ❌ fails |

H1 as a *single* explanation is wrong. The two synthetic failures have **different mechanisms**:
1. **VOC fails from genuine lack of headroom** — source already ~76 mIoU, weak gradient (cos 0.2, small ‖g_sup‖). The original "near-optimal source" intuition is cleanly true *here only*.
2. **Cityscapes fails despite having headroom AND a correct per-image gradient direction** indistinguishable from native. Therefore its failure **cannot** be a per-image geometry / headroom problem — it must come from the **continual-stream temporal dynamics** (15 corruptions × 500 = 7500 imgs/round with abrupt corruption-type switches vs ACDC's 406/round, 4 conditions). This is the new testable hypothesis **H2**.

**Headline for paper motivation**: per-image gradient geometry cannot separate "DivGate succeeds (native)" from "DivGate fails (Cityscapes synthetic)" — both have cos ≈ 0.4 and large headroom. CTTA failure on synthetic corruptions is therefore a *temporal-dynamics* problem, not a per-image-signal problem; and VOC vs Cityscapes are two distinct failure modes (no headroom vs temporal collapse).

### L-4: H2 (next test, cheap)

Shorten / narrow the Cityscapes stream (e.g. weather-5 or a 406-matched subset) and re-run DivGate. If it then approaches the native win, the failure is confirmed to come from stream length / heterogeneity, not the corruption being synthetic.

### L-5: Operational notes from this phase
- **Disk-quota truncation**: running the Cityscapes backfill while `/home` was at quota silently wrote one **truncated** cache file (`*_zoom_blur_s3.npy`, 1.05M vs 6.29M payload), which crashed that cell with a numpy reshape error. Deleting the file + re-running fixed it. A full scan of all 37500 Cityscapes corruption-cache files (header-declared size vs actual payload) found **0 other corrupt files** — the truncation hit only the in-flight write.
- `--skip_done` reads the aggregate `all.csv`, **not** the `raw/` dir — so deleting `results/exp3/raw/` (already parsed into CSV) is safe and does not force re-runs.
- Backfilling only missing cells: `python -m experiments.h1_validation.exp3_severity_sweep.run --dataset Cityscapes --corruption <name> --severity <n> --skip_done`.

---

## 16. Phase M: Smooth Anchor vs Source-Reset DivGate (2026-06-07)

**Motivation.** Advisor suggested, at a meeting, that the diversity-gate's
restoration should anchor to a *recent* model snapshot (smooth anchor) rather
than hard-reset to the frozen source, because "elastic retreat proportional to
instability" is easier to justify in a paper than "reset to pretrained."
`tent_divgate_smooth_anchor` (merged from 學長's branch, see
[merged_2026-06-04_inventory.md](merged_2026-06-04_inventory.md) §6) implements
exactly this: restore toward a snapshot `lag(H)` batches ago, where
`lag(H)=lag_scale/(H−h_floor)`; `H≥h_ceil`→no restore, `H≤h_floor`→source.

### M-1: Head-to-head, matched gate geometry — smooth anchor LOSES on all 6 datasets
To isolate the restoration TARGET (recent-anchor vs source), each smooth run was
matched to its dataset's tuned source-reset baseline: `h_ceil←h_threshold`,
`h_floor←h_warning`, `rst←cautious_rst`; LR/rounds/resize/patch identical. Means
over the common horizon (some smooth runs still finishing — trend already past peak):

| dataset | horizon | source-reset | smooth-anchor | Δ |
|---|---|---|---|---|
| **ACDC** | 150R | **31.59** (R150 31.34) | 29.53 (R150 27.41) | **−2.06** |
| VOC20 (weather) | 100R | 59.33 | 55.01 | −4.32 |
| Cityscapes (weather) | 90R | 19.77 | 15.66 | −4.11 |
| Dark Zurich | 1200R | 22.92 | 21.83 | −1.09 |
| Nighttime Driving | 1200R | 32.62 | 32.08 | −0.54 |
| DZ+ND Combined | 600R | 27.08 | 27.05 | −0.02 |

Gap is largest where there is headroom + the gate fires a lot (ACDC, VOC20,
Cityscapes); near-tie on the night datasets. ACDC trajectory is the clearest:
both peak ~33 @R20-30, then source-reset holds ~31 while smooth-anchor **decays
steadily to ~27 by R150** — it cannot hold the gains.

### M-2: Mechanistic root cause (from `entropy_log.csv` H_margin band)
Smooth-anchor's H_margin drifts much LOWER than source-reset's (ACDC median
**1.42 vs 1.75**) — i.e. the model collapses further toward a few dominant
classes. Cause: in the active-restore band the recent snapshot is **itself
already drifted**, so pulling toward it is a *weaker correction* than snapping to
the clean source. The gate is NOT idle (smooth is 86% restore-active vs
source-reset's ~99% aggressive) — it restores *more often* but toward a
*contaminated target*, so drift accumulates. **Takeaway so far: source is the
only clean anchor; recent snapshots are contaminated.**

### M-3: Recovery sweep — DONE (ACDC 150R, 2026-06-07). **Smooth anchor CAN beat source-reset.**
Launcher `bash/ACDC_10_round/sweep_smooth_anchor.sh` (results in
`save/ACDCDataset/smooth_sweep_*/`). All-round mean / R150 mIoU, vs references
matched(ceil1.6/floor1.4/lag90)=29.53 and **source-reset=31.59 (R150 31.34)**:

| config | mean | R150 | H_margin median | takeaway |
|---|---|---|---|---|
| **D: ceil1.8, floor1.55, lag90** | **32.27** | **32.74** | 1.505 | **BEATS source-reset (+0.68)** |
| C: ceil2.0, lag1000 | 31.70 | 32.13 | 1.675 | ties/beats source |
| B: lag1000 | 31.59 | 31.34 | 1.546 | **identical to source-reset** (deep lag → source) |
| A: ceil2.0 | 31.27 | 29.48 | 1.470 | mean↑ but still late-decays |
| A: ceil1.8 | 30.96 | 28.98 | — | partial |
| B: lag300 | 30.67 | 29.79 | — | medium depth |
| win: mon25 (coupled) | 30.48 | 30.57 | — | window: marginal |
| decoup buf50/mon10, mon5 | 29.50 | 27.4 | **1.418 (=matched)** | window: **no effect** |
| matched (ref) | 29.53 | 27.41 | 1.419 | the losing config |

**Verdict on the three hypotheses:**
1. **Raise h_ceil alone (A)** — partial: lifts mean (29.5→31.3) but R150 still
   decays to ~29 (still anchors to contaminated recent snapshots). NOT the fix.
2. **Update-window / decouple (win_*, decoup_*)** — essentially **no effect**;
   `decoup` has a bit-identical H_margin band to matched (1.418 vs 1.419). The
   failure is anchor *contamination/depth*, not lag-update cadence. Refutes the
   "finer update window helps" idea.
3. **Retreat-to-source sooner / deeper anchor (D, C, B_lag1000)** — THE lever.
   `B_lag1000` literally reproduces source-reset (deep lag → source fallback).
   **`D` (raise h_floor 1.4→1.55) WINS: 32.27 > 31.59**, because it hard-retreats
   to clean source whenever H<1.55 (64.5% of batches) yet still smooth-anchors to
   recent snapshots in the healthy 1.55–1.8 band — a hybrid that beats *both*
   pure source-reset and matched-smooth. Higher H_margin median tracks higher
   mIoU across the whole sweep.

**Paper takeaway:** the advisor's smooth-anchor narrative is salvageable AND
numerically superior — but the winning recipe is "smooth recent anchor when
mildly drifting + retreat to clean source when unhealthy (high floor)", not the
naive matched-geometry smooth anchor. Source remains the only clean anchor;
the win comes from *reaching it sooner* while keeping smooth dynamics on top.
Figures: `save/_compare/smooth_sweep_{bars,trajectories}.{png,svg}`.

### M-4b: Report figure set (`plot_smooth_report.py` → `save/_compare/report_*`)
Six presentation-ready panels: `A_headline_bars` (D beats source-reset, matched
loses), `B_trajectories` (D holds vs matched decays), `C_hmargin_violin`
(matched H-collapse vs source/D healthy), `D_hmargin_vs_miou` (sweet spot, not
monotone), `E_gate_regions` (D retreats to source 72% vs matched 54%; sar_mlmp
100% restore-active), `F_lag_curve` (mechanism). Speaker-notes were drafted in
chat (lead with A+E if only 2 slides).

### M-5: Generality test on VOC20 + Cityscapes — D PRINCIPLE CONFIRMED (2026-06-10)
**Key calibration lesson:** the gate thresholds are on the *dataset's H_margin
scale*, which differs (ACDC source-reset median ≈1.75, **VOC20 ≈1.15,
Cityscapes ≈2.2**). So D's absolute 1.55/1.8 must NOT be copied — instead apply
the *principle* (raise the floor → retreat-to-source sooner) on each dataset's
own band. Floor mini-sweeps (weather + subset-101 for fast turnaround; 150R; each
vs a subset-matched source-reset baseline at the same stream).

**VOC20 (5-corr weather, subset 101, 150R) — STRONG WIN, large margin:**

| config | floor | mean-all | R150 | H_margin median | %source-region |
|---|---|---|---|---|---|
| source-reset (h1.6/1.4) | — | 59.24 | 58.72 | 1.152 | 67.2% |
| smooth ceil2.3 | 1.8 | 61.72 | 59.70 | 1.262 | 81.5% |
| smooth ceil2.3 | 2.0 | 63.43 | 62.03 | 1.398 | 83.6% |
| **smooth ceil2.3** | **2.2** | **64.70** | **64.85** | 1.528 | 85.8% |

Every smooth config beats source-reset; best (floor2.2) = **+5.46 mean, +6.13 R150**.
Monotone in floor, and floor2.2 is the only run whose **R150 ≥ early rounds (no
net decay)**. Mechanism mirrors ACDC's D: higher floor → model held in a *healthier*
state (H median rises 1.15→1.53 with floor) → higher mIoU. Note source-reset has
the **lowest** H here — on VOC20 hard reset over-restores; smooth-anchoring in the
healthy band + source below floor is strictly better. Dirs:
`save/PascalVOC20Dataset/{tent_divgate_continual_sub101_weather, tdsa_Dtune_sub101_ceil2.3_floor*}`.

**Cityscapes (4-corr snow/frost/fog/contrast, subset 101, 150R) — same direction, tiny spread:**

| config | floor | mean-all | R150 | H_margin median |
|---|---|---|---|---|
| smooth ceil2.4 | 1.9 | 18.33 | 18.10 | 2.117 |
| smooth ceil2.4 | 2.1 | 18.65 | 18.38 | 2.242 |
| **smooth ceil2.4** | **2.3** | **18.87** | **18.84** | 2.354 |

Same monotone "higher floor → higher mIoU & higher H" trend, but spread is only
+0.54 mean — Cityscapes is **headroom-limited** (peak 19.07@R3, everything clusters;
consistent with the known Cityscapes negative). Matched source-reset baseline
(`tent_divgate_continual_sub101_4corr_thr2.4`, h2.4/2.1, 4-corr+101) RUNNING on
GPU 0 — fills in the source-reset comparison line; direction already clear from
the floor sweep. Dirs: `save/CityscapesDataset/tent_divgate_smooth_anchor_Dtune_city_*`.

**Verdict:** the D principle (raise floor → retreat-to-source sooner → smooth-anchor
beats source-reset) **generalises beyond ACDC**, decisively on VOC20 (where there
is headroom) and directionally on Cityscapes (where there is not). Runner change:
`SUBSET_SIZE` env added to `bash/v20/tent_divgate_{continual,smooth_anchor}.sh` and
`bash/cityscapes_continual/*`. Figures: `python plot_smooth_generality.py` →
`save/_compare/gen_*` (per-dataset trajectories+bars, `gen_floor_vs_miou` money panel,
`gen_hmargin_vs_floor` mechanism).

### M-5b: CONFOUND CONTROL — smooth-anchor does NOT beat a *tuned* source-reset (2026-06-10) ⚠️
The M-5 VOC20 "win" compared smooth against a **badly-tuned** source-reset baseline
(h1.6/1.4, set for ACDC's H band ≈1.75, but VOC20's median is only ≈1.15 → that gate
barely sat where it should). The monotone "higher floor → better" trend really said
"this gate restores to source more / in a better band" — which a *source-reset* can
do too. Control: re-ran source-reset on VOC20 (subset-101, weather-5, 150R) at raised
thresholds, incl. one whose gate geometry is **identical** to the winning smooth
(ceil2.3/floor2.2, rst0.01) so the ONLY difference is the restoration **target**:

| config | h_thr/h_warn | rst | mean | R150 |
|---|---|---|---|---|
| source-reset orig (mis-tuned) | 1.6/1.4 | 0.01/0.05 | 59.24 | 58.72 |
| source-reset t2.0 | 2.0/1.8 | 0.01/0.01 | 62.71 | 61.43 |
| **source-reset MATCH (= smooth geom)** | **2.3/2.2** | **0.01/0.01** | **64.62** | **64.61** |
| **source-reset t2.5+brake (best)** | **2.5/2.2** | **0.01/0.05** | **65.92** | **65.21** |
| smooth-anchor floor2.2 | ceil2.3/floor2.2 | 0.01 | 64.70 | 64.85 |

**Conclusion:** gate-matched source-reset (64.62) **TIES** smooth-anchor (64.70) — the
recent-snapshot target adds **nothing** (+0.08, within determinism). A *tuned* pure
source-reset (t2.5+brake, 65.92) **BEATS** smooth by +1.22. So the lever is **WHERE
the gate bands sit + how hard you reset** (tune to the dataset's H scale), NOT the
smooth-anchor target. The earlier +5.46 "win" was an artifact of an untuned baseline.
**Implication for the ACDC D-win (M-3, 32.27 vs 31.59): likely the same confound** —
that source-reset baseline (h1.6/1.4) is below ACDC's median 1.75 → under-restores.
Must re-run a source-reset tuned to ACDC's band (≈h1.9/h_warn1.55) before claiming D > source-reset.
Figures: `python plot_smooth_control.py` → `save/_compare/control_voc20_{bars,trajectories}`.
Cityscapes matched control (`srctrl_match_h2.4_w2.3_r0.01`) running (headroom too small to be decisive).
**Paper-narrative consequence:** smooth-anchor is, at best, a *cosmetic reframing* of
source-reset (ties it) — it is not empirically superior. Pitch it as "a smoother,
continuous form of the same retreat" if at all, not as a performance win.

### M-6: SAR-MLMP-SmoothAnchor (`adapt/sar_mlmp_smooth_anchor_continual.py`)
A user-built combo (SAR reliable-filter + SAM + full MLMP UAML eval + smooth
anchor). Uses ceil2.9/floor2.2/lag150 — calibrated to **MLMP's higher H_margin
band (~2.0–2.5)**, not TENT's. Already sits in the winning regime (floor just
above its own median → 70–86% source-retreat, same principle as D). Cityscapes
weather mean 20.42 (beats tent source-reset ~19.8 — notable on a hard negative);
ACDC 30.05 (below tent-divgate 31.59). **Do NOT transplant the tent D numbers
(1.55/1.8) here — different H scale.** For a fair claim it needs a same-machinery
(MLMP+SAR) source-reset baseline, not a comparison to the TENT champion.

### M-4: Tooling added this phase
- `scripts/analyze_gate.py` — reads a run's `entropy_log.csv` (per-batch h_margin,
  all methods) or `divgate_log.txt`, prints H_margin band + gate-activity + mean/R-last/peak.
- `plot_smooth_vs_source.py` — trajectory grid + grouped-bar (→ `save/_compare/`).
- `tent_divgate_smooth_anchor` runners for all 6 datasets (ACDC/v20/cityscapes_continual/
  dark_zurich/nighttime_driving/dz_nd_combined), gate geometry env-overridable.

> **Caveat:** this is the controlled (matched-geometry) comparison. A "best-config
> vs best-config" comparison (each method tuned on the same grid) is the M-3 sweep's
> job; equal tuning budget for both arms is the fairness rule (see discussion in
> the per-method gate-threshold notes).

---

## 17. Phase N: SAR-MLMP-SmoothAnchor — full results, gate diagnosis, ceil/floor sweep (2026-06-09)

Method: `adapt/sar_mlmp_smooth_anchor_continual.py` (spec
`docs/sar_mlmp_smooth_anchor_continual_spec.md`). SAR (reliable sample+pixel
filter + SAM two-step) + **full MLMP** (7 prompts, 18-layer UAML: `mean` fusion in
adapt, `adaptive_weighted_mean` in eval) + **smooth-anchor** restore
(`lag(H)=lag_scale/(H−floor)`, fp16-CPU snapshot deque; H≥ceil→no restore,
H≤floor→source). SAR hard recovery dropped. Flags `--uaml_in_adapt` (default 1),
`--alpha_cls` (ILE, default 0). LR 5e-6. Baseline geom: ceil2.9 floor2.2 lag150
rst0.005 monitor50 e_margin1.8 sam_rho0.05.

### N-1: Baseline results, 150 rounds (target = beat MLMP-episodic)

| Dataset (round size) | R1 | Peak@R | R150 | mean-all | **episodic** | verdict | dir |
|---|---|---|---|---|---|---|---|
| Cityscapes (404: 101×snow/frost/fog/contrast) | 19.86 | 20.64@70 | 20.46 | **20.42** | 20.00 | **BEATS ✅** | `save/CityscapesDataset/sar_mlmp_smooth_anchor_continual_weather/` |
| ACDC (406: fog/night/rain/snow) | 29.73 | 30.16@122 | 30.05 | 30.05 | 30.6 (step10; 29.84 step1) | below ❌ | `save/ACDCDataset/sar_mlmp_smooth_anchor_continual/` |
| V20 (404: 101×snow/frost/fog/contrast) | 74.07 | 77.67@36 | 73.98 | 75.63 | 76.21 | below ❌ (peak>epi, drifts) | `save/PascalVOC20Dataset/v20_acdc_matched/sar_mlmp_smooth_anchor_continual/` |

Stable on all three (no collapse to 150R). Headline: **most stable high performer**;
on V20 it is the **best continual method** (next best MLMP-DivGate 71.81) and peak
(77.67) > episodic, but drifts down. Figures: `acdc_sar_mlmp_round_miou.png`,
`v20_acdc_matched_sar_mlmp_round_miou.png`, `cityscapes_acdc_matched_round_miou.png`.
Analysis scripts: `scripts/analyze_acdc_v20_sar_mlmp.py`, `scripts/analyze_cityscapes_acdc_matched.py`.

### N-2: GATE DIAGNOSIS (`gate_log.csv`) — the two failures have OPPOSITE causes

| Dataset | H_margin range (mean) | gate OFF / active-lag / SOURCE | mean active lag |
|---|---|---|---|
| ACDC | 2.23–2.50 (2.29) | **0% / 89% / 11%** | **~1730 back** |
| V20 | 2.14–3.11 (2.71) | **34% / 65% / 1%** | 528 |
| Cityscapes | 2.22–2.55 (2.32) | 0% / 93% / 7% | 1373 |

- **ACDC is PINNED, not drifting**: gate active 89% restoring toward ~1730-batch-old
  (near-source) anchors → clamped at its UAML starting point (R1 29.73 ≈ plateau
  30.05), never climbs the +0.6 it needs. **Fix = LOOSEN** (lower ceil into its
  2.3–2.5 band so healthy windows go OFF / lower floor → shallow recent anchor /
  smaller rst).
- **V20 DRIFTS off the R36 peak**: gate OFF 34% (H often ≥ ceil 2.9, up to 3.11) → no
  anchor holds the peak. **Fix = RAISE ceil** so the gate stays engaged (+ floor↑ for
  source pulls on dips). (Matches user's own finding: ceil↑ helps, floor↑ → easier
  source retreat.)
- **Cityscapes has the SAME H-regime + gate behavior as ACDC** — it only "wins"
  because its episodic bound (20.0) is low. Gate is not the differentiator there;
  headroom is. So the metric "beat episodic" depends jointly on gate working-point
  AND the absolute margin to the (dataset-specific) episodic number.
- ACDC and V20 want **opposite gate tightness** → a single config beating all three
  is not guaranteed; aiming for a robust middle region.
- SAR reliable-filter pass rate: ACDC 100%, Cityscapes 100%, V20 93% (e_margin=1.8 is
  not starving adaptation — confirmed via `sar_log.txt`).

### N-3: ceil/floor/lag/rst sweep — DONE (16 configs × 150R, 2026-06-10)

Launcher `bash/sweep_sar_mlmp_ceil_floor.sh`. Save dirs
`save/{ACDCDataset|PascalVOC20Dataset/v20_acdc_matched}/smlp_c<ceil>_f<floor>_l<lag>_r<rst>/`.
Figures + per-config table: `python scripts/plot_sweep_sar_mlmp.py` →
`save/ACDCDataset/acdc_sweep_sar_mlmp.png`, `.../v20_acdc_matched/v20_sweep_sar_mlmp.png`.

**V20 weather (episodic 76.21, baseline 75.63) — ALL 8 BEAT EPISODIC ✅**

| config (ceil/floor/lag/rst) | meanAll | peak | last | note |
|---|---|---|---|---|
| c3.2/f2.2/l150/**r0.010** | **77.15** | 78.00 | 76.93 | best mean |
| c3.4/f2.2/l300/r0.005 | 77.00 | 77.79 | 76.47 | |
| **c3.4/f2.6/l150/r0.010** | 76.96 | 77.99 | **77.96** | best stability (last≈peak, zero decline) |
| c3.2/f2.4/l150/r0.005 | 76.75 | 77.81 | 75.08 | |
| c3.2/f2.5/l100/r0.005 | 76.63 | 77.78 | 75.48 | |
| c2.9/f2.5/l150/r0.005 | 76.37 | 77.66 | 75.40 | ceil too low → 37% gateOFF |
| c3.2/f2.2/l150/r0.005 | 76.32 | 77.77 | 74.23 | |
| c3.4/f2.2/l150/r0.005 | 76.32 | 77.77 | 74.23 | |

Mechanism (confirmed via `gate_log.csv`): the **old baseline ceil2.9** sat *inside*
V20's adapt-time H band (2.70–3.10) → `gateOFF 34%`, gate disengaged in healthy
windows, let the R36 peak slide to 73.98. **Raising ceil to 3.2/3.4 (above the H
band)** → `gateOFF 0%`, gate always engaged → peak held. `rst 0.010 > 0.005`
strengthens the hold (+0.5 mean). **Lever = ceil above H band + higher rst.**

**ACDC (episodic 30.6, baseline 30.05, No-Adapt 23.34) — NONE BEAT EPISODIC ❌**

| config | meanAll | peak | last |
|---|---|---|---|
| c2.9/f2.2/l150/**r0.002** | **30.05** | 30.14 | 30.03 | ← best, = baseline |
| c2.4/f2.2/l150/r0.005 | 30.04 | 30.12 | 30.06 |
| c2.5/f2.2/l150/r0.005 | 30.03 | 30.13 | 30.08 |
| c2.9/f2.4/l150/r0.005 | 29.78 | 29.80 | 29.77 | floor↑ HURTS |
| c2.4/f1.9/l100/r0.003 | 29.63 | 30.13 | 29.40 | floor↓ → climbs then drifts |

All 8 clustered 29.63–30.05 (≈ baseline), peak ≤30.2. **Gate tuning cannot break
ACDC's ceiling** — two `gate_log.csv`/`sar_log.txt` facts explain why:
1. Gate is PINNED: ACDC adapt-time H ≈ 2.28 (2.22–2.50), *below any reasonable
   ceil* → `gateOFF≈0%`, `deepLag>1000 = 88–100%` → always deep-restoring. ceil
   2.4 vs 2.9 makes ~no behavioral difference (H never reaches it). User's "raise
   ceil helps a bit" does **not** hold on ACDC (H too low to ever touch ceil).
2. SAR reliable filter is INACTIVE: ACDC mean pixel entropy ≈ **1.25 ≪ e_margin 1.8**
   → **0% samples filtered**. (V20 = 1.47, 17% filtered.) So e_margin in 1.4–2.2 is a
   no-op on ACDC; it would only engage if lowered *below* ~1.25.
- Floor↑ (user's "easier source restore" idea) **hurt** on ACDC (f2.4 worst): the
  model is already over-restoring (pinned deep), more source pull just drags it
  toward the 29.7 R1 start.
- **Conclusion: ACDC's ceiling is in the base SAR+UAML loss, not the gate.** R1
  starts 29.7, peak only ~30.2; smooth-anchor is second-order here.

### N-4: Infra notes
- **OpenCV thread saturation fix** (needed for many concurrent jobs): added
  `cv2.setNumThreads(OPENCV_NUM_THREADS, default 2)` at top of `main.py` +
  `main_continual.py` (forked DataLoader workers otherwise each spawn ~256 OpenCV
  threads → exhaust `ulimit -u`=8192 → "Can't spawn new thread res=11" kills workers).
  **Uncommitted** (left out of git to avoid entangling user's concurrent edits).
- `bash/cityscapes_continual/` all 11 method scripts aligned to ACDC-matched
  (4-corr snow/frost/fog/contrast + `--subset_size 101` = 404/round), GPU_ID default
  2. Parallel driver `run_all_parallel.sh`; sequential `run_all_acdc_matched.sh`.
- Cityscapes full method comparison (404/round, 150R): sar_mlmp 20.42 (best,
  >episodic) > MLMP-DivGate 19.65 > No-Adapt 18.47; naive TENT/MLMP/SAR-continual
  collapsed (R150 1.33 / 0.16 / 9.37). CoTTA was still running at writing.

---

## 18. Phase O: grad_norm gate-mechanism campaign (2026-06-21 → 06-25)

Goal: a DeYO+MLMP CTTA defense that is both HIGH and STABLE, beating the bar
**DeYO-DivGate** (學長, 150R: ACDC **31.8**, Cityscapes **23.6**, VOC20 **77.4**).
Plain-language mechanism reference: **docs/gate_mechanisms_catalog.md** (+ `_zh.md`).
All runs: DeYO+MLMP base, NA-CLIP ViT-L/14, LN-only, seed 0, 150R. Data paths on THIS
machine: ACDC `data/ACDC/`, VOC20 `data/VOC/VOC2012/`, Cityscapes `data/Cityscape/`.

### O-1. grad_norm as a LOSS PENALTY — FAILED (negative result)
Idea (user): grad_norm ∝ −mIoU (學長 §7), so add `λ‖∇L‖²` to the loss to suppress it.
Method `adapt/deyo_mlmp_gradpen_divgate_continual.py` (λ=0 = bit-identical divgate control;
modes raw/excess/ema + grad_clip). **No usable λ**: small=flat(=baseline), large=collapse;
excess/ema rise-penalties collapse too; grad_clip doesn't save it. **Cause:** grad_norm is
high even when healthy (per-step spikes ~88) so penalising it fights healthy learning, AND
the penalty gradient direction (Hessian·∇L) is destabilising. **Lesson: grad_norm is a good
degradation DETECTOR, a bad loss TARGET → use it for restoration, not in the loss.** Spec
`docs/superpowers/specs/2026-06-21-gradnorm-penalty-design.md` §8.

### O-2. gradslope (grad_norm slope → restore to lagged snapshot) — stable but loses to divgate
`deyo_mlmp_gradslope_continual`. Our sweep (`bash/sweep_gradslope.sh`): slope_deadzone ×
base_rst, **slope_window=10 / max_lag=3000 FIXED**. Result: stable, beats episodic on all 3,
**doesn't beat divgate** (VOC20 73.6, ACDC 31.5, Cityscapes 23.4). slope_deadzone INSENSITIVE;
base_rst 0.01>0.02. **Important:** we only swept the insensitive axis — our runs ARE 學長's
"sw10" line (numbers match: VOC20 pk75.0). The SENSITIVE axis is **slope_window (10/50/100)
+ max_lag** (學長's figure: sw100/lag300 reach higher peaks but DECAY badly). Not yet swept here.

### O-3. composite (mean_conf TRIGGER + grad_norm DEPTH → restore to source) — THE WIN
`deyo_mlmp_composite_gate_continual` (`bash/sweep_composite.sh`). Two stages:
- **First sweep (conf_ceil 0.78/0.83/0.86 ACDC etc.) — gate NEVER fired (0-2%).** Bug: conf_ceil
  was set to 學長's **evaluate-scale** mean_conf peak (0.85/0.71/0.73), but the gate computes
  mean_conf on the **adapt-scale (7-prompt ensemble)** which is ~0.05-0.07 LOWER. So conf_ceil
  sat above the gate's actual mean_conf range → never triggered → "composite" = bare base.
  (Same two-scales trap as H_margin eval-vs-gate-internal, §7.5 of the contribution doc.)
- **Calibrated sweep (conf_ceil to gate-internal scale): WIN.** Gate-internal mean_conf AT the
  mIoU peak: ACDC **0.696**, VOC20 0.584, Cityscapes 0.510. Set conf_ceil ≈ that.

**Calibrated results (150R, best configs):**
| dataset | best config | mean | peak | std(R30+) | vs divgate |
|---|---|---|---|---|---|
| **ACDC** | composite **cc0.70 rst0.02** | **32.30** | 33.40 | **0.49** | **✅ +0.5, & far smoother** |
| **VOC20** | composite cc0.60 rst0.005 | **77.74** | 78.73 | 0.28 | **✅ +0.34** |
| **Cityscapes** | composite **cc0.60 rst0.005** | **23.66** | 23.90 | **0.07** | **✅ +0.06 (proper calib)** |

- **ACDC is the headline**: calibrated composite beats divgate (32.3 vs 31.8) AND cuts the
  oscillation 3× (std 1.46→0.49, floor <28→~31.5). vs gradslope 31.5/std0.65, vs episodic 29.84.
- **Mechanism**: even with low firing (~1%), the FEW restores fire AT THE PEAK (when mean_conf
  crosses conf_ceil) and **cap over-confidence** — mean_conf held at ~0.70 (calibrated) vs 0.78
  (uncalibrated) — preventing the collapse-oscillation. **Timing (conf_ceil) matters, not
  frequency.** rst0.02 > rst0.04 (gentler restore = smoother).
- **VOC20**: gate barely fires (correct — VOC20 uniform-drift, restoring HURTS; gradslope's active
  restore is why it capped at 73.6). Calibration didn't hurt it.
- **Cityscapes**: full conf_ceil curve (0.48→0.72) shows an **inverted-U with a sweet spot at
  cc0.60** (fires ~3%, just after the over-confidence onset): mean **23.66**, std **0.07** —
  beats divgate 23.6. Too low (0.48/0.52) over-fires (10-30%)→22.8; too high (0.66/0.72)
  fires too late→22.98-23.4. **Lesson: conf_ceil sweet spot is a bit ABOVE the mIoU-peak
  mean_conf (0.51), catching the over-confidence rise.** So composite beats divgate on ALL 3.

### O-4. Open next steps
1. Cityscapes: re-run composite with conf_ceil {0.52,0.55,0.58} (≥ peak 0.51) — proper calibration.
2. gradslope: sweep the SENSITIVE axis (slope_window 10/50/100 × max_lag 3000/300), like 學長,
   to see if any holds high without the late decay.
3. Same-machine DeYO-DivGate baseline (currently using 學長's numbers; user said confirmed, low priority).
4. ACDC residual sawtooth: try damped trigger (hysteresis/deadband on conf_ceil, EMA-smoothed rst).

### O-5. Tooling added this phase
- Methods: `adapt/deyo_mlmp_{gradpen_divgate,gradslope,composite_gate}_continual.py` (gradslope/
  composite from 學長's 06-18 merge; gradpen new this session).
- Launchers: `bash/sweep_gradslope.sh`, `bash/sweep_composite.sh` (per-dataset, CONCURRENT to
  pack GPU mem, env-overridable GPU/ROUNDS/STAGGER; SEQ=1 for sequential).
- Figures: `scripts/plot_session_gates.py`, `plot_perf_by_dataset.py`, `plot_calibrated_composite.py`
  → `save/_compare/{session_*,perf_*,cal_*}.png`.
- Catalog: `docs/gate_mechanisms_catalog.md` (EN) + `_zh.md` (詳細中文，逐術語).
- Spec/plan: `docs/superpowers/{specs,plans}/2026-06-21-gradnorm-penalty*`.

---

## 19. Phase S: AdaGate — GDG-PA's two inert hyperparameters made self-calibrating (2026-08-02 → 08-05)

Spec: **[docs/adagate_continual_spec.md](adagate_continual_spec.md)**. Method
`adapt/deyo_mlmp_adagate_continual.py`; sweep `bash/sweep_adagate.sh`; collector
`scripts/collect_adagate.py`. Report artifact (updated, 3 datasets):
https://claude.ai/code/artifact/526949a9-3ccd-435d-a27a-eb5833788f91

### S-1. The diagnosis — two hyperparameters that measurably do nothing

Motivating question from the user: can GDG-PA's thresholds be made *adaptive*, for a better
paper framing? Measurement on `save/*/hmgate2_prompt_S0_baseline/gate_log.csv` found that
**two of them were already inert**, so making them adaptive *removes* tunables rather than
adding any:

- **(A) `slope_deadzone = 0.002`** is meant to reject noise-level grad-norm rises, but the
  slope noise band is ±0.03 (ACDC) / ±0.35 (VOC20) — 15× / 150× larger. It degenerates to
  `slope > 0`: firing 0.463 / 0.507 vs `P(slope>0)` ≈ 0.5. It filters nothing, and it is an
  absolute constant applied to two slope distributions 10× apart in scale.
- **(B) `lag_gain = 1500`** is meant to scale restore depth continuously with degradation
  speed, but saturates the cap whenever `slope > 6/1500 = 0.004` — true in **93.2 %** (ACDC)
  / **99.7 %** (VOC20) of active windows. Realised lag is binary: ACDC `{0:322, 6:162}`,
  VOC20 `{0:741, 6:709}`. Spreading lag over 1..6 would need `lag_gain ≈ 153` (ACDC) vs
  `≈ 17` (VOC20) — a 9× gap, so **no single absolute gain can work**.

Both pathologies reproduce independently on **Cityscapes** (`ctrl` fires 0.479, `graded` 0.02).

### S-2. The fix

- **(A) `trend_stat`**: normalise the OLS slope before thresholding. `mad` = robust z-score
  `slope / (1.4826·MAD(last 50 slopes))`; also `tstat` = `slope/SE(slope)`, `rel` =
  `slope/grad_norm`, `abs` = hmgate2. `trend_thr` becomes a unitless SNR deadzone.
- **(B) `lag_mode`**: restore depth as a **fraction of the budget**, `lag = ceil(cap·u)`.
  `ecdf` sets `u` = rank of z among the last 50 *active* z — rank-based, hence invariant to
  any monotone rescaling of the trend statistic and **free of tunables** (`lag_gain` deleted).
  `sat` uses `u = clamp(z/lag_sat)`; `gain` is hmgate2.

**Equivalence verified**: `trend_stat=abs, lag_mode=gain` consumes no extra RNG and reproduced
`hmgate2_prompt_S0_baseline` to every mIoU digit over 3 ACDC rounds and on all 12 gate windows.
This is what licenses attributing every delta to A/B.

**Negative result worth reporting**: the intuitive `slope/grad_norm` normalisation is
**insufficient** — ACDC p90 0.005 vs VOC20 0.042, still 8× apart, because it normalises the
signal's *magnitude* (5.25 vs 6.64, 1.3×) not its *noise level* (11×). `mad` gives 1.125 vs
1.114 (1.01×).

### S-3. Results — 19 runs × 150R × 3 datasets

| arm | ACDC (Δ) | VOC20 (Δ) | Cityscapes (Δ) |
|---|---|---|---|
| GDG-PA / `ctrl` | 31.06 | 77.11 | 23.85 |
| `Becdf` (B only) | 31.02 (−0.04) | **77.56 (+0.45)** | — |
| `Amad0` (A only, matched firing) | 31.03 (−0.03) | — | — |
| **`ABmad05_ecdf`** ← **recommended** | **31.31 (+0.25)** | **77.80 (+0.69)** | **23.97 (+0.12)** |
| `ABmad10_ecdf` | 31.68 (+0.62) | 77.90 (+0.79) | **23.74 (−0.10)** |
| `ABmad15_ecdf` | 31.81 (+0.74) | — | — |
| `ABmad20_ecdf` | 32.12 (+1.06) | — | — |
| `ABtstat10_ecdf` | 32.07 (+1.01) | 77.84 (+0.73) | 23.91 (+0.07) |

1. **B is fixed and matters on its own in the uniform-drift regime.** `graded` goes 0.02–0.07
   → 0.60–0.83. `Becdf` (B alone, trigger untouched) earns **+0.45 on VOC20**, where the
   collapse regime never fires so 100 % of restores take the lag path.
2. **A is the ACDC lever and shows GDG-PA was over-restoring.** Firing 0.51→31.03,
   0.32→31.16, 0.16→31.62, 0.035→32.12.
3. **The two fixes are complementary, one per regime.** On ACDC, B alone does nothing
   (−0.04) and A-at-matched-firing does nothing (−0.03). Maps onto the existing
   collapse-vs-uniform-degradation split (§4/§5, contribution doc §4).
4. **Transferability is the decisive result** — firing rate under one unitless threshold:

| threshold | ACDC | VOC20 | Cityscapes | spread | verdict |
|---|---|---|---|---|---|
| `mad 0.5` | 0.292 | 0.315 | 0.299 | **1.1×** | transfers |
| `mad 1.0` | 0.167 | 0.155 | 0.105 | 1.6× | marginal |
| `tstat 1.0` | 0.037 | 0.251 | 0.296 | **8.0×** | **does not transfer** |

**`tstat`'s ACDC win was an artifact of accidentally firing at 3.7 % there**, not of the
statistic being better: once VOC20/Cityscapes push its firing to 0.25–0.30 it matches
`mad 0.5` (77.84 vs 77.80; 23.91 vs 23.97). **The statistic choice is not the lever;
cross-dataset firing-rate stability is.** Do NOT frame `tstat` as a "significance test" in
the paper — with a 10-point sliding window sharing 9/10 points, the series is heavily
autocorrelated and the nominal t-distribution does not apply; call it a residual-scaled
unitless trend strength.

### S-4. Standing recommendation and open items

**`ABmad05_ecdf`** (`trend_stat=mad, trend_thr=0.5, lag_mode=ecdf`) is the configuration to
carry forward — the only arm positive on all three datasets with stable tails (peak→last
−0.25 / −0.35 / −0.05). `mad 1.0` scores higher on ACDC/VOC20 *mean* but is **negative on
Cityscapes** and decays −1.19 from peak on VOC20 (78.89@R84 → 77.70@R150). The ACDC-only
monotone "lower firing is better" trend is **dataset-specific** — the inverted-U turns much
earlier on Cityscapes.

Open: **all runs are seed=0** (Cityscapes' +0.12 needs multi-seed); **`trend_hist=50` was
never swept**; Cityscapes threshold bracket (mad 0.3 / 0.7) would locate its earlier turn.

Honest cost accounting for the paper: 2 *scale-bound* thresholds deleted
(`slope_deadzone`, `lag_gain`), 1 *scale-free* threshold introduced (`trend_thr`) plus 2
structural sample-count parameters (`trend_hist=50`, MAD warm-up min 5) that do not need
per-dataset calibration.

---

*End of research-arc document. For any detail not covered here, follow the reading map in §13.*
## 20. Phase T: plug-and-play campaign + the `top_block_exclude` discovery (2026-09-02 → 09-04)

Paper planning session (ARS `/ars-plan`). Full running notes in the session memory
`paper_writing_2026-09.md`. Method carried forward is unchanged: **`gradnorm_scaled`** =
`deyo_mlmp_adagate_continual --shallow_cap_mode growing_scaled --base_rst 0.01
--top_block_exclude 6 --lr 5e-6` (`save/ACDCDataset/batch_ablation/gradnorm_scaled_b1/`,
ACDC mean 31.74 / R150 31.92). **Label it "Ours" in all figures.**

### T-1. Plug-and-play: the same gate on four *published* base objectives

New methods (subclasses of `DeYOMLMPAdaGateContinual`; the gate is inherited unchanged and
學長's file was not touched): `tent_adagate_continual`, `mlmp_adagate_continual`,
`delta_adagate_continual`, `sar_adagate_continual`, plus `cma_adagate_continual` (appendix
only — CMA is ours, not a published method). Runner `bash/plugplay_adagate.sh`.

Each subclass overrides `_is_excluded` so `--top_block_exclude 0` freezes **nothing**, giving
all 100 visual LN params — exactly what `tent/mlmp/delta/sar_continual` train. That makes the
**already-published run a valid no-gate control**, so only one arm per base method is needed.

ACDC, 150 rounds, batch 1, LR 1e-5, seed 0:

| base objective | no gate R150 | +AdaGate R150 | Δ | gated R1→R150 | above no-adapt 23.34? |
|---|---|---|---|---|---|
| TENT (ICLR'21) | 7.90 | 22.48 | +14.58 | −5.88 | no |
| MLMP-continual (NeurIPS'25) | 1.53 | 23.61 | **+22.08** | −6.14 | +0.3 only |
| DELTA (ICLR'23) | 12.24 | **28.59** | +16.35 | **+0.36** | yes |
| SAR (ICLR'23) | 25.31 (oscillates) | 27.79 | +2.48 | −0.60 | yes |

**The duty cycle transfers across objectives**: one unitless threshold, five objectives, firing
0.292 / 0.298 / 0.299 / 0.308 / 0.310 (DELTA / Ours / SAR / TENT / MLMP) = **1.06× spread**.
Together with the cross-dataset 0.29/0.32/0.30 of §19, `trend_thr` self-calibration now has
evidence on two independent axes.

**Scope the claim carefully.** Supported: *the same gate at the same threshold, with no
re-tuning, prevents catastrophic collapse and improves the 150-round endpoint of four published
objectives by +2.5 to +22.1 mIoU.* NOT supported: that it lifts every objective above the
no-adaptation floor (TENT 22.48 < 23.34), nor that any objective keeps improving (only DELTA
grows). Also: every published CTTA method here that actually collapses is entropy-family, so
the claim scopes to **entropy-based** objectives — CoTTA never collapses and SHOT dies at R1,
so neither can serve as a non-entropy data point.

SAR is a **replacement, not an addition**: its own hard recovery (Filter C) is disabled so two
controllers do not fight, following `sar_divgate_continual.py`.

### T-2. `top_block_exclude` — a default nobody had ablated

ViT-L/14's visual encoder has 50 LayerNorm modules (`ln_pre` + 24 blocks × {ln_1, ln_2} +
`ln_post`) = 100 params. The AdaGate family defaults to `--top_block_exclude 6`, which freezes
`ln_post` + `resblocks 18-23` → **only 74 adapt, all in the lower 18 blocks**. It entered with
the GDG-PA / smooth-anchor lineage (`7039e92`, `fce2eee`); **no published baseline has the flag
at all**. Across every `cmd.sh` under `save/`: **241 runs used `6`, and the only runs at `0`
are from this campaign.**

It is worth ~5.7 mIoU on TENT, and the mechanism is visible in the gate log:

| TENT + AdaGate | R150 | trend | H_margin drop | deep-restore | collapse-regime |
|---|---|---|---|---|---|
| 74 params (tbe=6) | **28.15** | flat | **−2.3 %** | 19.5 % | 40.6 % |
| 100 params (tbe=0) | **22.48** | −5.9 | **−55.4 %** | 30.5 % | 97.4 % |

Class-marginal collapse happens in the **top blocks**; freezing them removes the mechanism,
whereas the gate only treats the symptom. Two independent routes protect the marginal —
freeze the top blocks, or reweight by inverse class frequency (DELTA's DOT keeps H at −3.8 %
even at 100 params). Either suffices; with neither, the gate can only slow the decline.
For TENT it is a trade (ungated peak 32.90@R19 at 100 params vs 28.26@R4 at 74), but
DeYO+MLMP peaks 33.50@R34 *at 74 params*, so its headroom does not live in the top blocks.

**Ablation running** (2026-09-04): `bash/ablation_allln.sh` →
`save/ACDCDataset/ablation/adagate_allln_b1/` — Ours with all 100 params via
`adapt/deyo_mlmp_adagate_allln_continual.py` (only `_is_excluded` differs; the parent at
`tbe=0` would give 98 because it always excludes `ln_post`). This is what says how much of
31.92 is the gate and how much is the frozen top.

### T-3. Two corrections to earlier claims

- **Ours does not "keep improving"**: 29.80 → 30.98 (R10) → **31.93 (R25)** → 31.90 / 31.83 /
  31.92 (R50/100/150). Fast early adaptation, then 125 rounds of retention. The best-state
  anchor freezes early in *every* run (last refresh at window 122–585 of 1218), so the climb is
  not a ratcheting anchor — it is the base objective still having signal. How long an objective
  has signal can be read off where its **ungated** arm peaks: TENT@74p R4, DELTA@100p R12,
  TENT@100p R19, DeYO+MLMP@74p R34.
- **Grad-norm lead time is a partial negative**: on the ungated runs the sustained grad-norm
  rise precedes the mIoU degradation onset by **+18 rounds on ACDC but −30 on Cityscapes**
  (i.e. it lags). Do not claim a lead. The defensible claim is that grad_norm is a label-free
  surrogate for the unobservable mIoU (Pearson −0.955 / −0.783); note that **H_margin is the
  more consistent correlate** (+0.952 / +0.947 on both). The trigger fires from R0–R3 at ~31 %
  of windows, so the gate is a continuous controller at a calibrated duty cycle, not an alarm.

### T-4. Figures and the protocol trap

`plot_tension.py` → `figures/tension/acdc_tension.*` (methods that improve collapse; methods
that never collapse never improve). `plot_plugplay.py` → `figures/plugplay/acdc_plugplay.*`
(2×2 gate vs no-gate). `plot_tent74.py` → `figures/plugplay/acdc_tent74.*`.

**Only ACDC has the classic baselines and Ours under one protocol.** Cityscapes baselines are
`subset 101` / 4 corruptions (no-adapt 18.47) while Ours is `subset 100` / weather-5 (no-adapt
20.56); VOC20 baselines are `full` / 4 corruptions (no-adapt 71.65) while Ours is `subset 100`
/ weather-5. Never overlay across those groups.

---
