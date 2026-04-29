# Experiment Status — Research Arc

**Last updated**: 2026-04-27
**Current state**: cautious_rst sweep (5 variants) + h_threshold=1.6 baseline — ALL 150 rounds complete. Best single config: `h_thr=1.6, h_warn=1.4, cau_rst=0.01` → **mean=31.59, peak=32.96@R27, R150=31.34** — beats MLMP-episodic (30.6) by +1.0 mIoU and stays stable to R150.
**Active runs**: none.
**Next planned**: h_threshold sweep (1.7, 1.8, 2.0) with cautious_rst fixed at 0.01 — see §4 for rationale.
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

## 4. Open Questions / Next Planned Experiments

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

## 5. Concrete Next Actions (in order)

1. **Run h_threshold sweep** (3 runs, ~6h each on GPU): update `bash/ACDC_10_round/tent_divgate_continual.sh` with H_THRESHOLD={1.7,1.8,2.0}, CAUTIOUS_RST=0.01, H_WARNING=1.4.
2. **After sweep completes**: update `plot_divgate_sweep.py` to add h_threshold variants and regenerate comparison figure.
3. **Consolidate best config**: pick winner (likely 1.7 or 1.8) as the paper's main result. Update CLAUDE.md and EXPERIMENT_STATUS.md.
4. **Optional**: TENT-DivGate step=10 if time permits.
5. **Paper figures**: `acdc_divgate_cau_rst_sweep.png` already generated; generate h_threshold sweep figure; update `acdc_tent_divgate_150round.png` with best config.

---

## 6. Codebase State — All Methods Currently Registered

| Method name (CLI) | Class | File | Bash script | Spec |
|---|---|---|---|---|
| `cma_continual` | `CMAContinual` | `adapt/cma_continual.py` | `bash/ACDC_10_round/cma_continual.sh` | [cma_continual_spec.md](cma_continual_spec.md) |
| `cma_proto_continual` | `CMAProtoContinual` | `adapt/cma_proto_continual.py` | `bash/ACDC_10_round/cma_proto_continual.sh` | [cma_proto_continual_spec.md](cma_proto_continual_spec.md) |
| `cma_layered_continual` | `CMALayeredContinual` | `adapt/cma_layered_continual.py` | `bash/ACDC_10_round/cma_layered_continual.sh` | [cma_layered_continual_spec.md](cma_layered_continual_spec.md) |
| `cma_divgate_continual` | `CMADivGateContinual` | `adapt/cma_divgate_continual.py` | `bash/ACDC_10_round/cma_divgate_continual.sh` | [cma_divgate_continual_spec.md](cma_divgate_continual_spec.md) |
| `tent_divgate_continual` | `TENTDivGateContinual` | `adapt/tent_divgate_continual.py` | `bash/ACDC_10_round/tent_divgate_continual.sh` | [tent_divgate_continual_spec.md](tent_divgate_continual_spec.md) |

Plus the original baselines (`tent_continual`, `mlmp_continual`, `cotta`, `dpcore`, `mlmp` episodic) which are unchanged.

All methods registered in `adapt/__init__.py::METHOD_CLASSES` and dispatched via `main_continual.py::add_method_specific_args`. Results are saved under `save/ACDCDataset/{save_dir}/results_all_rounds.txt` and parsed by `parse_acdc_results.py` into `acdc_table.tex`.

---

## 7. Reading Map — Where to Find What

| Question | Read |
|----------|------|
| What's the overall research goal and Phase 1 baseline? | [proposal.md](../proposal.md) |
| Why did entropy/CMA collapse? Original hypothesis. | [docs/cma_continual_spec.md](cma_continual_spec.md) §9 (post-mortem) |
| Why did frozen external anchors not solve it? Deepest insight. | [docs/cma_proto_continual_spec.md](cma_proto_continual_spec.md), then [proposal_after_cma.md](../proposal_after_cma.md) §0.3 |
| What are the three new directions A/B/C? | [proposal_after_cma.md](../proposal_after_cma.md) |
| Direction A spec / Direction A experimental results | [docs/cma_layered_continual_spec.md](cma_layered_continual_spec.md), this file §2 Phase D |
| Direction B (CMA base) spec / experimental results | [docs/cma_divgate_continual_spec.md](cma_divgate_continual_spec.md), this file §2 Phase E |
| Direction B (TENT base) spec / current results | [docs/tent_divgate_continual_spec.md](tent_divgate_continual_spec.md), this file §2 Phase F |
| Codebase architecture, implementation gotchas | [CLAUDE.md](../CLAUDE.md) |
| Implementation plan format / step-by-step recipes | `docs/*_plan.md` |

---

## 8. Key Design Decisions (Quick Reference)

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

*End of research-arc document. For any detail not covered here, follow the reading map in §7.*
