# Experiment Status — Session Handoff

**Last updated**: 2026-04-24
**Current state**: `cma_proto_continual` implemented and sanity-tested; NOT yet run on full ACDC stream.
**Next action**: `bash bash/ACDC_10_round/cma_proto_continual.sh`

This document is the entry point for anyone (including a new Claude session) picking up this work. Read this first, then drill into the referenced specs for detail.

---

## 1. The Big Picture

**Project**: MLMP — Test-Time Adaptation (TTA) framework for Open-Vocabulary Semantic Segmentation on NA-CLIP (ViT-L/14).

**Phase 1 (complete)**: Benchmark existing CTTA methods on ACDC (fog/night/rain/snow × 10 rounds, evaluate-before-adapt).

**Phase 2 (current — this work)**: Design a new Continual TTA method that is both **stable (CoTTA-level)** and **high-quality (MLMP-episodic-level)**. Existing methods fail one or the other:

| Baseline | R10 mean | Verdict |
|----------|----------|---------|
| No Adaptation | 23.3 | stable but no adaptation |
| TENT-continual (step=1) | 30.9 | improves early, collapses ~R80 |
| MLMP-continual (step=1) | 25.7 | gradual decline |
| CoTTA | 23.4 | stable but never improves |
| **MLMP episodic** | **30.6** | **best quality, requires per-sample reset** (unrealistic) |

The core tension: **every CTTA method that improves eventually collapses; every method that is stable never improves**.

---

## 2. The Research Arc — What We've Learned

### 2.1 Starting hypothesis (from `proposal.md`)

> Entropy-based losses (TENT, MLMP) collapse because `L = -Σ p log p` rewards *confidence* not *correctness*. Over time, LayerNorm drift accumulates and the model locks into predicting 1–2 dominant classes (road, sky) with high confidence for every pixel. The key structural feature of OVSS — the frozen text encoder — provides 19 diverse fixed targets in embedding space. Using those as the loss signal (cross-modal alignment) should break the trivial solution.

### 2.2 First attempt: CMA-continual (Direction 1) — **FAILED**

**Implementation**: `adapt/cma_continual.py`, spec in `docs/cma_continual_spec.md`.

**Loss**: `L_CMA = -mean_{i in top-K%} cos(v_i, t_{ĉ_i})` — pull each confident pixel's visual feature toward the text embedding of its predicted class.

**Result** (150 rounds on ACDC, `save/ACDCDataset/cma_continual_step_1/`):

| Phase | Rounds | Mean mIoU | Behavior |
|-------|--------|-----------|----------|
| Rise | R1 → R16 | 23.38 → **26.82** (+3.44) | Real adaptation |
| Turn | R15–17 | night drops first (hardest condition) | Leading indicator |
| Collapse | R17 → R40 | 26.82 → 1.29 | Rapid decline |
| Dead | R40+ | locked at 1.20 | Gradient ≈ 0, all pixels predict 1–2 classes |

**Diagnosis — the hypothesis was wrong at the deepest level**:

The problem is NOT the specific form of entropy loss. It's that **any loss whose target depends on the current model's predictions creates a confirmation bias feedback loop**. The text-embedding diversity only controls *which directions* the collapse occurs in, not *whether* it occurs:

```
CMA confirmation loop:
  High-confidence prediction of class c
  → CMA pulls v_i toward t_c
  → class c gets even more confidently predicted
  → more pseudo-labels for class c
  → ... loops until road/sky dominate every pixel
```

CMA collapsed at R35 vs TENT-continual step=1's R80 — meaning CMA was actually *faster* to collapse, because its per-step directional signal (explicit target vector per pixel) is more aggressive than TENT's indirect confidence sharpening.

**Full post-mortem**: `docs/cma_continual_spec.md` §9.

### 2.3 Refined hypothesis

> **To prevent collapse, the total loss must include at least one term whose target is completely independent of the current model's predictions** — an external anchor that does not participate in the feedback loop.

### 2.4 Current attempt: CMA-Proto-continual (Direction 1 + Direction 2) — **IMPLEMENTED, NOT YET RUN**

**Implementation**: `adapt/cma_proto_continual.py`, spec in `docs/cma_proto_continual_spec.md`, bash script `bash/ACDC_10_round/cma_proto_continual.sh`.

**Loss** — three cosine-alignment terms sharing the same Top-K% confident-pixel mask:

```
L_total = λ_CMA · L_CMA(t_c)        # text anchor (from CMA)
        + λ_src · L_src(p_src_c)     # fixed source visual prototype — THE EXTERNAL ANCHOR
        + λ_tgt · L_tgt(p_tgt_c)     # EMA target visual prototype
```

**The key design invariant**: `p_src` is computed ONCE before the stream begins, and NEVER updated during adaptation. This is the term that breaks confirmation bias — it is immune to the feedback loop because its target does not change based on what the model predicts.

**Source prototype initialization** (Option C from design — most realistic CTTA):
- Use ACDC fog as source proxy (no ground truth labels, following DPCore convention)
- Filter pseudo-labels: confidence ≥ 0.5 AND all 7 prompt templates agree
- Per-class visual centroid → L2-normalized → fixed forever
- Classes with zero confident pixels fall back to text embedding `t_c`

**Hyperparameter default**: `λ_CMA=1.0, λ_src=1.0, λ_tgt=0.5, ema_alpha=0.999, top_k=0.2`.
Degradation: `λ_tgt=0` → pure source-anchor variant.

---

## 3. Current Codebase State

### New/modified files from this work

| File | Status | What it is |
|------|--------|------------|
| `adapt/cma_continual.py` | done, experiment run | CMA baseline (Direction 1) |
| `adapt/cma_proto_continual.py` | done, sanity-tested | CMA + prototype bank (D1+D2), ready to run |
| `bash/ACDC_10_round/cma_continual.sh` | done, run with 150 rounds | baseline runner (user tweaked `TOP_K_PERCENT=0.5`) |
| `bash/ACDC_10_round/cma_proto_continual.sh` | done | D1+D2 runner (150 rounds, fog as source proxy) |
| `docs/cma_continual_spec.md` | done (with post-mortem §9) | CMA baseline spec + failure analysis |
| `docs/cma_proto_continual_spec.md` | done | D1+D2 full spec |
| `docs/EXPERIMENT_STATUS.md` | **this file** | session-handoff snapshot |
| `proposal.md` | unchanged | original research proposal |
| `adapt/__init__.py` | updated | registers both new methods |
| `main_continual.py` | updated | method-specific args, source-proxy loader |
| `parse_acdc_results.py` | updated | parses both new method outputs |
| `CLAUDE.md` | updated | project-level documentation |

### Sanity-test results for `cma_proto_continual` (passed)

- Prototype init produces unit-norm per-class prototypes (with text-embedding fallback for classes with no confident pixels)
- Adapt step: LayerNorm updates, `p_src` delta = 0 (frozen anchor invariant holds), `p_tgt` EMA drift ~1e-4 (matches `1−α=0.001`)
- `evaluate()` returns correct shape
- `λ_tgt=0` degradation path works (pure source-anchor variant)
- Dtype-safe: fp32 accumulators for numerically stable prototype summation, final storage in fp16 to match model

### Save dir convention

- Baseline CMA result: `save/ACDCDataset/cma_continual_step_1/results_all_rounds.txt`
- CMA-Proto output (when run): `save/ACDCDataset/cma_proto_continual_step_1/results_all_rounds.txt`

`parse_acdc_results.py` picks up both and generates the combined LaTeX table.

---

## 4. Success Criteria for CMA-Proto-continual

From `docs/cma_proto_continual_spec.md` §9:

1. **No collapse within 150 rounds**: R150 mean mIoU ≥ 15 (vs CMA's 1.20 at R40+).
2. **Stable improvement for ≥50 rounds**: R50 ≥ R10 − 2, AND R50 > R1.
3. **Matches/beats CMA peak**: best mean over 150 rounds ≥ 26.82.

**Stretch**: continues improving past R50, approaches MLMP-episodic (30.6).

### Interpretation paths

- **If criterion 1 holds but 3 fails (stable-low)**: source anchor is too strong → reduce `λ_src`, increase `λ_tgt`. Cheap tuning iteration.
- **If criterion 1 fails (still collapses)**: source anchor is not enough. Next steps: stronger filters (tri-view augmentation consistency), hybrid `p_src = α·v_centroid + (1−α)·t_c`, or weight-level anti-forgetting (stochastic restoration à la CoTTA).

---

## 5. Concrete Next Actions

1. **Run the experiment**:
   ```bash
   bash bash/ACDC_10_round/cma_proto_continual.sh
   ```
   This runs 150 rounds (~hours on a single GPU). Watch for:
   - Pre-stream log: `[CMA-Proto] Source images used: X`, `Top-3 classes by confident-pixel count`, `Fallback classes` — sanity check that prototype init is reasonable (expect road/sky/building to dominate, rider/train likely fall back to text)
   - Round-by-round `results_all_rounds.txt` — check if collapse is avoided

2. **Parse results into LaTeX table**:
   ```bash
   python parse_acdc_results.py
   ```

3. **If successful (criteria 1+2+3 met)**: write up findings, compare to CMA baseline trajectory round-by-round, consider running longer (300+ rounds) to confirm no eventual collapse.

4. **If partial (criterion 1 only)**: sweep hyperparameters `λ_src`, `λ_tgt`, `ema_alpha`. Easiest next is `λ_tgt=0` (pure hard anchor) — flip one number and re-run.

5. **If failed (still collapses)**: revisit spec §9 fallback options. Priority order: (a) stronger prototype filter, (b) hybrid text-visual prototype, (c) stochastic restoration on LayerNorm.

---

## 6. Reading Map — Where to Find What

| Question | Read |
|----------|------|
| What's the overall research goal and high-level proposal? | `proposal.md` |
| Why did CMA (alone) collapse? | `docs/cma_continual_spec.md` §9 (post-mortem) |
| What are the exact tensor ops in the CMA loss? | `docs/cma_continual_spec.md` §2 |
| Full design of the current method (CMA-Proto)? | `docs/cma_proto_continual_spec.md` |
| Prototype init filter stack, fallback behavior? | `docs/cma_proto_continual_spec.md` §4 |
| All hyperparameters and what they control? | `docs/cma_proto_continual_spec.md` §6 |
| Project-level codebase map / how pieces fit? | `CLAUDE.md` |
| Implementation details and gotchas (CLS token, fp16 dtype)? | `CLAUDE.md` §"Critical Implementation Details" |
| Current session state, what to do next? | **this file** |

---

## 7. Key Design Decisions (Quick Reference)

Made during brainstorming, recorded here for continuity:

1. **Update target = LayerNorm (γ, β)**: same as TENT/MLMP baselines, makes any difference attributable purely to the loss signal.
2. **Confidence filter = Top-K% per batch** (not fixed threshold): robust to overall confidence shifts across rounds.
3. **Start with Direction 1 alone, then add Direction 2**: clean attribution of stability gains to the prototype bank specifically. Direction 1 was tested first (failed), motivating Direction 2.
4. **Source prototype init uses pseudo-labels, not ground truth** (Option C): matches realistic CTTA setting. Bias is mitigated via cross-prompt-agreement filter + text-embedding fallback.
5. **Source prototype is completely frozen during stream** (not source-EMA hybrid): the whole point is an external anchor. Any update path participates in confirmation bias.
6. **Option B (source fixed + target EMA) chosen over Option A (source only)**: allows stronger adaptation signal; `λ_tgt=0` degrades to A if needed. The design keeps the safety of A as a tunable fallback.

---

*End of session handoff. For any detail not covered here, follow the reading map in §6.*
