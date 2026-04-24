# CMA-Continual: Cross-modal Alignment TTA — Experiment Spec

**Status**: Active development
**Created**: 2026-04-23
**Owner**: Tekai-Yen
**Related docs**: `proposal.md` (Section 4 / Direction 1), `CLAUDE.md`

---

## 1. Motivation

Existing entropy-based CTTA methods (TENT-continual, MLMP-continual) collapse catastrophically over long deployment because entropy minimization rewards confidence rather than correctness. The trivial solution is a degenerate prediction where 1–2 dominant classes (sky, road) win for every pixel — a single low-energy state with zero gradient.

**CMA hypothesis**: Replacing entropy with **cross-modal alignment loss** removes the trivial-solution problem. The frozen text embeddings `{t_c}` are geometrically diverse and fixed, so each pixel's gradient pulls its visual feature toward a *different* class direction. There is no single attractor for the system to collapse into.

This spec covers **Direction 1 only** from `proposal.md` — testing the CMA loss as a standalone replacement for entropy. Direction 2 (prototype memory bank) is deferred until the standalone CMA result is validated.

---

## 2. Loss Function

### 2.1 Definition

For each test batch, let `S_conf` be the set of pixels with prediction confidence in the top K% of the batch. The CMA loss is:

```
L_CMA = -(1 / |S_conf|) · Σ_{i ∈ S_conf} cos(v_i, t_{ĉ_i})
```

where:
- `v_i` — L2-normalized visual feature of pixel i (post-projection, in CLIP embedding space)
- `t_{ĉ_i}` — L2-normalized text embedding of the predicted class `ĉ_i = argmax_c p(c | x_i)`
- `cos(·, ·)` — cosine similarity (= dot product, since both vectors are unit-norm)

The negative sign converts the maximization objective (align visual to text) into a minimization loss for gradient descent.

### 2.2 Tensor mechanics

The model's `forward()` already returns the three tensors we need:

```
model(x, text_x, text_ensemble=True) →
  logits          : (T, B, C, w, h)   per-pixel class logits, T templates
  image_features  : (B, S, D)         L2-normalized visual features, S = w*h + 1 (CLS at index 0)
  text_features   : (T, C, D)         L2-normalized text embeddings
```

For ViT-L/14 with 224×224 input patches: `w = h = 16`, `S = 257`, `D = 768`, `C = 19`, `T = 7`.

**Computation pipeline (per gradient step)**:

```python
logits, image_features, text_features = model(x, text_x, True, interpolate=False)

# (1) Average over prompt templates
avg_logits = logits.mean(0)              # (B, C, w, h)
avg_text   = text_features.mean(0)       # (C, D)

# (2) Per-pixel pseudo-label and confidence (no_grad to avoid double-counting in graph)
with torch.no_grad():
    probs = avg_logits.softmax(1)
    confidence, pred_cls = probs.max(1)  # (B, w, h), (B, w, h)

# (3) Top-K% mask within this batch
n_total = confidence.numel()
k = max(1, int(n_total * top_k_percent))
threshold = confidence.flatten().topk(k).values[-1]
mask = confidence >= threshold           # (B, w, h)

# (4) Drop CLS, reshape patch features to spatial grid
B = image_features.shape[0]
patch_feats = image_features[:, 1:, :]   # (B, w*h, D)
vis_feat = patch_feats.reshape(B, w, h, -1)   # (B, w, h, D)

# (5) Lookup text target per pixel
text_target = avg_text[pred_cls]         # (B, w, h, D)

# (6) Cosine similarity (both normalized → dot product)
cos_sim = (vis_feat * text_target).sum(-1)    # (B, w, h)

# (7) CMA loss
loss = -cos_sim[mask].mean()
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### 2.3 Why this prevents collapse

- **Diverse attractors**: There are 19 distinct text embeddings, fixed throughout adaptation. The loss landscape has 19 separate minima — pulling a pixel toward "sky" gives no benefit to a pixel pulled toward "car".
- **No confidence reward**: Cosine similarity to a *specific target* is not maximized by overall sharpness. A pixel can have low entropy on the wrong class and still have low `cos(v_i, t_correct)`.
- **Fixed geometry**: The text encoder is frozen, so the targets never drift. LayerNorm updates that push visual features toward the wrong text region produce immediate loss penalties via mismatched pseudo-labels — but only for confident-but-wrong pixels, which Top-K filtering partially mitigates.

### 2.4 Limitations / known risks

- **Confirmation bias on pseudo-labels**: If the initial predictions for a confident pixel are wrong, CMA reinforces that wrong direction. Top-K% filter is the only protection. Direction 2 (prototype anchors) was designed to add a second safety mechanism.
- **No anti-forgetting**: LayerNorm parameters can still drift away from the source. CMA constrains direction (toward text geometry) but not magnitude (how far from source).

---

## 3. Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | `1e-5` | Same as `tent_continual.sh` — clean comparison |
| Steps per sample | `1` | Same as other continual baselines |
| Batch size | `1` | Same as other continual baselines (DPCore is the exception) |
| Top-K percent | `0.2` (20%) | Conservative starting point |
| Vision outputs | `(-1,)` | Last layer only |
| Prompt templates | 7 (from `prompts.yaml`) | Same as TENT/MLMP |
| Optimizer | Adam, β=(0.9, 0.999), wd=0 | Same as TENT |
| Updated parameters | LayerNorm (γ, β) of visual encoder only | Text encoder frozen |
| Continual rounds | 10 | Standard ACDC protocol |
| Conditions | fog → night → rain → snow | Standard ACDC order |

`--top_k_percent` is the only new CLI argument; everything else is shared with existing continual methods.

---

## 4. Code Architecture

### 4.1 New files

- **`adapt/cma_continual.py`** — `CMAContinual` class. Modeled on `tent_continual.py`. The only differences are:
  - `perform_adaptation()` calls a new `cma_loss()` helper instead of `softmax_entropy()`
  - Constructor takes `top_k_percent` argument (default `0.2`)
  - Stores `self.top_k_percent` for use in the loss
- **`bash/ACDC_10_round/cma_continual.sh`** — Bash launcher with the hyperparameters from §3

### 4.2 Modified files

- **`adapt/__init__.py`**: add `from .cma_continual import CMAContinual` and `'cma_continual': CMAContinual` in `METHOD_CLASSES`. No factory logic changes — the existing `inspect.Parameter.empty` mechanism handles `top_k_percent` automatically (it has a default value).
- **`main_continual.py`** in `add_method_specific_args()`: add an `elif method == 'cma_continual'` branch that registers `--top_k_percent` (type=float, default=0.2). No other changes to the main loop.
- **`parse_acdc_results.py`**: add `("CMA-continual", "cma_continual_step_1", False)` entry to the `METHODS` list so the next LaTeX table includes the new method.
- **`CLAUDE.md`**: add CMA-continual to the methods list and bash scripts table.

### 4.3 Non-invasive guarantees

- No changes to `ovss/clip/model.py` (uses existing `forward()` signature unchanged)
- No changes to existing `adapt/*.py` files
- No changes to existing bash scripts
- Existing `parse_acdc_results.py` continues to handle methods that have already been run; the new entry will be skipped with `[WARN] not found, skipping` until the CMA experiment runs

---

## 5. Evaluation Protocol

Identical to other continual baselines on ACDC:

- **Dataset**: ACDCDataset (`data/ACDC/`)
- **Conditions**: `fog night rain snow`, in order
- **Rounds**: 10 (model is instantiated once, never reset)
- **Patches**: `224×224` with stride `112` over `1120×560` resized images
- **Protocol**: evaluate-before-adapt (`main_continual.py` does `evaluate() → continual_adapt()` per sample)
- **Metric**: per-condition mIoU; round-level mean = average over conditions; experiment-level mean = average over 10 rounds
- **Output**: `save/ACDCDataset/cma_continual_step_1/results_all_rounds.txt` with columns `Round, fog, night, rain, snow, Mean_mIoU`

---

## 6. Success Criteria

CMA-continual is considered a **successful proof-of-concept** if:

1. **No catastrophic collapse**: Round 10 mean mIoU ≥ 25 (well above TENT-continual step=10's 5.1, ideally above No Adaptation's 23.3)
2. **Better than No Adaptation**: Mean mIoU over 10 rounds ≥ 24
3. **Stable trajectory**: No round-over-round drop greater than 5 mIoU after Round 3

**Stretch goal**: Mean mIoU over 10 rounds approaches MLMP-continual step=1 (28.9), demonstrating that CMA is at least as effective as entropy without the long-horizon collapse risk.

If criterion 1 fails, CMA alone is insufficient — proceed to Direction 2 (prototype bank) as planned. If all criteria pass, run the longer 100+ round experiment to verify the collapse is genuinely prevented (not just delayed).

---

## 7. Comparison Matrix (post-experiment)

After running, the result will be compared against existing baselines on ACDC step=1:

| Method | R1 Mean | R10 Mean | All-round Mean | Stable? |
|--------|---------|----------|----------------|---------|
| No Adaptation | 23.3 | 23.3 | 23.3 | yes |
| TENT-continual | 23.9 | 30.9 | 28.0 | no (collapses ~R80) |
| MLMP-continual | 29.8 | 25.7 | 28.9 | no (gradual decline) |
| CoTTA | 23.4 | 23.4 | 23.4 | yes (no improvement) |
| MLMP (episodic) | 30.6 | 30.6 | 29.8 | n/a (requires reset) |
| **CMA-continual** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## 8. Open Questions Deferred to Future Work

These were intentionally excluded from the v1 implementation to keep scope minimal:

1. **Adaptive K**: Should `top_k_percent` change over time or be class-balanced (top K% per predicted class)?
2. **Multi-layer features**: Use UAML-style multi-layer features for `v_i` instead of just the last layer?
3. **Confidence type**: Use entropy-based confidence (`-Σ p log p`) instead of max-prob?
4. **Combination with Direction 2**: Add prototype anchors as a second loss term (`λ₁ L_CMA + λ₂ L_proto`).

These are open research directions, not requirements for the initial implementation.

---

## 9. Post-mortem (2026-04-23): Why CMA Alone Collapsed

The 150-round experiment results (`save/ACDCDataset/cma_continual_step_1/results_all_rounds.txt`) show that CMA-continual, despite removing entropy minimization, still collapses — and collapses earlier than TENT-continual step=1:

| Phase | Round | Mean mIoU | Note |
|-------|-------|-----------|------|
| Rise | R1 → R16 | 23.38 → **26.82** | Genuine adaptation via text-visual alignment |
| Turn | R15–R17 | night drops first (24.62 → 24.10) | Hardest condition is leading indicator |
| Collapse | R17 → R40 | 26.82 → 1.29 | Rapid decline |
| Dead | R40+ | locked at 1.20 | Gradient ≈ 0 degenerate state |

**Diagnosis**: the initial assumption ("entropy form is the problem, text embedding diversity prevents trivial solutions") was only partially correct. The deeper issue is **confirmation bias from using model predictions as supervision targets**:

```
CMA feedback loop:
  High-confidence prediction of class c
  → CMA pulls v_i toward t_c
  → class c predicted even more confidently
  → more pseudo-labels for class c
  → ... loop amplifies until 1–2 dominant classes saturate every pixel
```

Text embedding diversity **delays but does not prevent** this loop. Once road/sky (the largest-area classes on ACDC) dominate, CMA actively trains the visual features to look more like road/sky text embeddings, accelerating collapse. The end state is geometrically meaningful (features clustered near road/sky text directions) but operationally degenerate (every pixel predicts one of two classes).

**Note on speed**: CMA collapses at ~R35 while TENT-continual step=1 collapses at ~R80. The larger per-step directional signal in CMA (explicit target vector per pixel) makes updates more aggressive than TENT's indirect sharpening — this helps the early climb (R1–R16 is faster than TENT's) but also accelerates the late fall.

**Conclusion**: any loss term that depends exclusively on current model predictions will have confirmation bias. Preventing collapse requires at least one signal that is **external to the current model's outputs** — a signal that survives the feedback loop. This is the motivation for Direction 2 (source prototype memory bank); see `docs/cma_proto_continual_spec.md`.
