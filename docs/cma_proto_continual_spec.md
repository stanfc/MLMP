# CMA-Proto-Continual: Cross-modal Alignment + Prototype Memory Bank

**Status**: Active development
**Created**: 2026-04-23
**Owner**: Tekai-Yen
**Related docs**: `proposal.md` (Direction 1 + Direction 2), `docs/cma_continual_spec.md` (baseline)

---

## 1. Motivation: Why the CMA Baseline Was Not Enough

The CMA-continual baseline (see `docs/cma_continual_spec.md`) replaced entropy minimization with cross-modal alignment loss. The experiment (150 rounds on ACDC) produced:

| Phase | Round | Mean mIoU | Behavior |
|-------|-------|-----------|----------|
| Rise | R1 → R16 | 23.38 → **26.82** | Genuine adaptation |
| Turn | R15–R17 | Night collapses first | Hardest condition fails first |
| Collapse | R17 → R40 | 26.82 → 1.29 | Rapid decline |
| Dead | R40+ | locked at 1.20 | gradient ≈ 0, degenerate state |

**Diagnosis — the confirmation bias loop is unbroken**:

```
CMA loop:
  High-confidence pred for class c → CMA pulls v_i toward t_c
  → class c predicted more often → more pseudo-labels for class c
  → more features pulled toward t_c → dominant classes (road/sky) take over
  → collapse
```

Text embedding diversity delayed collapse but did not prevent it: once one class becomes dominant, CMA reinforces that class's text direction just as efficiently as entropy minimization reinforces raw confidence. The geometry of the degenerate state is different, but the trajectory is the same.

**Root cause**: every term in the CMA loss depends on the model's own predictions. There is no signal that survives the confirmation feedback loop.

---

## 2. Solution Principle: An External Anchor

We need at least one supervision signal that is **completely independent of test-time predictions** — one that does not participate in confirmation bias. Source-domain per-class prototypes provide exactly this: they are computed **once**, **before** the continual stream begins, and **never updated** by test-time pseudo-labels.

Design philosophy: the total loss is a weighted sum of three cosine-alignment signals:

1. **Text anchor** (`L_CMA`): keep the CMA signal — pulls features toward CLIP language-space class direction.
2. **Source visual anchor** (`L_src`): **fixed**, domain-aware anchor. Anti-forgetting mechanism. Cannot drift.
3. **Target visual anchor** (`L_tgt`): EMA-updated prototype, slow adaptation to the test stream. Can be disabled (`λ_tgt = 0`) to isolate the source-only variant.

The source anchor is the component that solves the confirmation bias problem. The text and target anchors provide the adaptation signal.

---

## 3. Loss Function

### 3.1 Definition

For each pixel `i` in the top-K% confidence mask `S_conf`, let `ĉ_i = argmax_c p̄(c | x_i)` be the pseudo-label from the prompt-averaged logits. Three cosine alignment terms share this mask:

```
L_CMA(i) = -cos(v_i, t_{ĉ_i})            # Text embedding (frozen)
L_src(i) = -cos(v_i, p^src_{ĉ_i})         # Source visual prototype (frozen)
L_tgt(i) = -cos(v_i, p^tgt_{ĉ_i})         # Target visual prototype (EMA)

L_total  = (1/|S_conf|) · Σ [λ_CMA·L_CMA(i) + λ_src·L_src(i) + λ_tgt·L_tgt(i)]
```

All vectors are L2-normalized before dot-product, so cosine = dot-product.

### 3.2 What each term does

| Term | Target | Updated during stream? | Purpose |
|------|--------|------------------------|---------|
| `t_c` | CLIP text embedding for class `c` | No (text encoder frozen) | Language-space semantic direction |
| `p^src_c` | Source-domain visual prototype for class `c` | **No** (frozen forever) | Anti-confirmation-bias anchor |
| `p^tgt_c` | Target-domain EMA prototype for class `c` | Yes (slow EMA, momentum `α ≈ 0.999`) | Domain-specific adaptation signal |

### 3.3 Why text and source prototypes are both needed

The text embedding `t_c` and the source visual prototype `p^src_c` are NOT the same vector even though both claim to represent class `c`:

- `t_c` is where CLIP's language encoder places "a photo of a car".
- `p^src_c` is where actual fog-domain car-pixel visual features cluster after the visual encoder.

These differ by the well-known visual–text gap in CLIP plus domain-specific distribution. Using both gives the optimizer two complementary constraints: the text anchor is domain-invariant but imprecise; the source visual anchor is domain-specific but potentially biased by pseudo-label noise.

### 3.4 Degeneration to simpler variants

The loss is designed so that hyperparameters can disable components for ablations:

- `λ_tgt = 0` → source anchor only (Option A from design discussion: pure hard anchor)
- `λ_src = 0, λ_tgt = 0` → original `cma_continual` baseline
- `λ_src = 0, λ_CMA = 0` → pure target prototype loss
- `λ_src = 1, λ_CMA = 0, λ_tgt = 0` → pure source prototype loss

Running the full `λ_CMA = 1, λ_src = 1, λ_tgt = 0.5` configuration is the primary experiment.

---

## 4. Source Prototype Initialization

### 4.1 Source proxy data

Following the convention used by DPCore on ACDC, the first condition in the stream (fog) serves as the source proxy. Labels are NOT used — this is Option C from design discussion (pseudo-label init, realistic CTTA setting).

The source images for initialization are loaded via the same `prepare_data()` path as the main loop. Users can override via `--src_dataset / --src_data_dir / --src_corruption`; defaults are `conditions[0]` of the main dataset.

### 4.2 Bias-mitigating filter stack

Pseudo-labels are noisy, so initialization applies a conservative multi-stage filter. A pixel's feature contributes to `p^src_c` only if it passes **all three** filters:

1. **Confidence filter**: `max_c p̄(c | x_i) ≥ src_conf_threshold` (default `0.5`). Drops low-confidence pixels whose pseudo-label is unreliable.
2. **Cross-prompt agreement**: all 7 prompt templates must independently predict the same class for pixel `i`. Drops pixels where prompts disagree (a strong noise signal).
3. **Class-predicted consistency**: pixel is assigned to class `c` = argmax of prompt-averaged logits. Ensures the class assignment is the "main" one across prompts, not an artifact of one template.

The combined filter is conservative by design: fewer pixels contribute, but each is more likely correct.

### 4.3 Class fallback for zero-count classes

Some rare classes (e.g., "rider", "train" in fog) may have zero pixels passing all three filters. For these classes, the prototype falls back to the class's text embedding:

```
if count[c] == 0:
    p^src_c = t_c         # normalized
else:
    p^src_c = mean_{i: class_mask_c} v_i ; then L2-normalize
```

This ensures every class has a valid prototype while explicitly recording fallback counts in the log. Fallback classes are weaker anchors (they become redundant with `L_CMA`), but this is acceptable in a zero-shot regime.

### 4.4 Initialization pseudocode

```python
def obtain_src_prototypes(self, src_loader, max_samples=5000):
    feat_sum   = zeros(C, D)   # per-class feature accumulator
    feat_count = zeros(C)      # per-class pixel count
    seen       = 0

    with torch.no_grad():
        for batch in src_loader:
            images = batch['img_patches'].to(device)
            logits, image_features, text_features = model(images, text_x, True, interpolate=False)
            # logits:         (T, B, C, w, h)
            # image_features: (B, S, D), S = w*h + 1 (CLS at index 0)
            # text_features:  (T, C, D)

            # Per-prompt argmax for cross-prompt agreement
            per_prompt_pred = logits.argmax(dim=2)            # (T, B, w, h)
            agreement_mask  = (per_prompt_pred == per_prompt_pred[0:1]).all(0)

            # Prompt-averaged prediction and confidence
            avg_logits = logits.mean(0)                       # (B, C, w, h)
            probs      = avg_logits.softmax(1)
            confidence, pred_cls = probs.max(1)

            # Combined filter
            mask = (confidence >= src_conf_threshold) & agreement_mask

            # Patch features (drop CLS, reshape to B,w,h,D)
            patch = image_features[:, 1:, :].reshape(B, w, h, D)

            # Accumulate per class
            for c in range(C):
                mc = mask & (pred_cls == c)
                if mc.any():
                    feat_sum[c]   += patch[mc].sum(0)
                    feat_count[c] += mc.sum()

            seen += B
            if seen >= max_samples:
                break

    # Normalize + fallback
    avg_text = text_features.mean(0) / text_features.mean(0).norm(-1, keepdim=True).clamp(min=1e-8)
    fallback = 0
    p_src = zeros(C, D)
    for c in range(C):
        if feat_count[c] > 0:
            mu = feat_sum[c] / feat_count[c]
            p_src[c] = mu / mu.norm().clamp(min=1e-8)
        else:
            p_src[c] = avg_text[c]
            fallback += 1

    self.p_src = p_src                  # fixed
    self.p_tgt = p_src.clone()          # target init = source
    print(f'[CMA-Proto] prototypes from {seen} images; fallback classes: {fallback}/{C}')
```

---

## 5. Target Prototype EMA Update

During the stream, target prototypes are updated using only confident, in-mask pixels:

```python
def _update_target_prototypes(self, vis_feat, pred_cls, mask):
    for c in range(C):
        mc = mask & (pred_cls == c)
        if mc.any():
            mu_c = vis_feat[mc].mean(0)
            mu_c = mu_c / mu_c.norm().clamp(min=1e-8)
            self.p_tgt[c] = alpha * self.p_tgt[c] + (1 - alpha) * mu_c
            self.p_tgt[c] = self.p_tgt[c] / self.p_tgt[c].norm().clamp(min=1e-8)
```

Update is done **under `no_grad`** — EMA is not a differentiable path. Classes absent in the batch are not updated that step.

**Known risk**: target EMA is the one component that does participate in the confirmation bias loop. If the model starts over-predicting one class, the target prototype for that class will slowly drift toward a larger-volume feature cluster. The source anchor is the counterweight — as long as `λ_src ≥ λ_tgt`, the confirmation pressure from the EMA is outweighed by the fixed source pull. This is an empirical claim to validate.

---

## 6. Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `lr` | `1e-5` | LayerNorm optimizer LR (same as TENT/CMA) |
| `steps` | `1` | Gradient steps per sample |
| `batch_size` | `1` | Same as other continual baselines |
| `top_k_percent` | `0.2` | Top-K% confidence mask for all three loss terms |
| `lambda_cma` | `1.0` | Weight of text-alignment term |
| `lambda_src` | `1.0` | Weight of source prototype term (anti-collapse anchor) |
| `lambda_tgt` | `0.5` | Weight of target EMA prototype term |
| `ema_alpha` | `0.999` | Target prototype EMA momentum |
| `src_conf_threshold` | `0.5` | Prototype init confidence filter |
| `src_max_samples` | `5000` | Max source images used for prototype init |
| `continual_rounds` | `150` | Extended protocol (match CMA baseline run) |

`λ_tgt = 0` degrades the method to the source-only hard-anchor variant (Option A).

---

## 7. Code Architecture

### 7.1 New files

- **`adapt/cma_proto_continual.py`** — `CMAProtoContinual` class. Inherits conceptually from `cma_continual.py` (same LN updates, same Top-K mask, same prompt averaging) but adds:
  - `obtain_src_prototypes(data_loader)` — initialization before stream
  - `_update_target_prototypes(vis_feat, pred_cls, mask)` — called inside loss fn
  - `cma_proto_loss(logits, image_features, text_features)` — three-term loss
  - Prototype storage buffers `self.p_src`, `self.p_tgt`

- **`bash/ACDC_10_round/cma_proto_continual.sh`** — launcher with defaults from §6

### 7.2 Modified files (minimal)

- **`adapt/__init__.py`**: add `from .cma_proto_continual import CMAProtoContinual` and `'cma_proto_continual': CMAProtoContinual` to `METHOD_CLASSES`.
- **`main_continual.py`**:
  - In `add_method_specific_args()`, add a branch for `cma_proto_continual` registering `--top_k_percent`, `--lambda_cma`, `--lambda_src`, `--lambda_tgt`, `--ema_alpha`, `--src_conf_threshold`, `--src_max_samples`, `--src_corruption`.
  - In `main()`, extend the existing DPCore source-loading block to also trigger for `cma_proto_continual`, calling `obtain_src_prototypes()` instead of `obtain_src_stat()`. Uses `args.src_corruption` if set, else `conditions[0]`.
- **`parse_acdc_results.py`**: add `("CMA-Proto-continual", "cma_proto_continual_step_1", False)` to `METHODS`.
- **`CLAUDE.md`**: add method entry + bash script table row.

No existing method files are modified. Existing `cma_continual` is preserved as a baseline.

---

## 8. Evaluation Protocol

Identical to CMA-continual (150 rounds, 4 conditions, evaluate-before-adapt).

Output: `save/ACDCDataset/cma_proto_continual_step_1/results_all_rounds.txt`

Additional logs written at the top of the run:
- Number of source images processed for prototype init
- Number of classes that fell back to text embedding
- Top-3 classes by pixel count in prototype init (sanity check for road/sky dominance)

---

## 9. Success Criteria

CMA-Proto-continual is considered **successful** if:

1. **No collapse within 150 rounds**: Round 150 mean mIoU ≥ 15 (much better than CMA-continual's 1.20 at R150).
2. **Monotonic or near-monotonic improvement for ≥ 50 rounds**: Round 50 mean mIoU ≥ Round 10 mean mIoU − 2, AND Round 50 > Round 1.
3. **Matches or beats CMA peak**: best mean mIoU over 150 rounds ≥ 26.82 (CMA's peak at R16).

**Stretch**: Continues improving past R50, approaches or exceeds MLMP-episodic (30.6). This would be evidence that the combined method achieves both stability AND quality.

**Fallback interpretation**: If criterion 1 holds but criterion 3 fails (stable but low performance), the source anchor is TOO strong — reduce `λ_src` and/or increase `λ_tgt`. This tuning is a cheap iteration.

If criterion 1 fails (collapse still happens), the source anchor's geometry is insufficient. Next steps would be: stronger filters (tri-view consistency on augmentations), hybrid prototype = α·v_centroid + (1−α)·t_c, or weight-level anti-forgetting (stochastic restoration).

---

## 10. Open Questions (Deferred)

1. **Should the prototype be multi-level (UAML-style)?** Current design uses last-layer features only.
2. **Should EMA momentum `α` be class-aware?** Rare classes may need faster updates since they see fewer pixels.
3. **Should the Top-K mask be class-balanced?** Currently global top-K; could over-represent dominant classes.
4. **Should `p^src` adapt very slowly (e.g., `α_src = 0.9999`)?** Current design keeps it completely frozen, which is principled but might be overly conservative.
5. **Augmentation-based prototype refinement**: one-time TTA-style refine prototypes using augmented views of source images.

These are follow-up experiments, not part of v1.
