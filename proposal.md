# Research Proposal: Stable Continual TTA for Open-Vocabulary Semantic Segmentation

## 1. Background & Problem Formulation

### 1.1 Setting

This work targets **Continual Test-Time Adaptation (CTTA)** for **Open-Vocabulary Semantic Segmentation (OVSS)** using NA-CLIP (ViT-L/14). The model adapts to a continuous, non-stationary stream of adverse-condition images (fog, night, rain, snow) without ever resetting.

**Evaluate-before-adapt protocol**: predictions are made before each update, ensuring the metric reflects real-world deployment where the model must act before it can learn.

### 1.2 Experimental Observations (ACDC, 10 Rounds)

| Method | Round 1 Mean | Round 10 Mean | Verdict |
|--------|-------------|--------------|---------|
| No Adaptation | 23.3 | 23.3 | Stable baseline |
| TENT-continual (step=1) | 23.9 | 30.9 | Steady improvement, collapses at ~R80 |
| TENT-continual (step=10) | 28.3 | 5.1 | Peaks at R4 (35.3!), collapses by R10 |
| MLMP-continual (step=1) | 29.8 | 25.7 | Gradual decline |
| MLMP-continual (step=10) | 28.7 | 1.5 | Collapses by R2 |
| CoTTA | 23.4 | 23.4 | Stable but no improvement |
| MLMP (episodic) | 30.6 | 30.6 | Best quality, requires reset |

**Core tension**: The best-performing method (MLMP episodic, 30.6 mIoU) requires per-sample reset — unrealistic in deployment. Entropy-based continual methods (TENT, MLMP) improve initially but eventually collapse catastrophically. CoTTA is stable but never improves beyond the no-adaptation baseline.

---

## 2. Root Cause Analysis: Why Entropy-based Methods Collapse

### 2.1 The Trivial Solution Problem

Entropy minimization (`L = -Σ p log p`) rewards **confidence**, not **correctness**. In the continual setting, LayerNorm parameters (γ, β) accumulate drift with each update. The sequence of failure is:

```
Phase 1 — Adaptation (R1–R20):
  Entropy minimization reduces uncertainty → genuine performance gain

Phase 2 — Gradual drift (R20–R80):
  LayerNorm drift slowly pushes visual features outside CLIP's text-compatible space

Phase 3 — Catastrophic collapse (R80–R110):
  Visual features enter a region where 1–2 dominant classes always win
  Error accumulation triggers runaway positive feedback

Phase 4 — Degenerate steady state (~7.9 mIoU):
  All pixels predicted as 1–2 classes with high confidence
  Gradient ≈ 0, model locks into trivial solution
```

### 2.2 Why OVSS Delays (but Does Not Prevent) Collapse

Unlike closed-set supervised models, NA-CLIP's **frozen text encoder** provides semantic anchors. This delays collapse by preventing the model from drifting in the full parameter space — but LayerNorm updates can still push visual features out of the text-compatible regime given sufficient steps or rounds.

### 2.3 Why Excluding These Loss Families Is Necessary

- **Entropy-based** (TENT, MLMP): minimizes prediction uncertainty → trivial solution collapse
- **Batch-wise statistics** (BN-adaptation): operates on global unconditional distribution → cannot distinguish class-specific drift; not applicable to ViT-L/14 which uses LayerNorm, not BatchNorm

---

## 3. Design Constraints

| Constraint | Value |
|-----------|-------|
| Backbone | NA-CLIP (ViT-L/14), LayerNorm only |
| Text encoder | Frozen throughout |
| Adaptation scope | Any lightweight learnable component (LayerNorm, visual prompts, adapters) |
| Source data at test time | Limited: small proxy available before deployment (e.g., first condition as source) |
| Label availability | None (unsupervised TTA) |
| Reset policy | No reset — continual setting |
| Loss families excluded | Entropy minimization, global batch statistics |

---

## 4. Proposed Research Directions

### Direction 1 — Cross-modal Alignment TTA ⭐ PRIMARY PROPOSAL

**Principle**: CLIP's frozen text encoder is a free semantic teacher. Rather than minimizing entropy, directly maximize the alignment between visual features and their corresponding text class embeddings.

**Loss function**:
```
L_CMA = -Σ_{i ∈ S_conf} cos(v_i, t_{ĉ_i})

where:
  v_i     = visual feature of pixel i (from visual encoder)
  t_{ĉ_i} = text embedding of predicted class ĉ_i (frozen)
  S_conf  = set of pixels with prediction confidence > τ
```

**Why this prevents collapse**:
- Text embeddings `{t_c}` are **fixed and geometrically diverse** (spread across the embedding space)
- Each pixel's gradient pulls its visual feature toward a *different* class direction
- There is no single "low energy" degenerate state — the loss landscape has distinct class-specific minima

**What to update**: LayerNorm (γ, β), or optionally visual prompt tokens for stronger expressivity without backbone corruption.

**Confidence filtering**: Only pixels where `max_c p(c|x_i) > τ` contribute to the loss. This prevents reinforcing wrong predictions, especially early in adaptation.

**Key novelty**: This loss is uniquely available for OVSS with CLIP. Closed-set segmentation models have no text encoder and cannot use this signal. This positions the contribution squarely in the OVSS-specific advantages space.

---

### Direction 2 — Semantic Prototype Memory Bank

**Principle**: Replace global batch statistics with **per-class, class-conditional** feature centroids (prototypes). Prototypes from the source domain serve as anti-forgetting anchors.

**Mechanism**:
1. **Initialization**: compute per-class feature centroid `μ_c` from a small source proxy set
2. **Adaptation loss**: contrastive alignment — pull current features toward same-class prototypes, push away from others
3. **Online update**: target-domain prototypes updated via exponential moving average (slow, `α ≈ 0.999`)
4. **Anti-forgetting**: source prototypes stored separately; regularization term penalizes deviation from source centroids

**Relationship to DPCore**: DPCore's coreset is a discrete, instance-level version of this idea. The prototype bank is continuous, class-level, and simpler to implement and tune.

---

### Direction 3 — Confident Pseudo-Label Self-Training

**Principle**: Replace entropy minimization with cross-entropy against filtered pseudo-labels from an EMA teacher. Cross-entropy has a "target answer" — it is directional, not just uncertainty-reducing.

```
L_PL = -Σ_{i ∈ S_conf} ŷ_i · log p_student(x_i)

where ŷ_i = soft pseudo-label from EMA teacher
```

**Why more stable than entropy**: Entropy minimization is a second-order loss (it operates on the distribution itself). Pseudo-label cross-entropy is a first-order loss with an explicit target, making optimization more predictable.

**Risk**: pseudo-label quality sets the performance ceiling. Noise accumulation remains a concern without additional regularization.

---

## 5. Recommended Approach

**Primary method: Direction 1 + Direction 2 (combined)**

The two directions are complementary:
- Direction 1 (cross-modal alignment) provides the **per-pixel adaptation signal** using text-visual geometry
- Direction 2 (prototype bank) provides **class-level memory** that prevents accumulated drift from erasing gains

Concretely:
1. Maintain per-class prototype bank initialized from source proxy
2. For each test batch, compute loss as weighted sum:
   `L = λ₁ · L_CMA + λ₂ · L_proto`
3. Update lightweight learnable components (visual prompts or LayerNorm)
4. Update target prototypes via EMA; source prototypes frozen

Direction 3 serves as an **ablation baseline** (entropy → pseudo-label cross-entropy) to isolate the benefit of the cross-modal alignment signal.

---

## 6. Positioning Against Existing Baselines

| Method | Loss Signal | Anti-forgetting | OVSS-specific |
|--------|------------|----------------|---------------|
| TENT-continual | Entropy (prediction) | None | No |
| MLMP-continual | Entropy (multi-prompt) | None | Partial |
| CoTTA | Entropy + aug-avg | Stochastic restoration | No |
| DPCore | Source-discrepancy | Coreset reuse | Partial |
| **Proposed** | **Cross-modal alignment** | **Prototype anchors** | **Yes** |

The key differentiator: **this is the only method that uses CLIP's text-visual geometry as the adaptation signal**. It turns OVSS's main architectural feature (frozen text encoder) from a passive constraint into an active anti-collapse mechanism.

---

## 7. Open Questions

1. **Confidence threshold τ**: how to set adaptively per round vs. fixed globally?
2. **Source proxy quality**: ACDC has no clean source — fog round 1 is used as proxy. How much does source quality affect prototype initialization?
3. **Learnable component choice**: LayerNorm vs. visual prompts — do they have different stability profiles with cross-modal loss?
4. **Prototype drift rate**: EMA α for target prototypes — too slow = no adaptation, too fast = forgetting.
