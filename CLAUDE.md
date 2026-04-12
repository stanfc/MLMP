# MLMP Codebase — Project Guide for Claude

## Research Context

This repo is the official implementation of **MLMP** (Multi-Level Multi-Prompt), a TTA framework for Open-Vocabulary Semantic Segmentation (OVSS), published at NeurIPS 2025 (arXiv:2505.21844).

**Current research goal**: Convert the standard (episodic) TTA setup into **Continual TTA (CTTA)**, following the CoTTA paper (Wang et al., arXiv:2203.13591, CVPR 2022). The key difference is that in CTTA the model state **persists across all test samples** — it is never reset — which forces the method to handle error accumulation and catastrophic forgetting.

---

## Papers Summary

### MLMP (the repo's method)

**Task**: Test-time adaptation of CLIP-based OVSS models (no source data, no labels at test time).

**Model**: NA-CLIP with ViT-L/14 backbone. Only the **LayerNorm (γ, β) parameters of the visual encoder** are updated. Text encoder is completely frozen.

**Three core ideas**:
1. **Uncertainty-Aware Multi-Level Fusion (UAML)**: Instead of using only the last layer of ViT, features from multiple intermediate layers `l ∈ {-1, -2, ..., -18}` are extracted. Each layer `l` gets a confidence weight `α^l ∝ exp(-β · h^l)` where `h^l` is the batch-wise entropy of that layer's predictions. Lower entropy → higher weight. `β=0` during adaptation (uniform), `β=1` during evaluation (sharpened).
2. **Multi-Prompt Adaptation (loss-level)**: 7 prompt templates (loaded from `prompts.yaml`). Instead of averaging text embeddings (like WATT/text-level), each template produces its own loss and they are averaged: `L_final = (1/T) Σ_t (L_UAML(T_t) + L_ILE(T_t))`. This reduces gradient variance by 1/T.
3. **Image-Level Entropy (ILE)**: An additional entropy term using the CLS token (global feature), weighted by `alpha_cls` (default 1.0). Complements the per-pixel spatial token entropy.

**Loss**: Sum of pixel-level softmax entropy over all prompt templates + `alpha_cls` × CLS entropy.

**Standard TTA (episodic) — current behavior**:
- For every new batch: `reset()` → `adapt()` → `evaluate()`
- `reset()` restores model and optimizer to the **initial pretrained state** (saved at `__init__`)
- This means adaptation is **per-sample/per-batch only** — no knowledge is carried forward

---

### CoTTA (the CTTA reference)

**Task**: Continual TTA — target domain shifts continuously over time. Data arrives as an online stream.

**Two problems CoTTA solves**:
1. **Error accumulation**: Pseudo-labels get progressively noisier → wrong gradient signal
2. **Catastrophic forgetting**: Model drifts away from source knowledge → can't recover

**CoTTA's three mechanisms** (all must be implemented for CTTA):

#### 1. Weight-Averaged Pseudo-Labels (Teacher-Student EMA)
- Maintain a **teacher model** `f_{θ_t}` updated by EMA of student weights
- EMA update: `θ_{t+1} = α·θ_t + (1-α)·θ_{t+1}` (α ≈ 0.999)
- Teacher initialized to source model weights at t=0
- Loss: cross-entropy of student predictions against teacher soft pseudo-labels
- Key insight: teacher encodes history → less susceptible to single-step noise

#### 2. Augmentation-Averaged Pseudo-Labels (Confidence Gating)
- Let `p_th` be a confidence threshold (e.g. 0.1)
- If `conf(f_{θ_0}(x_t)) ≥ p_th` (source model is confident): use teacher prediction directly
- Else (domain gap is large): apply N random augmentations, average teacher predictions → more robust pseudo-label
- `ỹ_t = (1/N) Σ_{n=0}^{N-1} f_{θ_t}(aug_n(x_t))`
- Purpose: Apply augmentations only when truly needed (domain gap is large)

#### 3. Stochastic Restoration (Anti-Forgetting)
- After each optimizer step, stochastically reset a small fraction of weights back to **source weights** `W_0`
- Restoration probability `p = 0.01` (very small)
- Per-parameter: `M ~ Bernoulli(p)`, then `W_{t+1} = M ⊙ W_0 + (1-M) ⊙ W_{t+1}`
- Applied to all trainable conv/LN layers each step
- This is equivalent to a random sparse Dropout toward the source model
- Prevents catastrophic forgetting without requiring source data access

**CoTTA adaptation loop (per time step t)**:
```
1. Compute weight-averaged + augmentation-averaged pseudo-label ỹ_t from teacher
2. Update student f_{θ_t} → f_{θ_{t+1}} via consistency loss against ỹ_t (Eq. 5)
3. Update teacher: θ_{t+1} = α·θ_t + (1-α)·θ_{t+1}  (Eq. 2)
4. Stochastic restore: W_{t+1} = M⊙W_0 + (1-M)⊙W_{t+1}  (Eq. 8)
```

---

## Codebase Architecture

```
MLMP/
├── main.py                     # Entry point — argument parsing, data loop, metrics
├── adapt/
│   ├── __init__.py             # get_method() factory — maps method name → class
│   ├── mlmp.py                 # MLMP class (the proposed method)
│   ├── tent.py                 # TENT baseline
│   ├── tpt.py                  # TPT baseline (learnable prompt tokens)
│   ├── clipartt.py             # CLIPArTT baseline
│   └── watt.py                 # WATT baseline
├── ovss/
│   └── clip/
│       ├── model.py            # Modified CLIP — supports multi-layer visual output & weighted fusion
│       └── clip.py             # CLIP loading + tokenizer
├── utils/
│   ├── segmentation_datasets.py  # All dataset classes (CityscapesDataset etc.)
│   ├── metrics.py              # intersect_and_union, process_metrics → mIoU/mDice/mAcc
│   ├── misc.py                 # set_global_seeds, aggregate_pred_patches, load_prompts_from_yaml
│   └── mm_transforms.py        # Corruption augmentation transforms
├── bash/
│   ├── cityscapes/             # {mlmp,tent,tpt,clipartt,watt,no_adapt}.sh
│   ├── coco_obj/               # same set of scripts
│   ├── coco_stuff/             # same set of scripts
│   ├── p59/                    # Pascal Context 59
│   ├── p60/                    # Pascal Context 60
│   ├── v20/                    # Pascal VOC 20
│   └── v21/                    # Pascal VOC 21
└── prompts.yaml                # 7 text prompt templates used by MLMP/WATT
```

---

## Key Code Locations — Critical Logic

### `main.py` — The Training/Evaluation Loop

**The episodic reset** (the line that must change for CTTA):
```python
# main.py:329
adapt_method.reset()   # ← REMOVED in CTTA; model persists across batches
```

**The per-corruption model re-initialization** (also must change for CTTA):
```python
# main.py:296-301
if c_idx > 0:
    del adapt_method          # ← REMOVED in CTTA
    torch.cuda.empty_cache()  # keep but reorganize
adapt_method = get_method(args, device)  # ← Called ONCE before the corruption loop in CTTA
```

**The outer loop** runs over `args.corruptions_list`. In CTTA, all corruptions form a **single continual stream** — the model sees `original → gaussian_noise → shot_noise → ...` without resetting.

**Evaluation** is always done at `adapt_method.evaluate(inputs)` — this does NOT update gradients (`@torch.no_grad()`).

### `adapt/mlmp.py` — MLMP Class

| Method | Role |
|--------|------|
| `__init__` | Loads OVSS model, freezes text encoder, sets LN grads, creates Adam optimizer, saves `model_state`/`optimizer_state` (initial checkpoint), pre-computes text embeddings |
| `adapt(x)` | Calls `self.reset()` then `self.perform_adaptation(x)` — **episodic** |
| `reset()` | Restores model + optimizer to initial state via `load_state_dict` |
| `perform_adaptation(x)` | Runs `self.steps` (default 10) gradient updates. Uses `prompt_integration='loss'` by default |
| `evaluate(x)` | `@torch.no_grad()` forward pass using uncertainty-weighted multi-layer features |
| `set_ln_grads(model)` | Freezes all visual encoder params, then enables grad only for `nn.LayerNorm` layers |
| `collect_ln_params(model)` | Returns list of LN weight/bias tensors passed to optimizer |
| `softmax_entropy(x, dim=-3)` | `-(softmax(x) * log_softmax(x)).sum(dim)` — pixel-wise entropy |

**Loss computation** (`prompt_integration='loss'`):
```python
logits, _, _, cls_logits = self.model(x, self.text_x[:-1], ...)
# text_x[:-1] = per-template embeddings (T templates × C classes × D)
# text_x[-1]  = averaged embedding (used only in evaluate())
entropy_per_pixel = self.softmax_entropy(logits)  # (#template, B, H, W)
entropy_per_cls   = self.softmax_entropy(cls_logits, dim=2)
loss = entropy_per_pixel.mean() + alpha_cls * entropy_per_cls.mean()
```

### `adapt/tent.py` — TENT Baseline

Same structure as MLMP but simpler: single prompt (averaged), only pixel-level entropy, no multi-level features (uses only last layer, `vision_outputs=(-1,)`).

Also episodic: calls `self.reset()` inside `adapt()`.

### `adapt/tpt.py` — TPT Baseline

Different: adapts **soft prompt tokens** (4 learnable tokens in text encoder) instead of visual LN. The text encoder IS partially used (transformer forward pass). Episodic.

### `adapt/clipartt.py` — CLIPArTT Baseline

Uses a self-distillation loss: takes top-K predicted classes, builds a refined prompt "a photo of class_i or class_j or...", computes image–text similarity matrix, and minimizes cross-entropy against that soft target. Episodic.

### `adapt/watt.py` — WATT Baseline

Multiple prompt templates × multiple weight-averaging rounds. For each prompt template, runs `l` gradient steps, saves LN weights, then weight-averages across all templates. Repeats `m` rounds. Also episodic.

---

## The CTTA Conversion — What Needs to Change

### Core Principle
**Standard TTA (current)**: `reset → adapt → evaluate → reset → adapt → evaluate → ...`
**CTTA (target)**: `adapt → evaluate → adapt → evaluate → ...` (no reset; model is a single persistent instance processing a continuous stream)

### Required Changes by File

#### `main.py`

1. **Remove per-corruption model re-instantiation** — instantiate `adapt_method` once before the corruption loop.
2. **Remove `adapt_method.reset()`** at line 329 — do NOT reset between samples.
3. **Add `--continual` flag** to distinguish CTTA from TTA mode.
4. **Trials (seeds) no longer make sense for CTTA** — CTTA is non-deterministic in stream order. Consider `trials=1` for CTTA.
5. The evaluation structure (metrics per corruption, saving results) can remain largely the same.

#### Each `adapt/*.py` class

To create CTTA variants (e.g. `adapt/mlmp_cotta.py`, `adapt/tent_cotta.py`):

**Remove from `adapt()`**:
```python
self.reset()  # DELETE this line
```

**Add to `__init__()`**:
```python
# Teacher model (EMA of student)
self.teacher_model = copy.deepcopy(self.model)
self.teacher_model.requires_grad_(False)
self.ema_alpha = 0.999  # smoothing factor

# Source weights backup (for stochastic restoration)
self.source_state = copy.deepcopy(self.model.state_dict())  # already exists as model_state

# Confidence threshold for augmentation gating
self.conf_threshold = 0.1  # p_th in CoTTA paper

# Number of augmentations for pseudo-label averaging
self.n_augmentations = 32  # N in CoTTA paper
```

**Update `perform_adaptation()`**:
```python
def perform_adaptation(self, x):
    # 1. Generate augmentation-averaged pseudo-label from teacher
    with torch.no_grad():
        conf = source_confidence(self.model_state, x)  # use source model
        if conf >= self.conf_threshold:
            pseudo_label = self.teacher_model(x, ...)  # direct teacher prediction
        else:
            aug_preds = [self.teacher_model(aug(x), ...) for _ in range(self.n_augmentations)]
            pseudo_label = torch.stack(aug_preds).mean(0)  # averaged

    # 2. Student update via consistency loss against pseudo-label
    for _ in range(self.steps):
        student_logits = self.model(x, ...)
        loss = cross_entropy(student_logits, pseudo_label.softmax(dim=...))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    # 3. EMA update of teacher
    for t_param, s_param in zip(self.teacher_model.parameters(), self.model.parameters()):
        t_param.data = self.ema_alpha * t_param.data + (1 - self.ema_alpha) * s_param.data

    # 4. Stochastic restoration (only LN params, since only LN is trainable)
    self._stochastic_restore()
```

**Add stochastic restoration**:
```python
def _stochastic_restore(self, p=0.01):
    """Randomly restore fraction p of LN weights back to source."""
    for (name, param), (_, src_param) in zip(
        self.model.named_parameters(), 
        # iterate source_state matching names
    ):
        if param.requires_grad:  # only LN params
            mask = torch.bernoulli(torch.ones_like(param) * p).bool()
            param.data[mask] = src_param.data[mask]
```

#### Bash Scripts

For CTTA, each script needs:
1. Add `--continual` flag (or a new `--mode ctta` argument)
2. Pass corruptions in a specific **ordered sequence** that simulates domain drift (e.g., severity 1→5, or grouped by type)
3. Use `--trials 1` (no random seed repetition makes sense for online adaptation)
4. Typically NO `--plot_loss` needed (stream is too long to plot meaningfully)

**Example CTTA bash script** (`bash/cityscapes/mlmp_cotta.sh`):
```bash
GPU_ID=0
DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
# Sequential corruption stream — ordered to simulate real-world drift
ALL_CORRUPTIONS="original gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
METHOD="mlmp_cotta"
# ... same OVSS settings ...
BATCH_SIZE=1
LR=0.00001          # lower LR for continual (avoid overshooting)
STEPS=1             # single step per sample is common in CTTA
TRIALS=1            # no trials for CTTA

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --adapt \
    --continual \   # ← new flag
    --method $METHOD \
    # ...
```

---

## Per-Method CTTA Notes

### MLMP → MLMP-CoTTA
- The multi-level feature fusion (UAML) is **compatible** with CoTTA — it only changes the loss signal quality, not the adaptation mechanism.
- The multi-prompt loss-level integration is also compatible.
- The teacher model must also use `vision_outputs` and `vision_out_type="adaptive_weighted_mean"` to match the student's forward pass.
- The ILE term (`alpha_cls`) should be kept but applied to **consistency loss** against teacher CLS predictions, not raw entropy.

### TENT → TENT-CoTTA
- Simplest conversion: remove `reset()`, add EMA teacher, stochastic restoration.
- Single prompt, single-level entropy loss → replace with teacher consistency loss.

### TPT → TPT-CoTTA
- Learnable prompt tokens `ctx` accumulate across samples (no reset).
- Teacher model also uses the current (EMA) soft prompts.
- Stochastic restoration applies to `ctx` parameters (restore toward initial `a photo of a` initialization).

### WATT → WATT-CoTTA
- Weight averaging across prompt templates is still valid.
- Remove outer `m` rounds structure (it was episodic by design).
- Use `l=1` steps per prompt template, then EMA teacher update after weight averaging.

### CLIPArTT → CLIPArTT-CoTTA
- Self-distillation loss is already somewhat compatible with CTTA.
- Replace the top-K classes from current model with top-K from EMA teacher for more stable pseudo-labels.

---

## Data Pipeline Details

### Dataset Classes (`utils/segmentation_datasets.py`)
Datasets supported: `CityscapesDataset`, `COCOStuffDataset`, `COCOObjectDataset`, `PascalVOC20Dataset`, `PascalVOC21Dataset`, `PascalContext59Dataset`, `PascalContext60Dataset`

Each dataset class:
- Inherits from `mmseg.datasets.BaseSegDataset`
- Has `METAINFO` with `classes` and `palette`
- Has optional `class_extensions` (extended class names for better CLIP matching) and `extentions_to_real_class_idx` (mapping back to original class indices)
- `ignore_index` varies per dataset

### `prepare_data()` in `segmentation_datasets.py`
Returns a `DataLoader` yielding batches with:
- `data['img_patches']`: tensor `(B, N_patches, C, H_patch, W_patch)` — patches extracted with sliding window
- `data['gt_patches']`: tensor `(B, N_patches, 1, H_patch, W_patch)` — GT for each patch
- `data['gt']`: list of original-size GT maps
- `data['meta']['patch_grid_shape']`: grid of patches per image
- `data['meta']['img_shape']`: original image shapes

### Corruption Types (15 synthetic + original)
`original`, `gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `frost`, `fog`, `brightness`, `contrast`, `elastic_transform`, `pixelate`, `jpeg_compression`

Applied via `utils/mm_transforms.py`. These are the same ImageNet-C corruption types applied to segmentation images.

### Patch Processing Flow
1. Image is resized to `INIT_RESIZE` (e.g., 1120×560 for Cityscapes)
2. Sliced into patches of `patch_size` (224×224) with `patch_stride` (112) overlap
3. Each patch fed independently to CLIP visual encoder
4. Patch logits aggregated back via `aggregate_pred_patches()` to reconstruct full-resolution segmentation

---

## Model Architecture Notes (`ovss/clip/model.py`)

The CLIP model was **modified** from the original to support:
1. **Multi-layer visual output**: `vision_outputs` parameter specifies which transformer block outputs to extract (negative indices from the end, e.g., `-1` = last block, `-18` = first block)
2. **Output fusion modes**: `vision_out_type` argument:
   - `"mean"`: uniform average across selected layers (used during adaptation)
   - `"adaptive_weighted_mean"`: entropy-weighted average (used during evaluation, β=1)
3. **CLS token return**: `return_vanilla_cls=True` returns CLS token predictions separately for ILE loss
4. **Weight tracking**: `model.weights_track` stores layer weights per forward pass for visualization
5. **Interpolation**: `interpolate=True` upsamples patch predictions to full image size

Text features `self.text_x` shape: `(T+1, C, D)` where T=number of prompts, +1 is the averaged embedding at index `[-1]`.

---

## Hyperparameter Defaults by Dataset

| Dataset | `INIT_RESIZE` | `BATCH_SIZE` | `LR` | `STEPS` | Notes |
|---------|--------------|-------------|------|---------|-------|
| Cityscapes | 1120×560 | 1 (mlmp) / 2 (others) | 1e-3 | 10 | Large images, batch=1 for MLMP |
| Pascal VOC 20/21 | 224×224 | 2 | 1e-3 | 10 | Smallest resolution |
| Pascal Context 59/60 | 224×224 | 2 | 1e-3 | 10 | |
| COCO-Stuff / COCO-Obj | varies | 2 | 1e-3 | 10 | |

For CTTA: lower LR (1e-4 or 1e-5) and fewer steps (1) per sample are recommended to avoid overshooting.

---

## Common Pitfalls for CTTA Conversion

1. **`adapt()` calls `reset()` internally** — ALL five adapt classes call `self.reset()` as their first line inside `adapt()`. Remove this call, not just the one in `main.py`.

2. **Model is re-instantiated per corruption** — `main.py:296-301` deletes and recreates the model between corruption types. This must be removed; the model must persist across the entire corruption stream.

3. **`trials` loop in main.py** — For CTTA, trials=1. The outer `for t in range(args.trials):` loop at `main.py:312` resets the whole experiment. Either remove or guard with `if args.continual: assert args.trials == 1`.

4. **Text embeddings are frozen** — `self.text_x` is computed once in `__init__` with `torch.no_grad()`. This does NOT need to change for CTTA; it remains fixed.

5. **EMA teacher must also use LN-only params** — the teacher model is a deep copy of the full CLIP model. Only LN parameters are updated in the student, but EMA should apply to ALL parameters (the teacher drifts slowly everywhere, not just LN).

6. **Stochastic restoration probability** — CoTTA uses `p=0.01` for classification. For segmentation with LN-only adaptation, this may need tuning. The LN layers have very few parameters so even `p=0.01` restores a noticeable fraction.

7. **Augmentation for CTTA in OVSS** — CoTTA's N=32 augmentations is expensive. For OVSS segmentation with patch processing, consider N=4–8. The augmentation must be applied to the full image before patching, not to individual patches.
