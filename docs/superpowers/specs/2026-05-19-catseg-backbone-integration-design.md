# CAT-Seg Backbone Integration — Design Spec

**Date**: 2026-05-19
**Author**: tekai-yen (with Claude)
**Status**: Design — pending implementation plan
**Scope**: Add CAT-Seg (ViT-L/14) as an alternative OVSS backbone behind the
existing `--ovss_type` flag, so that bash scripts on VOC20 (and later other
datasets) can swap NA-CLIP → CAT-Seg by changing a single value. First
target methods: `tent_continual` and `tent_divgate_continual`.

---

## 0. Motivation & Goal

The Cityscapes (§4) and VOC20 (§5) phases of the project (see
`docs/EXPERIMENT_STATUS.md`) established a **headroom barrier**: continual
TTA methods do not produce positive adaptation when the gap between the
source-model mIoU and the MLMP-episodic upper bound is below ~5–7 mIoU.
On VOC20 weather-5, NA-CLIP's headroom is only 4.6 mIoU, and every
continual method we have tried (TENT-DivGate, MLMP-DivGate) ends up below
the source baseline.

**Hypothesis being tested**: a stronger OVSS backbone (CAT-Seg, which is
CLIP + a trained cost-aggregation transformer pre-trained on COCO-Stuff)
may shift the loss landscape such that TENT-style entropy minimisation
produces a usable gradient direction even when absolute mIoU is already
high.

**Success criterion (this spec is about *enabling* the experiment, not
verifying the hypothesis)**: after integration, every existing bash script
that consumes `OVSS_TYPE` can swap NA-CLIP → CAT-Seg by changing one line
(`OVSS_TYPE="catseg"`) plus a `SAVE_DIR` suffix, with **zero changes** to
any code under `adapt/`, `main_continual.py`, `utils/`, or
`ovss/clip/`.

---

## 1. Architecture & File Layout

### 1.1 New / changed files

```
MLMP/
├── ovss/
│   ├── __init__.py              [CHANGED]  load_ovss() gains 'catseg' branch
│   ├── clip/                    [UNCHANGED]  existing NA-CLIP / SCLIP / CLIP path
│   └── catseg/                  [NEW]
│       ├── __init__.py          load_catseg() entry point + tokenizer re-export
│       ├── catseg_wrapper.py    CATSegWrapper class — the only public API
│       ├── aggregator.py        cost-aggregation transformer (extracted from official repo)
│       ├── model_utils.py       checkpoint loading helpers, state-dict remapping
│       ├── configs/vitl.yaml    CAT-Seg ViT-L/14 hyperparams (extracted from official)
│       └── README.md            checkpoint download instructions, commit hash, license
├── adapt/                       [UNCHANGED]  every adapt method works as-is
├── main_continual.py            [UNCHANGED]  --ovss_type already accepts arbitrary string
├── bash/v20/
│   ├── tent_continual.sh        [UNCHANGED]  preserves existing NA-CLIP result
│   ├── tent_continual_catseg.sh [NEW]        OVSS_TYPE="catseg", new SAVE_DIR
│   ├── tent_divgate_continual.sh                  [UNCHANGED]
│   ├── tent_divgate_continual_catseg.sh           [NEW]
│   ├── no_adapt_catseg.sh                         [NEW]  source baseline for CAT-Seg
│   └── mlmp_episodic_catseg.sh                    [NEW]  upper bound for CAT-Seg
└── data/.cache/catseg/                            [NEW]  CAT-Seg checkpoint location
```

### 1.2 Interface contract

`CATSegWrapper` MUST expose the same public surface as
`ovss/clip/model.py::CLIP`, so that the adapt methods (which import this
through `load_ovss()`) need no changes:

| Member | Type | Used by | Behaviour |
|---|---|---|---|
| `wrapper(x, text_x, text_ensemble=True, interpolate=False)` | callable | `adapt/*.py::perform_adaptation`, `evaluate` | Returns `(logits, image_features, text_features)`. `logits` shape `(#templates, B, #class, W, H)`. |
| `wrapper.encode_text(tokens)` | method | `extract_text_embeddings` | Returns `(N, dim)`, identical signature to CLIP. |
| `wrapper.visual` | `VisionTransformer` instance | `set_ln_grads`, `collect_ln_params`, `torch.nn.utils` traversal | MUST be a real `nn.Module` whose `.modules()` walk reaches all CLIP backbone LayerNorms. NOT a proxy. |
| `wrapper.transformer` | `nn.Module` | `tent_continual.py:68` (text encoder freezing) | The text-side transformer. |
| `wrapper.ln_final` | `nn.LayerNorm` | `tent_continual.py:69` | Text-side final LN. |
| `wrapper.token_embedding` | `nn.Embedding` | `tent_continual.py:70` | Text token embedding. |
| `wrapper.logit_scale` | `nn.Parameter` | logit temperature scaling | Copied from CLIP, frozen. |

### 1.3 Trainable-parameter invariant

After `CATSegWrapper.__init__()` + `tent_continual` style setup, the following must hold:

- `wrapper.visual` LN γ,β: `requires_grad=True` (via `set_ln_grads`)
- `wrapper.visual` non-LN params: `requires_grad=False`
- `wrapper.transformer`, `wrapper.ln_final`, `wrapper.token_embedding`: `requires_grad=False`
- `wrapper.aggregator`: `requires_grad=False` for **every** param, and `.eval()` mode (dropout disabled)

The aggregator is **not** under `wrapper.visual` — it's a sibling attribute. This means `for m in wrapper.visual.modules(): if isinstance(m, nn.LayerNorm): ...` naturally skips it, satisfying the "CLIP backbone LN only" constraint without special-casing.

### 1.4 Reproducibility anchors

- `ovss/catseg/README.md` records: (a) source repo URL, (b) exact commit hash used for code extraction, (c) checkpoint file SHA-256, (d) any state-dict keys skipped during `strict=False` load.

---

## 2. CATSegWrapper.forward — Internal Data Flow

```
Input:  x ∈ (B, 3, 224, 224)
        text_x ∈ (#templates, #classes, dim)   ← pre-encoded class embeddings
        text_ensemble=True (always; text encoded externally via extract_text_embeddings)
        interpolate ∈ {True, False}
```

### Step 1 — Input resize 224×224 → 384×384

```python
x_384 = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
# 384 / 14 = 27.43 → CAT-Seg uses 24×24 or 27×27 grid depending on patch size.
# ViT-L/14 with input=384 gives 27×27 patch grid (no pos-embed mismatch).
```

### Step 2 — CLIP visual encode (gradient flows through this only)

```python
visual_feats, _ = self._encode_image_dense(x_384)
# visual_feats: (B, 729, 768)   ← 729 = 27*27 patches, 768 = ViT-L embed dim
# Gradients propagate to wrapper.visual LN γ,β only (other params frozen).
```

`_encode_image_dense` reaches into `self.visual` to grab post-final-LN dense
patch tokens (skipping the CLS token in slot 0). This mirrors what
CAT-Seg's own dense-feature hook does in the upstream repo.

### Step 3 — Cost-volume computation

```python
visual_n = F.normalize(visual_feats, dim=-1)               # (B, 729, 768)
text_n   = F.normalize(text_x, dim=-1)                     # (T, C, 768)
cost     = torch.einsum('bsd, tcd -> tbcs', visual_n, text_n)
# cost: (T, B, C, 729) → reshape → (T*B, C, 27, 27)
```

### Step 4 — Cost aggregation transformer (frozen, eval mode)

```python
logits_27 = self.aggregator(cost_volume=cost_reshaped,
                            guidance=visual_n)
# logits_27: (T*B, C, 27, 27)
# Aggregator parameters frozen; dropout disabled (eval mode).
# Forward still computes — gradients flow through it to reach visual LN params.
```

### Step 5 — Output resize and reshape

```python
if interpolate:
    logits = F.interpolate(logits_27, size=(224, 224),
                           mode='bilinear', align_corners=False)
    logits = logits.view(T, B, C, 224, 224)
else:
    logits = logits_27.view(T, B, C, 27, 27)

return logits, visual_feats, text_x
```

### Design rationale (key decisions)

| Decision | Choice | Rationale |
|---|---|---|
| `interpolate=False` output size | 27×27 (aggregator grid) | Matches NA-CLIP's `interpolate=False` returning patch-grid logits; used by adapt loss for efficiency. |
| `interpolate=True` output size | 224×224 (input size) | Matches NA-CLIP eval contract; downstream metric code expects per-input-pixel logits. |
| Aggregator gradient behaviour | Params `requires_grad=False`, but forward computes normally so grads pass through to upstream `visual` LN | Standard PyTorch behaviour — frozen params still propagate grads. |
| Aggregator mode | `.eval()` permanently (set in `__init__`, never toggled) | Disables dropout, ensures deterministic forward. Aggregator is never the optimisation target. |
| Multi-template | First dim of `text_x` preserved through einsum and reshape | Future-proofs for prompt ensembles; current v20 scripts use 1 template (no `--prompt_dir`). |

### Caveat (acknowledged, deferred to §6 Limitations)

The 224 → 384 upsample is **not** CAT-Seg's native input pipeline.
CAT-Seg's original training uses 384 directly from disk. Interpolating
from 224 introduces upsampling artefacts. This is the price of the
"identical patch convention" comparison. An alternative (native 384,
new patch convention) is listed under Out-of-Scope.

---

## 3. load_ovss + Checkpoint Loading

### 3.1 `ovss/__init__.py` change

```python
def load_ovss(ovss_type, ovss_backbone, device='cpu'):
    if ovss_type == 'clip':
        # ... unchanged ...
    elif ovss_type == 'sclip':
        # ... unchanged ...
    elif ovss_type == 'naclip':
        # ... unchanged ...
    elif ovss_type == 'catseg':                       # ← NEW
        from ovss.catseg import load_catseg
        ovss_model, tokenize = load_catseg(
            backbone=ovss_backbone,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported OVSS type: {ovss_type}")
    return ovss_model, tokenize
```

**Key point**: the `catseg` branch does NOT call
`visual.set_params(arch, attn_strategy, gaussian_std)`. That helper is
specific to NA-CLIP / SCLIP / vanilla-CLIP attention modifications.
CAT-Seg uses standard CLIP visual attention.

### 3.2 `ovss/catseg/__init__.py`

```python
from ovss.catseg.catseg_wrapper import CATSegWrapper
from ovss.clip import tokenize as clip_tokenize     # reuse existing tokenizer

def load_catseg(backbone='ViT-L/14', device='cpu'):
    wrapper = CATSegWrapper(backbone=backbone, device=device)
    return wrapper, clip_tokenize
```

### 3.3 `CATSegWrapper.__init__` — six-step loader

```
1. Load base CLIP backbone (reuse ovss.clip.load — gives us the modified
   CLIP class with multi-layer/CLS-return hooks, but we'll only use the
   vanilla forward path):

       base_clip, _ = ovss.clip.load('ViT-L/14', device='cpu')
       base_clip.visual.set_params(arch="vanilla",
                                   attn_strategy="vanilla",
                                   gaussian_std=5.0)

2. Build aggregator (CAT-Seg cost-aggregation transformer, extracted code):

       self.aggregator = CATSegAggregator(embed_dim=768, num_heads=8,
                                          num_layers=4,
                                          num_classes_train=171, ...)
       config from ovss/catseg/configs/vitl.yaml

3. Load CAT-Seg checkpoint and split into sub-state-dicts:

       ckpt = torch.load(CATSEG_CKPT_PATH, map_location='cpu')
       remapped = remap_clip_state_dict(ckpt)   # in model_utils.py

       base_clip.visual.load_state_dict(remapped['visual'], strict=False)
       base_clip.transformer.load_state_dict(remapped['transformer'], strict=False)
       self.aggregator.load_state_dict(remapped['aggregator'], strict=False)

   Any keys skipped during strict=False are written to
   ovss/catseg/README.md for reproducibility.

4. Freeze aggregator, set eval mode:

       self.aggregator.requires_grad_(False)
       self.aggregator.eval()

5. Move to target device:

       base_clip = base_clip.to(device)
       self.aggregator = self.aggregator.to(device)

6. Expose CLIP sub-modules as wrapper attributes (so adapt methods can
   reach them via wrapper.visual / wrapper.transformer / ...):

       self.visual          = base_clip.visual
       self.transformer     = base_clip.transformer
       self.ln_final        = base_clip.ln_final
       self.token_embedding = base_clip.token_embedding
       self.logit_scale     = base_clip.logit_scale
       self.positional_embedding = base_clip.positional_embedding
       self.text_projection = base_clip.text_projection
```

### 3.4 Checkpoint path & download

| Item | Value |
|---|---|
| Official checkpoint | `model_large.pth` (ViT-L/14, COCO-Stuff trained) |
| Source | `cvlab-kaist/CAT-Seg` README — Google Drive / HuggingFace link |
| Local path | `data/.cache/catseg/model_large.pth` |
| Override | env var `CATSEG_CKPT_PATH` |
| Missing-file behaviour | `FileNotFoundError` with a message pointing to `ovss/catseg/README.md` for download steps |
| Integrity check | SHA-256 logged in `ovss/catseg/README.md`; wrapper warns (not errors) if mismatch |

### 3.5 Pre-conditions before running any experiment

1. `data/.cache/catseg/model_large.pth` exists.
2. `ovss/catseg/README.md` records the source commit hash and any
   skipped state-dict keys from the first run.

---

## 4. Code Extraction Scope — What Comes From the Official Repo

Source: `github.com/cvlab-kaist/CAT-Seg`, commit hash recorded in
`ovss/catseg/README.md`.

### 4.1 Mapping

| Upstream path | Becomes |
|---|---|
| `cat_seg/modeling/transformer/cat_seg_predictor.py` | `ovss/catseg/aggregator.py` (main aggregator) |
| `cat_seg/modeling/transformer/model.py` | `ovss/catseg/aggregator.py` (transformer blocks) |
| `cat_seg/modeling/transformer/attention.py` | `ovss/catseg/aggregator.py` (custom attention) |
| `cat_seg/modeling/heads/cat_seg_head.py` | `ovss/catseg/aggregator.py` (head combining spatial + class agg) |
| `cat_seg/utils/` (subset) | `ovss/catseg/model_utils.py` |
| `configs/vitl_*.yaml` | `ovss/catseg/configs/vitl.yaml` |

### 4.2 Classes expected in `ovss/catseg/aggregator.py`

```python
class CATSegAggregator(nn.Module):
    """
    Cost-volume aggregator from CAT-Seg (Cho et al., CVPR 2024).
    Spatial aggregation (Swin-style local-window attention over spatial dims)
    + class aggregation (self-attention over class dim).

    Extracted from cvlab-kaist/CAT-Seg @ <commit-hash>, modified to:
      - Remove all Detectron2 dependencies
      - Accept pre-computed CLIP visual features (no internal CLIP forward)
      - Eval-mode-only path (no training-time augmentation hooks)
    """
    def forward(self, cost_volume, guidance):
        """
        cost_volume: (B*T, C_classes, H, W)
        guidance:    (B*T, H*W, embed_dim)
        Returns:     (B*T, C_classes, H, W)
        """

class AggregatorBlock(nn.Module):
    """One spatial-then-class aggregation layer."""
```

### 4.3 NOT extracted

| Excluded | Reason |
|---|---|
| Detectron2 trainer / dataloader / evaluator | We use `main_continual.py` |
| CAT-Seg's training loss head | TTA does not train against ground truth |
| Dataset registry (COCO-Stuff, ADE20K class mappings) | We pass VOC20 class names directly (zero-shot path) |
| ImageNet preprocessing helpers | We use `utils/mm_transforms.py` |
| `learnable_background` prompt handling | VOC20 setup excludes background from the 20 foreground classes (consistent with NA-CLIP) |
| Multi-scale / sliding-window inference | Single 224×224 patch matches v20 convention |

### 4.4 Known integration issues and their resolutions

| Issue | Resolution |
|---|---|
| CAT-Seg's CLIP visual encoder uses its own dense-feature hook with a different forward signature from `ovss/clip/model.py::VisionTransformer` | Wrapper manually extracts patches from the existing modified-CLIP `visual` forward (`conv1 → transformer → ln_post → drop CLS`), bypassing CAT-Seg's hook |
| Aggregator was trained with `num_classes_train=171` (COCO-Stuff) but inferenced on 20 classes (VOC20) | CAT-Seg's class aggregation is permutation-equivariant self-attention over the class dim → directly feeding 20-class cost volume works without retraining |
| Aggregator state-dict may contain 171-sized buffers (e.g., positional embeddings over class dim) | Load with `strict=False`; if any buffer turns out to be class-count-sized AND critical, write an adapter in `model_utils.py` (likely truncate to first 20 entries — to be verified during implementation) |
| State-dict key naming differs (CAT-Seg's Detectron2-style vs our OpenAI-style) | `remap_clip_state_dict()` in `model_utils.py` handles prefix stripping and key renaming |
| License compatibility | CAT-Seg is MIT (compatible with MLMP). Each extracted file gets a header noting source repo, commit, and original authors. |

### 4.5 Sanity-check sequence (verifies wrapper before launching real experiments)

1. **Shape check**: dummy `(1, 3, 224, 224)` → assert logits shape `(1, 1, 20, 224, 224)` with `interpolate=True`, `(1, 1, 20, 27, 27)` with `interpolate=False`.
2. **Source baseline**: `no_adapt_catseg.sh` on VOC20 weather-5 for 1 round. Expected: mean mIoU > 65 (CAT-Seg paper reports ~93 on clean VOC20; weather-corrupted should be in 75–85 range).
3. **Gradient flow check**: 5 batches of `tent_continual.adapt()`. Assert `wrapper.visual` LN γ,β changed; assert no `wrapper.aggregator` param changed (snapshot before/after with `torch.allclose`).
4. **Episodic upper bound**: `mlmp_episodic_catseg.sh` on VOC20 weather-5 → records the headroom number (`episodic − source`). **This number determines whether full-experiment runs are worth launching.**

---

## 5. Bash Script Mechanics

### 5.1 Minimum-diff swap

To switch any existing v20 script to CAT-Seg, the only required changes are:

```bash
OVSS_TYPE="catseg"                                   # was "naclip"
SAVE_DIR="save/${DATASET}/${METHOD}_catseg_weather/" # was ".../${METHOD}_weather/"
```

All other variables — `OVSS_BACKBONE`, `INIT_RESIZE`, `patch_size`,
`patch_stride`, `BATCH_SIZE`, `LR`, `STEPS`, `CONTINUAL_ROUNDS`,
`CORRUPTIONS_LIST`, `--method`, `--seed` — are unchanged. The 224→384
resize is handled transparently inside `CATSegWrapper.forward`.

### 5.2 Recommended convention: new scripts, not edits in place

| Don't | Do |
|---|---|
| Edit `tent_continual.sh` in place — risks confusing future runs and produces a misleading git diff over a working result | Copy to `tent_continual_catseg.sh` with two-line diff |
| Rely on `OVSS_TYPE=catseg bash tent_continual.sh` env-var override — current default `SAVE_DIR` does not encode backbone name → two backbones' results would collide | New script with explicit `SAVE_DIR` that includes `_catseg` suffix |

### 5.3 New scripts in this spec

```
bash/v20/
├── tent_continual_catseg.sh           [NEW]
├── tent_divgate_continual_catseg.sh   [NEW]
├── no_adapt_catseg.sh                 [NEW]   source-baseline reference
└── mlmp_episodic_catseg.sh            [NEW]   upper-bound reference
```

### 5.4 Template (`tent_continual_catseg.sh`)

```bash
#!/bin/bash
# TENT-Continual on PascalVOC20Dataset with CAT-Seg backbone (ViT-L/14).
# Fair comparison to tent_continual.sh (NA-CLIP):
#   - same patch convention (INIT_RESIZE 224x224, patch=224, stride=112)
#   - same LR, STEPS, BATCH_SIZE, CONTINUAL_ROUNDS, seed
#   - same CORRUPTIONS_LIST (weather-5 by default)
# Backbone swap is handled inside ovss/catseg/CATSegWrapper:
#   - 224 -> 384 internal resize (CAT-Seg native resolution)
#   - cost-volume aggregation
#   - output resize back to 224
# Only CLIP visual-encoder LN params are trained (aggregator frozen).

GPU_ID=1

DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

CORRUPTIONS_ARRAY=( snow frost fog brightness contrast )
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

METHOD="tent_continual"
OVSS_TYPE="catseg"                # ← ONLY architectural change vs tent_continual.sh
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_catseg_weather/}"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CORRUPTIONS_LIST \
                        --workers $WORKERS \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

### 5.5 Save-dir naming (preserves existing parser compatibility)

| Experiment | save_dir |
|---|---|
| NA-CLIP + TENT-continual (existing) | `save/PascalVOC20Dataset/tent_continual_weather/` |
| CAT-Seg + TENT-continual (new) | `save/PascalVOC20Dataset/tent_continual_catseg_weather/` |
| NA-CLIP + TENT-DivGate (existing) | `save/PascalVOC20Dataset/tent_divgate_continual_h_thr_1.6/` |
| CAT-Seg + TENT-DivGate (new) | `save/PascalVOC20Dataset/tent_divgate_continual_catseg_h_thr_1.6/` |
| CAT-Seg + no_adapt (new) | `save/PascalVOC20Dataset/no_adapt_catseg/` |
| CAT-Seg + MLMP-episodic (new) | `save/PascalVOC20Dataset/mlmp_episodic_catseg/` |

---

## 6. Limitations, Caveats, and Success Criteria

### 6.1 Limitations of the fair-comparison setup

1. **224 → 384 internal upsample**: CAT-Seg natively expects 384×384 input. The wrapper interpolates 224 → 384 before forward, and 27×27 logits → 224×224 after, to preserve the bash-side patch convention identical to NA-CLIP scripts. This may slightly underperform native-384 evaluation.

2. **CAT-Seg's CLIP backbone is fine-tuned on COCO-Stuff with semantic-segmentation supervision**; NA-CLIP uses vanilla pretrained CLIP. The comparison is therefore "two complete OVSS solutions" (backbone + attention strategy + optional aggregator), not "two attention strategies on the same CLIP weights". Reports MUST label methods as `tent_continual + NA-CLIP` vs `tent_continual + CAT-Seg` to prevent misreading.

3. **Aggregator is frozen and in eval mode** during adaptation. Only CLIP backbone LayerNorm γ,β are updated. Parameter count matches NA-CLIP setup (≈ 100 LN params total in ViT-L/14: 24 blocks × 2 LN × 2 params + ln_pre × 2 + ln_post × 2).

4. **Checkpoint may carry COCO-Stuff-specific buffers** (e.g., class embeddings sized 171). Loaded with `strict=False`; any skipped keys are logged in `ovss/catseg/README.md` for reproducibility.

5. **Headroom dependency is the most likely failure mode**. The ACDC → Cityscapes → VOC20 arc established that continual TTA only produces upward trends when episodic-vs-source headroom ≥ ~5–7 mIoU. If CAT-Seg's higher source baseline does not produce a proportionally higher episodic upper bound, this experiment may merely confirm the headroom barrier on a stronger backbone — itself a publishable negative result, but not the "CAT-Seg fixes TTA" story.

### 6.2 Success criteria

**Step 0 — Pre-experiment sanity (gates whether to launch full runs)**

- CAT-Seg + no_adapt on VOC20 weather-5: mean mIoU > 65 → confirms checkpoint loaded correctly and wrapper forward is correct.
- CAT-Seg + MLMP-episodic on VOC20 weather-5: mean mIoU ≥ no_adapt + 2 → confirms the adaptation mechanism is reachable on this backbone.
- Wrapper unit test: 1 step of `tent_continual.adapt()` mutates `wrapper.visual` LN γ,β but does NOT change any `wrapper.aggregator` param (snapshot before/after via `torch.allclose`).

**Step 1 — Main experiment criteria (`tent_divgate_continual + CAT-Seg`)**

- Primary: 150-round mean mIoU **exceeds** `no_adapt + CAT-Seg` baseline. Failing this means the backbone swap did not break the headroom barrier — a negative-but-informative result.
- Stretch: matches or exceeds the existing NA-CLIP-based best (`tent_divgate_continual + NA-CLIP` mean = 59.24 weather-5). This would justify backbone-swap as a generalisable improvement worth a paper section.

**Final table**: report `{no_adapt, mlmp_episodic, tent_continual, tent_divgate_continual} × {NA-CLIP, CAT-Seg}` in a single 4×2 grid.

### 6.3 Out of scope (future work)

- CAT-Seg + non-TENT methods (MLMP-continual, MLMP-DivGate, CoTTA, SAR, EATA). Extend after TENT family validates.
- CAT-Seg ViT-B/16 variant. Only ViT-L/14 in this spec.
- Native-384 patch convention. Would require new `INIT_RESIZE` / `patch_size` / `stride` defaults and a separate experimental table — not directly comparable to existing v20 results.
- Cityscapes / ACDC CAT-Seg integration. VOC20 first (cheapest test, most direct fit with CAT-Seg's COCO-Stuff pretraining domain).
- Adapting CAT-Seg's aggregator LN params. Current spec freezes aggregator entirely; relaxing this is a follow-up experiment with its own design.

---

## 7. Implementation Plan Reference

The detailed step-by-step implementation plan (file-by-file changes, test
order, commit boundaries) is produced by the `writing-plans` skill in a
separate document, invoked immediately after this spec is approved.

---

## 8. Reading Map

| Question | Where |
|---|---|
| Why CAT-Seg now? Project context. | `docs/EXPERIMENT_STATUS.md` §4 (Cityscapes), §5 (VOC20) — headroom problem |
| Existing NA-CLIP backbone code | `ovss/__init__.py`, `ovss/clip/model.py` |
| Adapt method interfaces this spec must satisfy | `adapt/tent_continual.py`, `adapt/tent_divgate_continual.py` |
| Bash script convention | `bash/v20/tent_continual.sh` (template), `docs/superpowers/specs/2026-05-08-voc-v20-continual-scripts-design.md` (patch convention) |
| CAT-Seg paper / official code | Cho et al., CVPR 2024; `github.com/cvlab-kaist/CAT-Seg` |
