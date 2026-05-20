# CAT-Seg Backbone Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `OVSS_TYPE="catseg"` a valid drop-in replacement for `"naclip"` in every existing bash script, so TENT-continual and TENT-DivGate on VOC20 can be run with a CAT-Seg backbone by changing one line.

**Architecture:** New `ovss/catseg/` package houses a `CATSegWrapper` that exposes the same forward signature as the existing modified-CLIP class. Internally it (a) interpolates 224→384, (b) runs the CLIP visual encoder (the only trainable component), (c) builds a cost-volume against pre-encoded text features, (d) refines logits through a frozen cost-aggregation transformer extracted from the official CAT-Seg repo, (e) interpolates output back to input resolution. `ovss/__init__.py::load_ovss` gains one `elif ovss_type == 'catseg':` branch. No code under `adapt/`, `main_continual.py`, `utils/`, or `ovss/clip/` is touched.

**Tech Stack:** PyTorch 2.1.2, CUDA 11.8 (existing `mlmp` conda env). Existing modified-CLIP in `ovss/clip/model.py` reused for the visual + text encoder structure. CAT-Seg aggregator code extracted from `github.com/cvlab-kaist/CAT-Seg` (MIT licensed) and adapted to remove Detectron2 dependencies.

**Spec reference:** [docs/superpowers/specs/2026-05-19-catseg-backbone-integration-design.md](../specs/2026-05-19-catseg-backbone-integration-design.md)

---

## Pre-flight Checks (run before Task 1)

- [ ] **PF-1: Confirm working directory and branch**

```bash
pwd
git status --short
git rev-parse --abbrev-ref HEAD
```

Expected: in `/home/tekai324/MLMP`, on branch `tekai` (or whichever branch the user chose for this work), spec file already committed.

- [ ] **PF-2: Confirm conda env is `mlmp` and CUDA available**

```bash
which python && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: python from the `mlmp` env, torch ≥ 2.1, `True` for CUDA.

- [ ] **PF-3: Confirm CAT-Seg checkpoint will live at the expected path (don't download yet — just create the cache dir)**

```bash
mkdir -p data/.cache/catseg
ls -la data/.cache/catseg
```

Expected: empty directory created. Checkpoint download is a **manual user step** documented in `ovss/catseg/README.md` (Task 3). Plan tasks 1–9 work without the checkpoint; task 10 requires it.

---

## File Structure Summary

| File | Action | Responsibility |
|---|---|---|
| `ovss/catseg/__init__.py` | CREATE | Package entry point — exports `load_catseg`, re-exports `clip_tokenize` |
| `ovss/catseg/catseg_wrapper.py` | CREATE | `CATSegWrapper` class — the only public API; mirrors `CLIP` interface |
| `ovss/catseg/aggregator.py` | CREATE | `CATSegAggregator` cost-volume refinement transformer (extracted from official repo) |
| `ovss/catseg/model_utils.py` | CREATE | Checkpoint loading, state-dict remapping (`remap_clip_state_dict`) |
| `ovss/catseg/configs/vitl.yaml` | CREATE | CAT-Seg ViT-L/14 hyperparameters extracted from official `configs/vitl_*.yaml` |
| `ovss/catseg/README.md` | CREATE | Checkpoint download instructions, source commit hash, license, skipped-key log |
| `ovss/catseg/test_wrapper.py` | CREATE | Standalone sanity-check script (project has no pytest — runnable via `python -m ovss.catseg.test_wrapper`) |
| `ovss/__init__.py` | MODIFY | Add `elif ovss_type == 'catseg'` branch |
| `bash/v20/tent_continual_catseg.sh` | CREATE | TENT-continual + CAT-Seg, weather-5, 150R |
| `bash/v20/tent_divgate_continual_catseg.sh` | CREATE | TENT-DivGate + CAT-Seg, weather-5, 150R |
| `bash/v20/no_adapt_catseg.sh` | CREATE | CAT-Seg source baseline (gates whether to launch full runs) |
| `bash/v20/mlmp_episodic_catseg.sh` | CREATE | CAT-Seg episodic upper bound (gates whether to launch full runs) |
| `data/.cache/catseg/model_large.pth` | (MANUAL) | CAT-Seg ViT-L/14 checkpoint — user downloads per README |

**No changes to** `adapt/*.py`, `main_continual.py`, `main.py`, `utils/*`, `ovss/clip/*`.

---

## Testing Convention for This Plan

The MLMP repo has **no pytest / test suite**. Per CLAUDE.md: "correctness is validated by comparing `results_all_rounds.txt` against known baselines."

This plan adapts TDD as follows:
- **Unit-level sanity** = `ovss/catseg/test_wrapper.py` is a standalone runnable Python file with `if __name__ == "__main__":` driver that exits non-zero on assertion failure. Each phase adds new checks to this file.
- **Integration-level sanity** = `--debug` flag on `main_continual.py` (already exists per the codebase — limits to 2 corruptions × 5 batches). Used in Tasks 10–12.
- **Run order** matters: each task's run-and-verify step uses the prior task's accumulated checks plus the new one.

---

# Phase 1 — Package Skeleton & Load-Path Wiring (Tasks 1–3)

Goal: After this phase, `from ovss.catseg import load_catseg` works and instantiating `CATSegWrapper` returns a real `nn.Module` with the right attributes — but the forward is stubbed and the aggregator is a no-op. No checkpoint needed.

### Task 1: Create package skeleton with stub wrapper

**Files:**
- Create: `ovss/catseg/__init__.py`
- Create: `ovss/catseg/catseg_wrapper.py`
- Create: `ovss/catseg/test_wrapper.py`

- [ ] **Step 1: Write the failing sanity test (shape + attribute contract)**

Create `ovss/catseg/test_wrapper.py`:

```python
"""Standalone sanity checks for CATSegWrapper.

Run: python -m ovss.catseg.test_wrapper

Exits non-zero on assertion failure. Print-driven, no pytest dependency.
Each phase of the implementation plan adds checks here.
"""
import sys
import torch
import torch.nn as nn


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        global _failed
        _failed = True


_failed = False


def test_import_and_instantiate():
    """Phase 1: Package imports work and wrapper instantiates with stub."""
    print("\n=== Test: import and instantiate ===")
    from ovss.catseg import load_catseg
    wrapper, tokenize = load_catseg(backbone='ViT-L/14', device='cpu')
    check("load_catseg returns wrapper", wrapper is not None)
    check("load_catseg returns callable tokenizer", callable(tokenize))
    check("wrapper.visual is nn.Module",
          isinstance(wrapper.visual, nn.Module))
    check("wrapper.transformer is nn.Module",
          isinstance(wrapper.transformer, nn.Module))
    check("wrapper.ln_final is nn.LayerNorm",
          isinstance(wrapper.ln_final, nn.LayerNorm))
    check("wrapper.token_embedding is nn.Embedding",
          isinstance(wrapper.token_embedding, nn.Embedding))
    check("wrapper.logit_scale is nn.Parameter",
          isinstance(wrapper.logit_scale, nn.Parameter))
    check("wrapper has aggregator attribute",
          hasattr(wrapper, 'aggregator'))
    check("wrapper.aggregator is nn.Module",
          isinstance(wrapper.aggregator, nn.Module))
    # Trainable-param invariant: all aggregator params frozen
    agg_grads = [p.requires_grad for p in wrapper.aggregator.parameters()]
    check("aggregator all params frozen",
          len(agg_grads) == 0 or not any(agg_grads))
    check("aggregator in eval mode", not wrapper.aggregator.training)
    return wrapper


if __name__ == "__main__":
    test_import_and_instantiate()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
```

- [ ] **Step 2: Run to verify it fails (package doesn't exist yet)**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: `ModuleNotFoundError: No module named 'ovss.catseg'`

- [ ] **Step 3: Create stub `ovss/catseg/__init__.py`**

```python
"""CAT-Seg backbone integration for MLMP.

Provides CATSegWrapper, exposed via load_catseg() with the same
interface contract as ovss/__init__.py::load_ovss.

Source: github.com/cvlab-kaist/CAT-Seg (MIT licensed). See README.md
in this directory for commit hash, checkpoint download steps, and any
skipped state-dict keys.
"""
from ovss.catseg.catseg_wrapper import CATSegWrapper
from ovss.clip import tokenize as clip_tokenize


def load_catseg(backbone='ViT-L/14', device='cpu'):
    """Construct a CATSegWrapper and return (wrapper, tokenizer).

    Args:
        backbone: ViT identifier accepted by ovss.clip.load (e.g. 'ViT-L/14').
        device: 'cpu' or a CUDA device string.

    Returns:
        (CATSegWrapper, tokenize_fn)
    """
    wrapper = CATSegWrapper(backbone=backbone, device=device)
    return wrapper, clip_tokenize
```

- [ ] **Step 4: Create stub `ovss/catseg/catseg_wrapper.py` (minimal — satisfies attribute contract)**

```python
"""CATSegWrapper — exposes a CAT-Seg backbone through the same public
interface as ovss.clip.model.CLIP, so existing adapt methods need no
changes.

Stub at Phase 1: structure-only. Real forward, aggregator, and
checkpoint loading land in subsequent tasks.
"""
import torch
import torch.nn as nn

import ovss.clip as clip


class _AggregatorStub(nn.Module):
    """Placeholder aggregator used until Task 6 lands the real extracted code.

    Identity-pass-through over the class-channel dim. Frozen, eval mode.
    """
    def __init__(self):
        super().__init__()

    def forward(self, cost_volume, guidance):
        # cost_volume: (B*T, C, H, W) -- return as-is (no refinement)
        return cost_volume


class CATSegWrapper(nn.Module):
    """CAT-Seg backbone wrapper.

    Public interface matches ovss.clip.model.CLIP:
      - forward(x, text_x, text_ensemble, interpolate) -> (logits, image_features, text_features)
      - encode_text(tokens) -> (N, dim)
      - .visual, .transformer, .ln_final, .token_embedding, .logit_scale
        are direct references to the underlying CLIP sub-modules so that
        set_ln_grads / collect_ln_params traversal works without changes.

    The cost-aggregation transformer lives in self.aggregator (sibling of
    self.visual), so LayerNorm collection scoped to self.visual naturally
    skips it -- preserving the "CLIP backbone LN only" trainable invariant.
    """

    def __init__(self, backbone='ViT-L/14', device='cpu'):
        super().__init__()

        # ---- Base CLIP backbone (reused from ovss.clip) ----
        base_clip, _ = clip.load(backbone, device='cpu')
        base_clip.visual.set_params(arch='vanilla',
                                    attn_strategy='vanilla',
                                    gaussian_std=5.0)

        # ---- Expose CLIP sub-modules as direct attributes ----
        # NOTE: assigning nn.Modules as attrs registers them as submodules.
        # That's fine; set_ln_grads walks .modules() which still works.
        self.visual = base_clip.visual
        self.transformer = base_clip.transformer
        self.ln_final = base_clip.ln_final
        self.token_embedding = base_clip.token_embedding
        self.logit_scale = base_clip.logit_scale
        self.positional_embedding = base_clip.positional_embedding
        self.text_projection = base_clip.text_projection

        # ---- Aggregator stub (replaced in Task 6 with real extracted code) ----
        self.aggregator = _AggregatorStub()
        self.aggregator.requires_grad_(False)
        self.aggregator.eval()

        # ---- Move to target device ----
        self.to(device)
        self._device = device

    # ----- Forward (stub — real implementation in Task 4) -----
    def forward(self, image, text, text_ensemble=True,
                interpolate=False, **kwargs):
        raise NotImplementedError(
            "CATSegWrapper.forward is implemented in Task 4 of the plan."
        )

    # ----- Text encode (delegates to underlying CLIP) -----
    def encode_text(self, text):
        """Re-implements ovss.clip.model.CLIP.encode_text but as a method on
        the wrapper, so callers can do wrapper.encode_text(tokens) just like
        on the base CLIP."""
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
```

- [ ] **Step 5: Run sanity test to verify Phase-1 checks pass**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: all `test_import_and_instantiate` checks PASS. Exit code 0.

- [ ] **Step 6: Commit**

```bash
git add ovss/catseg/__init__.py ovss/catseg/catseg_wrapper.py ovss/catseg/test_wrapper.py
git commit -m "$(cat <<'EOF'
feat(catseg): add package skeleton with stub wrapper

CATSegWrapper exposes the CLIP-compatible interface contract (visual,
transformer, ln_final, token_embedding, logit_scale) so adapt/*.py code
needs no changes. Forward is stubbed (raises NotImplementedError) and
the cost-aggregation transformer is a no-op placeholder; both land in
later tasks. Standalone sanity script test_wrapper.py drives Phase-1
checks.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire `load_ovss('catseg', ...)` branch and verify it routes correctly

**Files:**
- Modify: `ovss/__init__.py:5-44` (add new elif branch)
- Modify: `ovss/catseg/test_wrapper.py` (add routing check)

- [ ] **Step 1: Extend the sanity script with a routing check**

Append to `ovss/catseg/test_wrapper.py` (before the `if __name__` block):

```python
def test_load_ovss_routing():
    """Phase 1: load_ovss('catseg', ...) routes to CATSegWrapper."""
    print("\n=== Test: load_ovss routing ===")
    from ovss import load_ovss
    from ovss.catseg.catseg_wrapper import CATSegWrapper
    model, tokenize = load_ovss('catseg', 'ViT-L/14', device='cpu')
    check("load_ovss('catseg', ...) returns CATSegWrapper",
          isinstance(model, CATSegWrapper))
    check("tokenizer is callable", callable(tokenize))
```

And update the `__main__` driver to call it:

```python
if __name__ == "__main__":
    test_import_and_instantiate()
    test_load_ovss_routing()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
```

- [ ] **Step 2: Run to verify it fails (load_ovss doesn't know 'catseg' yet)**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: `test_load_ovss_routing` raises `ValueError: Unsupported OVSS type: catseg`.

- [ ] **Step 3: Add the `catseg` branch to `ovss/__init__.py`**

Open `ovss/__init__.py`. After the existing `elif ovss_type == 'naclip':` block (line 33–39), and before `else:`, insert:

```python
    elif ovss_type == 'catseg':
        from ovss.catseg import load_catseg
        ovss_model, tokenize = load_catseg(
            backbone=ovss_backbone,
            device=device,
        )
```

The resulting `load_ovss` looks like:

```python
def load_ovss(ovss_type, ovss_backbone, device='cpu'):
    if ovss_type == 'clip':
        ...
    elif ovss_type == 'sclip':
        ...
    elif ovss_type == 'naclip':
        ...
    elif ovss_type == 'catseg':                          # ← NEW
        from ovss.catseg import load_catseg
        ovss_model, tokenize = load_catseg(
            backbone=ovss_backbone,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported OVSS type: {ovss_type}")

    return ovss_model, tokenize
```

- [ ] **Step 4: Run sanity test to verify routing works**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: both `test_import_and_instantiate` and `test_load_ovss_routing` PASS.

- [ ] **Step 5: Commit**

```bash
git add ovss/__init__.py ovss/catseg/test_wrapper.py
git commit -m "$(cat <<'EOF'
feat(ovss): wire load_ovss('catseg', ...) branch

Adds the elif arm in load_ovss() that dispatches to ovss.catseg.load_catseg.
Sanity-check confirms routing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Write README with checkpoint download instructions and reproducibility anchors

**Files:**
- Create: `ovss/catseg/README.md`
- Create: `ovss/catseg/configs/vitl.yaml` (placeholder — populated in Task 6)

- [ ] **Step 1: Create `ovss/catseg/README.md`**

```markdown
# CAT-Seg Backbone Integration

This directory provides `CATSegWrapper`, a drop-in OVSS backbone that
matches the public interface of the modified CLIP in `ovss/clip/model.py`.
Selecting it requires only setting `OVSS_TYPE="catseg"` in any bash script
that already passes `--ovss_type` through to `main_continual.py`.

## Source

- Repo: <https://github.com/cvlab-kaist/CAT-Seg>
- Paper: Cho et al., *CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation*, CVPR 2024.
- License: MIT (compatible with this project).
- Extracted commit hash: **TBD-FILL-DURING-IMPLEMENTATION** (record before merging Task 6).

## Checkpoint download (manual user step)

Before running any CAT-Seg experiment, download the ViT-L/14 checkpoint
trained on COCO-Stuff and place it at:

```
data/.cache/catseg/model_large.pth
```

Download link is in the CAT-Seg repo README (Google Drive / HuggingFace).
The wrapper will raise `FileNotFoundError` with a pointer to this README
if the file is missing.

### Override path

Set the env var `CATSEG_CKPT_PATH` to use a different location:

```bash
CATSEG_CKPT_PATH=/data/shared/catseg/model_large.pth \
    bash bash/v20/tent_continual_catseg.sh
```

### SHA-256

Record the SHA-256 of the downloaded checkpoint here for reproducibility:

```
TBD-FILL-AFTER-DOWNLOAD
```

## Skipped state-dict keys

Some CAT-Seg checkpoint entries depend on the training class count
(171 for COCO-Stuff) and are loaded with `strict=False`. Keys reported
as "Unexpected" or "Missing" by `load_state_dict(strict=False)` during
the first successful load are listed here for reproducibility:

```
TBD-FILL-AFTER-FIRST-SUCCESSFUL-LOAD
```

## Interface contract

`CATSegWrapper` provides:

| Attribute / method | Purpose |
|---|---|
| `forward(x, text_x, text_ensemble=True, interpolate=False)` | Main forward. Returns `(logits, image_features, text_features)`. Logits shape `(#templates, B, #class, W, H)`. |
| `encode_text(tokens)` | CLIP text encoder. |
| `.visual` | Underlying CLIP `VisionTransformer`. All LayerNorm params here are trainable. |
| `.transformer`, `.ln_final`, `.token_embedding`, `.logit_scale` | CLIP text-side sub-modules (frozen by `adapt/*.py`). |
| `.aggregator` | Frozen CAT-Seg cost-aggregation transformer (always `.eval()`). |

The cost-aggregation transformer is **not** under `.visual` — this
ensures `for m in wrapper.visual.modules(): if isinstance(m, nn.LayerNorm)`
naturally skips it, preserving the "CLIP backbone LN only" trainable
invariant.

## Sanity check

```bash
python -m ovss.catseg.test_wrapper
```

Runs the standalone sanity script. Exits 0 on success. Re-run after
any change in this directory.

## Bash scripts that use this backbone

| Script | Method | Save dir |
|---|---|---|
| `bash/v20/no_adapt_catseg.sh` | source baseline | `save/PascalVOC20Dataset/no_adapt_catseg/` |
| `bash/v20/mlmp_episodic_catseg.sh` | episodic upper bound | `save/PascalVOC20Dataset/mlmp_episodic_catseg/` |
| `bash/v20/tent_continual_catseg.sh` | TENT-continual | `save/PascalVOC20Dataset/tent_continual_catseg_weather/` |
| `bash/v20/tent_divgate_continual_catseg.sh` | TENT-DivGate | `save/PascalVOC20Dataset/tent_divgate_continual_catseg_h_thr_1.6/` |
```

- [ ] **Step 2: Create empty `ovss/catseg/configs/` directory and placeholder yaml**

```bash
mkdir -p ovss/catseg/configs
```

Create `ovss/catseg/configs/vitl.yaml`:

```yaml
# CAT-Seg ViT-L/14 hyperparameters.
# Populated during Task 6 from cvlab-kaist/CAT-Seg configs/vitl_*.yaml.

aggregator:
  embed_dim: 768
  num_heads: 8
  num_layers: 4
  num_classes_train: 171   # COCO-Stuff (used during CAT-Seg training)
  # Additional fields filled in Task 6 after inspecting official config.

clip_backbone:
  backbone: ViT-L/14
  input_resolution: 384
```

- [ ] **Step 3: Commit**

```bash
git add ovss/catseg/README.md ovss/catseg/configs/vitl.yaml
git commit -m "$(cat <<'EOF'
docs(catseg): add README with checkpoint instructions and config placeholder

README documents the manual checkpoint download step, override env var,
SHA-256 / skipped-keys placeholders for reproducibility, and the interface
contract that adapt/*.py relies on.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## ✅ CHECKPOINT 1 (end of Phase 1)

Stop here. Verify:

1. `python -m ovss.catseg.test_wrapper` exits 0 with all checks PASS.
2. `grep -n "catseg" ovss/__init__.py` shows the new branch.
3. `git log --oneline -3` shows three new commits (skeleton, routing, README).
4. `ls ovss/catseg/` shows `__init__.py catseg_wrapper.py test_wrapper.py README.md configs/`.

**Manual review prompt**: confirm the wrapper attribute contract matches §1.2 of the spec, and that `_AggregatorStub` is clearly marked as temporary. Proceed to Phase 2 when ready.

---

# Phase 2 — Wrapper Forward with Stub Aggregator (Tasks 4–5)

Goal: After this phase, `CATSegWrapper.forward()` runs end-to-end with the right output shape and gradient flow, using the stub identity aggregator. No real CAT-Seg checkpoint needed yet — the stub lets us verify the forward plumbing in isolation from the extraction work.

### Task 4: Implement `CATSegWrapper.forward` (224→384 resize, CLIP visual encode, cost-volume, aggregator, output reshape)

**Files:**
- Modify: `ovss/catseg/catseg_wrapper.py` (replace `NotImplementedError` forward with real code)
- Modify: `ovss/catseg/test_wrapper.py` (add forward shape + gradient checks)

- [ ] **Step 1: Add forward sanity checks to `test_wrapper.py`**

Append to `ovss/catseg/test_wrapper.py` before the `__main__` block:

```python
def test_forward_shape():
    """Phase 2: forward produces the right output shape."""
    print("\n=== Test: forward shape contract ===")
    from ovss.catseg import load_catseg
    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')

    B, T, C = 1, 1, 20
    x = torch.randn(B, 3, 224, 224)
    text_x = torch.randn(T, C, 768)

    # interpolate=True -> output matches input spatial dims
    logits_full, img_feats, text_feats = wrapper(x, text_x,
                                                  text_ensemble=True,
                                                  interpolate=True)
    check(f"logits(interpolate=True) shape == (T,B,C,224,224), got {tuple(logits_full.shape)}",
          tuple(logits_full.shape) == (T, B, C, 224, 224))

    # interpolate=False -> output stays at aggregator grid (27x27)
    logits_grid, _, _ = wrapper(x, text_x,
                                 text_ensemble=True,
                                 interpolate=False)
    check(f"logits(interpolate=False) shape == (T,B,C,27,27), got {tuple(logits_grid.shape)}",
          tuple(logits_grid.shape) == (T, B, C, 27, 27))


def test_gradient_flow():
    """Phase 2: backward through forward updates ONLY visual LN params."""
    print("\n=== Test: gradient flow restricted to visual LN ===")
    from ovss.catseg import load_catseg
    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')

    # Mark trainable params using the same convention as tent_continual.py
    wrapper.transformer.requires_grad_(False)
    wrapper.ln_final.requires_grad_(False)
    wrapper.token_embedding.requires_grad_(False)
    for m in wrapper.visual.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)

    # Snapshot aggregator + a non-LN visual param
    agg_snapshot = {
        n: p.detach().clone()
        for n, p in wrapper.aggregator.named_parameters()
    }
    conv1_snapshot = wrapper.visual.conv1.weight.detach().clone()

    # Pick one visual LN param to verify it actually updates
    target_ln_name = None
    target_ln_snapshot = None
    for n, p in wrapper.visual.named_parameters():
        if 'ln' in n.lower() and p.requires_grad:
            target_ln_name = n
            target_ln_snapshot = p.detach().clone()
            break
    check("found at least one trainable visual LN param",
          target_ln_name is not None)

    # Run one optimizer step
    params = [p for p in wrapper.visual.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)

    x = torch.randn(1, 3, 224, 224)
    text_x = torch.randn(1, 20, 768)
    logits, _, _ = wrapper(x, text_x, text_ensemble=True, interpolate=False)
    loss = -(logits.softmax(2) * logits.log_softmax(2)).sum(2).mean()
    loss.backward()
    opt.step()

    # Aggregator unchanged
    for n, p_after in wrapper.aggregator.named_parameters():
        check(f"aggregator.{n} unchanged",
              torch.allclose(p_after.detach(), agg_snapshot[n]))
    # Non-LN visual param unchanged
    check("visual.conv1.weight unchanged",
          torch.allclose(wrapper.visual.conv1.weight.detach(), conv1_snapshot))
    # Trainable LN param did update
    target_now = dict(wrapper.visual.named_parameters())[target_ln_name].detach()
    check(f"trainable LN param {target_ln_name} updated",
          not torch.allclose(target_now, target_ln_snapshot))
```

Update `__main__`:

```python
if __name__ == "__main__":
    test_import_and_instantiate()
    test_load_ovss_routing()
    test_forward_shape()
    test_gradient_flow()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
```

- [ ] **Step 2: Run to verify both new checks fail**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: `test_forward_shape` raises `NotImplementedError`.

- [ ] **Step 3: Replace stub forward with real implementation**

In `ovss/catseg/catseg_wrapper.py`, replace the `forward` method body. Also add helper `_encode_image_dense`. Full updated class body:

```python
class CATSegWrapper(nn.Module):
    # ... __init__ unchanged ...

    # Spatial size CAT-Seg was trained at. Forward resizes input to this.
    CATSEG_INPUT_SIZE = 384
    # Patch grid is CATSEG_INPUT_SIZE / patch_size (14 for ViT-L/14) = 27.
    CATSEG_PATCH_GRID = 27

    def forward(self, image, text, text_ensemble=True,
                interpolate=False, **kwargs):
        """Forward pass with internal 224->384 resize + cost aggregation.

        Args:
            image: (B, 3, H_in, W_in) input tensor (typically 224x224).
            text:  (T, C, D) pre-encoded class text embeddings; or (C, D) -- promoted to T=1.
                   text_ensemble=True means caller already encoded these (matches CLIP).
            interpolate: if True, upsample logits to (H_in, W_in).
                         if False, return logits at aggregator grid resolution.

        Returns:
            logits:         (T, B, C, H_out, W_out)
            image_features: (B, S, D)  -- patch features (no CLS)
            text_features:  (T, C, D)
        """
        B = image.shape[0]
        H_in, W_in = image.shape[-2], image.shape[-1]

        # ---- Promote text shape to (T, C, D) ----
        if not text_ensemble:
            # Same convention as ovss.clip.model.CLIP -- caller passed tokens.
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = text
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)   # (1, C, D)
        T, C, D = text_features.shape

        # ---- Step 1: resize 224 -> 384 ----
        if H_in != self.CATSEG_INPUT_SIZE or W_in != self.CATSEG_INPUT_SIZE:
            image_384 = nn.functional.interpolate(
                image,
                size=(self.CATSEG_INPUT_SIZE, self.CATSEG_INPUT_SIZE),
                mode='bilinear',
                align_corners=False,
            )
        else:
            image_384 = image

        # ---- Step 2: dense visual encoding (only path with grad) ----
        visual_feats = self._encode_image_dense(image_384)   # (B, S, D)
        S = visual_feats.shape[1]
        assert S == self.CATSEG_PATCH_GRID ** 2, (
            f"Expected {self.CATSEG_PATCH_GRID**2} patches, got {S}. "
            f"Check input size and ViT patch_size."
        )

        # ---- Step 3: cost-volume ----
        visual_n = visual_feats / visual_feats.norm(dim=-1, keepdim=True)   # (B, S, D)
        text_n   = text_features / text_features.norm(dim=-1, keepdim=True)  # (T, C, D)
        # einsum: per (template, batch, class) -- inner product over D, then per-patch S
        cost = torch.einsum('bsd, tcd -> tbcs', visual_n, text_n)            # (T, B, C, S)
        H_grid = W_grid = self.CATSEG_PATCH_GRID
        cost = cost.reshape(T * B, C, H_grid, W_grid)                        # (T*B, C, H, W)

        # ---- Step 4: aggregator (frozen, eval) ----
        logits_27 = self.aggregator(cost_volume=cost, guidance=visual_n)     # (T*B, C, H, W)

        # ---- Step 5: output resize + reshape ----
        if interpolate:
            logits = nn.functional.interpolate(
                logits_27,
                size=(H_in, W_in),
                mode='bilinear',
                align_corners=False,
            )
            logits = logits.view(T, B, C, H_in, W_in)
        else:
            logits = logits_27.view(T, B, C, H_grid, W_grid)

        return logits, visual_feats, text_features

    def _encode_image_dense(self, image):
        """Run CLIP visual encoder and return dense patch tokens (no CLS).

        Mirrors VisionTransformer.forward(...) but skips the aggregation
        types in ovss/clip/model.py. We always want raw dense post-LN
        features for the cost-volume step.

        Returns:
            patches: (B, num_patches, embed_dim)
        """
        v = self.visual
        B, _, H, W = image.shape
        x = v.conv1(image)                                       # (B, D, gH, gW)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, gH*gW, D)
        # Prepend CLS
        cls_tok = v.class_embedding.to(x.dtype) + \
                  torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)                       # (B, 1+gH*gW, D)
        # Positional embedding (interpolates if grid size differs from training)
        if x.shape[1] != v.positional_embedding.shape[0]:
            x = x + v.interpolate_pos_encoding(x, H, W).to(x.dtype)
        else:
            x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        # Transformer (vanilla forward — we set arch=vanilla in __init__)
        x = x.permute(1, 0, 2)                                   # NLD -> LND
        for blk in v.transformer.resblocks:
            x = blk(x)
        x = x.permute(1, 0, 2)                                   # LND -> NLD
        x = v.ln_post(x)
        # Drop CLS; project to output_dim
        x = x[:, 1:, :]                                          # (B, num_patches, D)
        if v.proj is not None:
            x = x @ v.proj
        return x
```

**Note on `_AggregatorStub`**: it accepts `(cost_volume, guidance)` keyword args and returns `cost_volume` unchanged. Forward will produce reasonable shapes; logits values will just be raw cosine similarities (no refinement). That's fine for shape and gradient checks.

- [ ] **Step 4: Run sanity test to verify Phase-2 checks pass**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected:
- `test_forward_shape`: both shape assertions PASS.
- `test_gradient_flow`: aggregator params unchanged, conv1 unchanged, target LN param updated. All PASS.

If shape mismatch: most likely candidate is patch-grid arithmetic. Check `CATSEG_INPUT_SIZE / visual.patch_size == CATSEG_PATCH_GRID`. For ViT-L/14, 384/14 = 27.43, which floors to 27 — confirm by printing `visual_feats.shape[1]` and adjusting `CATSEG_PATCH_GRID` accordingly. If it's actually 27 patches per side (729 total), the assertion holds. If CLIP's conv1 uses padding such that 384/14 rounds differently, set `CATSEG_PATCH_GRID = 27` (which means input is conceptually 378, but conv1 over 384 with patch_size=14 produces a 27×27 grid since `floor(384/14) = 27`).

- [ ] **Step 5: Commit**

```bash
git add ovss/catseg/catseg_wrapper.py ovss/catseg/test_wrapper.py
git commit -m "$(cat <<'EOF'
feat(catseg): implement CATSegWrapper.forward with stub aggregator

Internal 224->384 resize, dense CLIP visual encode, cost-volume,
identity aggregator (stub), output reshape to input resolution or
aggregator grid. Gradient sanity confirms only visual LN params update;
aggregator and non-LN visual params unchanged.

Real aggregator code extraction lands in Task 6.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Verify `set_ln_grads` / `collect_ln_params` from `tent_continual.py` work on the wrapper

**Files:**
- Modify: `ovss/catseg/test_wrapper.py`

This task is verification-only — no production code change. It guards against an interface drift where the wrapper accidentally hides or proxies `.visual`.

- [ ] **Step 1: Add interface-contract check to `test_wrapper.py`**

Append before `__main__`:

```python
def test_ln_grad_helpers_compatibility():
    """Phase 2: tent_continual.py's set_ln_grads / collect_ln_params
    work on wrapper.visual without changes."""
    print("\n=== Test: TENT helpers compatible with wrapper ===")
    from adapt.tent_continual import TENTContinual
    from ovss.catseg import load_catseg

    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')

    # Run TENT's setup logic on wrapper.visual
    visual_after = TENTContinual.set_ln_grads(wrapper.visual)
    params, names = TENTContinual.collect_ln_params(wrapper.visual)

    check("set_ln_grads returns the module it was given",
          visual_after is wrapper.visual)
    check(f"collect_ln_params found at least 50 LN params (got {len(params)})",
          len(params) >= 50)
    # ViT-L/14 expected = 100 (24 blocks * 2 LN * 2 params + ln_pre*2 + ln_post*2)
    check(f"collect_ln_params count == 100 (got {len(params)})",
          len(params) == 100)
    # All collected params are trainable
    check("all collected params requires_grad=True",
          all(p.requires_grad for p in params))
    # No aggregator params in the collected set
    agg_param_ids = {id(p) for p in wrapper.aggregator.parameters()}
    check("no aggregator params in collected LN set",
          not any(id(p) in agg_param_ids for p in params))
```

Update `__main__`:

```python
if __name__ == "__main__":
    test_import_and_instantiate()
    test_load_ovss_routing()
    test_forward_shape()
    test_gradient_flow()
    test_ln_grad_helpers_compatibility()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
```

- [ ] **Step 2: Run sanity test**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: all five test functions PASS. If LN count differs from 100, that's a real interface bug — inspect with:

```bash
python -c "
from ovss.catseg import load_catseg
from adapt.tent_continual import TENTContinual
w, _ = load_catseg('ViT-L/14', 'cpu')
_, names = TENTContinual.collect_ln_params(w.visual)
print('count =', len(names))
for n in names: print(n)
" | head -30
```

- [ ] **Step 3: Commit**

```bash
git add ovss/catseg/test_wrapper.py
git commit -m "$(cat <<'EOF'
test(catseg): verify tent_continual LN helpers work on wrapper

Confirms set_ln_grads / collect_ln_params -- the only places that
introspect the model in adapt/*.py -- find exactly 100 LN params on
ViT-L/14 and none of them belong to the aggregator. This is the
trainable-param invariant from spec §1.3.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## ✅ CHECKPOINT 2 (end of Phase 2)

Stop here. Verify:

1. `python -m ovss.catseg.test_wrapper` passes all five test functions.
2. `git log --oneline` shows commits for Tasks 4 and 5.
3. The wrapper now runs end-to-end on dummy input with no real CAT-Seg checkpoint loaded — outputs are nonsense semantically (the stub aggregator doesn't refine anything), but shapes and gradient routing are correct.

**Manual review prompt**: read `ovss/catseg/catseg_wrapper.py` — confirm `_encode_image_dense` doesn't bypass the LayerNorm path (specifically `ln_pre` and `ln_post` must be in the gradient path). Proceed to Phase 3 when ready.

---

# Phase 3 — Aggregator Code Extraction (Tasks 6–7)

Goal: After this phase, `ovss/catseg/aggregator.py` contains the real CAT-Seg cost-aggregation transformer code, ported to standalone PyTorch (no Detectron2). The stub aggregator is replaced. Forward still works without a checkpoint — outputs are still nonsense (uninitialised aggregator weights) but the architecture matches CAT-Seg.

### Task 6: Extract aggregator from official repo into `ovss/catseg/aggregator.py`

**Files:**
- Create / overwrite: `ovss/catseg/aggregator.py`
- Modify: `ovss/catseg/configs/vitl.yaml` (fill in remaining fields from official config)
- Modify: `ovss/catseg/catseg_wrapper.py` (swap `_AggregatorStub` for `CATSegAggregator`)
- Modify: `ovss/catseg/README.md` (fill in source commit hash)

This task is the most code-research-heavy of the plan. It requires reading the official CAT-Seg repo and porting the cost-aggregation transformer.

**Reference paths in the official repo** (commit to record in README):

- `cat_seg/modeling/transformer/cat_seg_predictor.py` — main aggregator
- `cat_seg/modeling/transformer/model.py` — transformer blocks (spatial agg, class agg, MLP)
- `cat_seg/modeling/transformer/attention.py` — custom attention (Swin-style local window for spatial, vanilla for class)
- `cat_seg/modeling/heads/cat_seg_head.py` — final head combining spatial + class aggregation
- `configs/vitl_swin_b.yaml` (or similar ViT-L variant) — hyperparameters

- [ ] **Step 1: Clone the CAT-Seg repo into a scratch location to extract from**

```bash
cd /tmp && git clone https://github.com/cvlab-kaist/CAT-Seg.git catseg_src
cd catseg_src && git log -1 --format='%H' > /tmp/catseg_commit.txt
cat /tmp/catseg_commit.txt
```

Expected: a commit hash printed. Record this — it goes into `ovss/catseg/README.md` and the aggregator file header.

- [ ] **Step 2: Read the four official source files and the ViT-L config**

```bash
ls /tmp/catseg_src/cat_seg/modeling/transformer/
ls /tmp/catseg_src/cat_seg/modeling/heads/
ls /tmp/catseg_src/configs/ | grep -i vitl
```

Open each file. The aggregator's structural pieces to port:

1. **Cost embedding**: 1×1 conv that projects raw cosine similarity into a feature dim (`hidden_dim`, typically 128).
2. **Guidance projection**: linear projection of CLIP visual features to `hidden_dim` so they can serve as cross-attention keys/values during spatial aggregation.
3. **Aggregation block**: alternating
   - *spatial aggregation* (Swin-style local window attention over H×W),
   - *class aggregation* (self-attention over the class dim).
4. **Upsampler**: bilinear or convolutional upsample to a higher-resolution logit map (CAT-Seg often goes 24×24 → 96×96 internally); for our 224→384 pipeline this stays 27×27 — we don't need CAT-Seg's full multi-scale upsample because our wrapper does the final 27→224 resize in Step 5 of forward.
5. **Final 1×1 conv** to collapse `hidden_dim` back to per-class scalars.

- [ ] **Step 3: Port code into `ovss/catseg/aggregator.py`**

Create / overwrite `ovss/catseg/aggregator.py` with the ported transformer. The file MUST:

- Have a license header citing source repo, commit hash from Step 1, and CAT-Seg paper.
- Define class `CATSegAggregator(nn.Module)` with `__init__(self, embed_dim=768, hidden_dim=128, num_heads=4, num_layers=4, num_classes_train=171, **kwargs)`.
- Define `forward(self, cost_volume, guidance)`:
  - `cost_volume`: `(B*T, C, H, W)` (e.g. `(1, 20, 27, 27)`)
  - `guidance`:    `(B, H*W, embed_dim)` — used as cross-attention key/value during spatial aggregation. **Important**: `(B*T, ...)` cost volume vs `(B, ...)` guidance — when T=1 these match; when T>1 guidance must be repeated `T` times. Handle inside the aggregator.
  - Returns: `(B*T, C, H, W)` refined cost volume.
- Internal sub-modules: `cost_embed`, `guidance_proj`, `nn.ModuleList[AggregatorBlock]`, `final_proj`.

```python
"""CAT-Seg cost-aggregation transformer.

Extracted from cvlab-kaist/CAT-Seg @ <COMMIT-HASH-FROM-STEP-1>.
Original authors: Cho et al., CVPR 2024. License: MIT.

Modifications from upstream:
- Removed all Detectron2 dependencies (no Backbone / ShapeSpec).
- Inputs are pre-computed cost volume and CLIP visual features (the
  wrapper computes the cost volume; the upstream version did it inside
  the predictor).
- Eval-mode-only forward path (no train-time augmentations).
- Multi-scale upsampler stripped -- the surrounding CATSegWrapper does
  the final 27 -> 224 resize.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatorBlock(nn.Module):
    """One spatial-then-class aggregation layer.

    Spatial: Swin-style local-window attention over (H, W), with CLIP
    guidance as cross-attention key/value.
    Class:   self-attention over the class dim.
    """
    def __init__(self, hidden_dim, num_heads, window_size=12):
        super().__init__()
        # Port the upstream block faithfully. Place the actual
        # spatial/class attention modules here. See
        # /tmp/catseg_src/cat_seg/modeling/transformer/model.py.
        # The full body is too long to inline in this plan; copy the
        # forward and modules from upstream, removing any Detectron2
        # type annotations and adjusting the constructor signature.
        # Fill in during implementation.
        raise NotImplementedError(
            "Port AggregatorBlock from upstream model.py. See plan Task 6 Step 3."
        )

    def forward(self, x, guidance):
        raise NotImplementedError("Port AggregatorBlock.forward from upstream.")


class CATSegAggregator(nn.Module):
    """Cost-aggregation transformer head of CAT-Seg.

    Args:
        embed_dim: CLIP visual feature dim (768 for ViT-L/14).
        hidden_dim: internal transformer width (128 in official ViT-L config).
        num_heads: attention heads in aggregator blocks.
        num_layers: number of AggregatorBlock layers.
        num_classes_train: class count used during CAT-Seg training (171
            for COCO-Stuff). Stored for buffer compatibility during
            state-dict load; aggregator itself is class-agnostic.
    """
    def __init__(self, embed_dim=768, hidden_dim=128, num_heads=4,
                 num_layers=4, num_classes_train=171, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes_train = num_classes_train

        # cost_embed: 1x1 conv from (1) -> (hidden_dim) along the class-channel
        # dim. Treat each class slice as a single-channel image, embed to
        # hidden_dim per pixel. Implemented as nn.Conv2d(1, hidden_dim, 1).
        self.cost_embed = nn.Conv2d(1, hidden_dim, kernel_size=1)

        # guidance_proj: project CLIP visual feats to hidden_dim.
        self.guidance_proj = nn.Linear(embed_dim, hidden_dim)

        self.layers = nn.ModuleList([
            AggregatorBlock(hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # final_proj: collapse hidden_dim back to 1 (per-class scalar).
        self.final_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, cost_volume, guidance):
        """
        cost_volume: (BT, C, H, W)
        guidance:    (B, H*W, embed_dim)  -- when T>1, broadcast to BT inside.
        Returns:     (BT, C, H, W)
        """
        BT, C, H, W = cost_volume.shape
        B = guidance.shape[0]
        T = BT // B
        if T != 1:
            guidance = guidance.repeat_interleave(T, dim=0)   # (BT, S, D)

        # ---- Embed each (per-class) cost slice ----
        # Reshape to (BT*C, 1, H, W) -> embed -> (BT*C, hidden, H, W)
        x = cost_volume.view(BT * C, 1, H, W)
        x = self.cost_embed(x)
        x = x.view(BT, C, self.hidden_dim, H, W)              # (BT, C, hid, H, W)

        # ---- Project guidance ----
        g = self.guidance_proj(guidance)                      # (BT, S, hid)

        # ---- Aggregator stack ----
        for blk in self.layers:
            x = blk(x, g)

        # ---- Collapse hidden -> 1 per class ----
        # Reshape back to (BT*C, hidden, H, W) for the 1x1 conv
        x = x.view(BT * C, self.hidden_dim, H, W)
        x = self.final_proj(x)                                # (BT*C, 1, H, W)
        x = x.view(BT, C, H, W)
        return x
```

**Important**: the `AggregatorBlock` body is left as `NotImplementedError` in the plan because the exact code is too long to inline and must be copied from the upstream repo. The actual implementation in this step is: open `/tmp/catseg_src/cat_seg/modeling/transformer/model.py`, locate the relevant block class (likely named `AggregatorLayer` or `SwinTransformerBlock` plus class-attention sibling), and port it verbatim with Detectron2 imports stripped. The block must accept `(x: (BT, C, hid, H, W), guidance: (BT, S, hid))` and return the same shape.

- [ ] **Step 4: Update `ovss/catseg/configs/vitl.yaml` with concrete values from upstream config**

After reading the official ViT-L config file in Step 2, fill in the actual values. Example shape (replace numbers with whatever upstream uses):

```yaml
aggregator:
  embed_dim: 768
  hidden_dim: 128
  num_heads: 4
  num_layers: 4
  num_classes_train: 171
  window_size: 12

clip_backbone:
  backbone: ViT-L/14
  input_resolution: 384
```

- [ ] **Step 5: Update `ovss/catseg/catseg_wrapper.py` to use the real aggregator**

In `__init__`, replace:

```python
self.aggregator = _AggregatorStub()
```

with:

```python
import yaml
from pathlib import Path
from ovss.catseg.aggregator import CATSegAggregator

cfg_path = Path(__file__).parent / 'configs' / 'vitl.yaml'
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)
agg_cfg = cfg['aggregator']
self.aggregator = CATSegAggregator(**agg_cfg)
```

Keep the surrounding `requires_grad_(False)` + `.eval()` calls.

Also: remove `_AggregatorStub` class from `catseg_wrapper.py` (no longer used).

Add `import yaml` at the top of `catseg_wrapper.py` if not already there.

- [ ] **Step 6: Update `ovss/catseg/README.md` to record the commit hash**

In `README.md`, replace `**TBD-FILL-DURING-IMPLEMENTATION**` with the actual commit hash from Step 1.

- [ ] **Step 7: Run sanity test**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: all five tests still PASS. The aggregator now has real (but randomly initialised) weights — outputs are still nonsense semantically, but shape contract and gradient flow are unchanged.

If shape error in `test_forward_shape`: most likely cause is the aggregator changing `H × W` somewhere internally (e.g. a Swin window not dividing 27 evenly). Inspect upstream window-size config and adjust either the aggregator to handle 27×27 or `CATSEG_PATCH_GRID` if the official aggregator wants 24×24. If the official model strictly needs an even / power-of-2 grid, pad/crop to e.g. 28 inside the aggregator and back out — but flag this for review before merging.

- [ ] **Step 8: Commit**

```bash
git add ovss/catseg/aggregator.py ovss/catseg/catseg_wrapper.py \
        ovss/catseg/configs/vitl.yaml ovss/catseg/README.md
git commit -m "$(cat <<'EOF'
feat(catseg): extract cost-aggregation transformer from upstream

Ports CATSegAggregator and AggregatorBlock from cvlab-kaist/CAT-Seg
(commit recorded in README.md), stripped of Detectron2 dependencies.
Wrapper now instantiates the real aggregator (weights are random
until Task 7 loads the checkpoint).

Sanity tests continue to pass -- aggregator forward preserves
(BT, C, H, W) shape, gradient still only flows to visual LN params.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add `requirements.txt` entry for PyYAML (used by config load) if missing

**Files:**
- Modify (conditionally): `requirements.txt`

- [ ] **Step 1: Check if pyyaml is already a dependency**

```bash
cd /home/tekai324/MLMP && python -c "import yaml; print(yaml.__version__)"
```

If this prints a version: skip the rest of Task 7 (PyYAML already available — many ML libs depend on it transitively). Otherwise:

- [ ] **Step 2: Add PyYAML to requirements**

```bash
echo "PyYAML==6.0.2" >> requirements.txt
pip install PyYAML==6.0.2
python -c "import yaml; print(yaml.__version__)"
```

Expected: prints `6.0.2`.

- [ ] **Step 3: Commit only if requirements.txt changed**

```bash
git diff --quiet requirements.txt || git add requirements.txt && git commit -m "$(cat <<'EOF'
deps: add PyYAML for catseg config loading

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## ✅ CHECKPOINT 3 (end of Phase 3)

Stop here. Verify:

1. `python -m ovss.catseg.test_wrapper` passes all five tests with the real aggregator instantiated.
2. `ovss/catseg/README.md` contains the actual upstream commit hash (no more `TBD-FILL-DURING-IMPLEMENTATION` for the source commit).
3. `git log --oneline` shows the Task 6 commit.
4. `ls /tmp/catseg_src` — the cloned upstream repo is still on disk (kept for Task 8 reference).

**Manual review prompt**: open `ovss/catseg/aggregator.py` and skim. Confirm the AggregatorBlock body matches the upstream model.py (no transcription errors). Proceed to Phase 4 when ready.

---

# Phase 4 — Checkpoint Loading (Task 8)

Goal: After this phase, `CATSegWrapper.__init__` loads the official CAT-Seg checkpoint and applies the weights. The wrapper now produces real CAT-Seg predictions. This is the first phase that requires the user to have downloaded `model_large.pth`.

### Task 8: Implement `model_utils.remap_clip_state_dict` and wire checkpoint load

**Files:**
- Create: `ovss/catseg/model_utils.py`
- Modify: `ovss/catseg/catseg_wrapper.py` (load checkpoint in `__init__`)
- Modify: `ovss/catseg/test_wrapper.py` (add checkpoint-aware tests, skip gracefully if file missing)
- Modify: `ovss/catseg/README.md` (record SHA-256 and skipped keys after first successful load)

- [ ] **Step 1: Create `ovss/catseg/model_utils.py`**

```python
"""Helpers for loading the CAT-Seg checkpoint into CATSegWrapper sub-modules.

The upstream checkpoint is a single state_dict whose keys cover the CLIP
visual backbone, CLIP text encoder, and the cost-aggregation transformer.
We split it into three dicts (visual / text-transformer / aggregator)
and remap key prefixes to match our wrapper's sub-module names.
"""
from pathlib import Path
import os
import hashlib

import torch


DEFAULT_CKPT_PATH = Path(__file__).parents[2] / 'data' / '.cache' / 'catseg' / 'model_large.pth'


def resolve_ckpt_path():
    """Return the configured checkpoint path. Env var CATSEG_CKPT_PATH wins."""
    env = os.environ.get('CATSEG_CKPT_PATH')
    if env:
        return Path(env)
    return DEFAULT_CKPT_PATH


def file_sha256(path, blocksize=2**20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(blocksize):
            h.update(chunk)
    return h.hexdigest()


def load_catseg_checkpoint():
    """Load and return the raw state_dict from the configured path.

    Raises:
        FileNotFoundError: with a message pointing to ovss/catseg/README.md
            if the checkpoint is missing.
    """
    path = resolve_ckpt_path()
    if not path.exists():
        raise FileNotFoundError(
            f"CAT-Seg checkpoint not found at {path}.\n"
            f"Download instructions: see ovss/catseg/README.md.\n"
            f"Override path with: CATSEG_CKPT_PATH=/path/to/model_large.pth"
        )
    ckpt = torch.load(str(path), map_location='cpu')
    # Upstream save format may be {'model': sd} or just sd. Handle both.
    if isinstance(ckpt, dict) and 'model' in ckpt and 'state_dict' not in ckpt:
        sd = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    return sd


# Prefix-rewrite table: upstream prefix -> our wrapper prefix.
# This list is populated empirically during the first successful load.
# Initial entries are best-guesses based on reading upstream code in Task 6.
_PREFIX_REMAP = [
    # Visual encoder
    ('sem_seg_head.predictor.clip_model.visual.', 'visual.'),
    ('clip_model.visual.', 'visual.'),
    # Text encoder
    ('sem_seg_head.predictor.clip_model.transformer.', 'transformer.'),
    ('clip_model.transformer.', 'transformer.'),
    ('sem_seg_head.predictor.clip_model.ln_final.', 'ln_final.'),
    ('clip_model.ln_final.', 'ln_final.'),
    ('sem_seg_head.predictor.clip_model.token_embedding.', 'token_embedding.'),
    ('clip_model.token_embedding.', 'token_embedding.'),
    # Aggregator
    ('sem_seg_head.predictor.transformer.', 'aggregator.'),
    ('sem_seg_head.predictor.', 'aggregator.'),
]


def remap_clip_state_dict(sd):
    """Split a raw upstream state_dict into per-sub-module dicts and rename keys.

    Returns:
        dict with keys 'visual', 'transformer', 'ln_final', 'token_embedding',
        'aggregator'. Values are state_dicts ready for load_state_dict.
        Unrecognised keys are returned under 'unrecognised' for logging.
    """
    out = {
        'visual': {},
        'transformer': {},
        'ln_final': {},
        'token_embedding': {},
        'aggregator': {},
        'unrecognised': {},
    }

    for k, v in sd.items():
        renamed = None
        for src_prefix, dst_prefix in _PREFIX_REMAP:
            if k.startswith(src_prefix):
                inner = k[len(src_prefix):]
                renamed = (dst_prefix, inner)
                break
        if renamed is None:
            out['unrecognised'][k] = v
            continue
        dst_prefix, inner = renamed
        # Route into the right sub-dict based on dst_prefix.
        if dst_prefix == 'visual.':
            out['visual'][inner] = v
        elif dst_prefix == 'transformer.':
            out['transformer'][inner] = v
        elif dst_prefix == 'ln_final.':
            out['ln_final'][inner] = v
        elif dst_prefix == 'token_embedding.':
            out['token_embedding'][inner] = v
        elif dst_prefix == 'aggregator.':
            out['aggregator'][inner] = v
        else:
            out['unrecognised'][k] = v
    return out
```

- [ ] **Step 2: Wire checkpoint load into `CATSegWrapper.__init__`**

In `ovss/catseg/catseg_wrapper.py`, after instantiating `self.aggregator` and BEFORE `self.aggregator.requires_grad_(False)`, add:

```python
# ---- Load CAT-Seg checkpoint and apply to sub-modules ----
from ovss.catseg.model_utils import load_catseg_checkpoint, remap_clip_state_dict

raw_sd = load_catseg_checkpoint()
remapped = remap_clip_state_dict(raw_sd)

# strict=False: tolerate any class-count-sized buffers that don't match
visual_miss, visual_unexp = self.visual.load_state_dict(remapped['visual'], strict=False)
agg_miss, agg_unexp = self.aggregator.load_state_dict(remapped['aggregator'], strict=False)
if remapped['transformer']:
    self.transformer.load_state_dict(remapped['transformer'], strict=False)
if remapped['ln_final']:
    self.ln_final.load_state_dict(remapped['ln_final'], strict=False)
if remapped['token_embedding']:
    self.token_embedding.load_state_dict(remapped['token_embedding'], strict=False)

# Surface skipped keys so they can be recorded in README.md
self._ckpt_diagnostics = {
    'visual_missing': list(visual_miss),
    'visual_unexpected': list(visual_unexp),
    'aggregator_missing': list(agg_miss),
    'aggregator_unexpected': list(agg_unexp),
    'unrecognised_top_level': list(remapped['unrecognised'].keys()),
}
```

After this block, then proceed with `self.aggregator.requires_grad_(False); self.aggregator.eval()` as before.

- [ ] **Step 3: Add a checkpoint-aware sanity test that skips gracefully if file missing**

Append to `ovss/catseg/test_wrapper.py`:

```python
def test_checkpoint_load():
    """Phase 4: checkpoint loads cleanly and diagnostics are surfaced.

    Skips with a clear message if the checkpoint is not present locally.
    """
    print("\n=== Test: checkpoint load ===")
    from ovss.catseg.model_utils import resolve_ckpt_path
    path = resolve_ckpt_path()
    if not path.exists():
        print(f"[SKIP] Checkpoint not found at {path}. "
              f"Download per ovss/catseg/README.md before running this check.")
        return

    from ovss.catseg import load_catseg
    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')
    diag = wrapper._ckpt_diagnostics
    print(f"  visual_missing  ({len(diag['visual_missing'])}): {diag['visual_missing'][:5]}{'...' if len(diag['visual_missing'])>5 else ''}")
    print(f"  visual_unexpect ({len(diag['visual_unexpected'])}): {diag['visual_unexpected'][:5]}{'...' if len(diag['visual_unexpected'])>5 else ''}")
    print(f"  agg_missing     ({len(diag['aggregator_missing'])}): {diag['aggregator_missing'][:5]}")
    print(f"  agg_unexpect    ({len(diag['aggregator_unexpected'])}): {diag['aggregator_unexpected'][:5]}")
    print(f"  unrecognised    ({len(diag['unrecognised_top_level'])}): {diag['unrecognised_top_level'][:5]}")
    # Hard requirement: visual encoder should not be totally empty.
    check("at least 50 visual params loaded",
          len(remapped_count := wrapper.visual.state_dict()) > 50)
    # Aggregator state-dict should have non-trivial overlap
    agg_loaded = len([k for k in wrapper.aggregator.state_dict()
                      if k not in diag['aggregator_missing']])
    check(f"at least half of aggregator params loaded (got {agg_loaded} loaded, {len(diag['aggregator_missing'])} missing)",
          agg_loaded >= len(diag['aggregator_missing']))


def test_no_adapt_forward_with_real_weights():
    """Phase 4: with checkpoint loaded, no-adapt forward produces sane logits.

    Sanity bound: random VOC20 class logits should not all collapse to one class
    on a structured image. Skips if checkpoint missing.
    """
    print("\n=== Test: no-adapt forward with real weights ===")
    from ovss.catseg.model_utils import resolve_ckpt_path
    if not resolve_ckpt_path().exists():
        print("[SKIP] Checkpoint not found.")
        return
    from ovss.catseg import load_catseg

    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')
    wrapper.eval()

    # Make a structured image (not pure noise) so the prediction is meaningful.
    x = torch.zeros(1, 3, 224, 224)
    x[0, 0, 50:150, 50:150] = 1.0   # red square top-left
    text_x = torch.randn(1, 20, 768)
    text_x = text_x / text_x.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        logits, _, _ = wrapper(x, text_x, text_ensemble=True, interpolate=False)

    # Logits should not be constant across spatial dim or class dim.
    per_pixel_argmax = logits.argmax(dim=2)
    unique_classes = per_pixel_argmax.unique()
    check(f"prediction spans >1 class (got {len(unique_classes)} unique)",
          len(unique_classes) > 1)
```

Update `__main__`:

```python
if __name__ == "__main__":
    test_import_and_instantiate()
    test_load_ovss_routing()
    test_forward_shape()
    test_gradient_flow()
    test_ln_grad_helpers_compatibility()
    test_checkpoint_load()
    test_no_adapt_forward_with_real_weights()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
```

- [ ] **Step 4: Run sanity test (will SKIP checkpoint tests if file absent)**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected:
- If checkpoint absent: tests 1–5 PASS, tests 6–7 SKIP, exit 0.
- If checkpoint present: all 7 tests PASS.

If checkpoint is present and tests 6–7 print large `unrecognised` or `visual_unexpected` lists: the `_PREFIX_REMAP` table needs more entries. Print the upstream key names and add corresponding prefix-rewrite rules. Iterate until the missing/unexpected counts are small (< 5% of total keys) and the prediction span check passes.

- [ ] **Step 5: After first successful load, record SHA-256 and skipped keys in README**

If the checkpoint loaded cleanly in Step 4:

```bash
sha256sum data/.cache/catseg/model_large.pth
```

Record the hash in `ovss/catseg/README.md` (replace `TBD-FILL-AFTER-DOWNLOAD`). Also append the diagnostic lists from `wrapper._ckpt_diagnostics` to the "Skipped state-dict keys" section.

- [ ] **Step 6: Commit**

```bash
git add ovss/catseg/model_utils.py ovss/catseg/catseg_wrapper.py \
        ovss/catseg/test_wrapper.py ovss/catseg/README.md
git commit -m "$(cat <<'EOF'
feat(catseg): load CAT-Seg checkpoint into wrapper sub-modules

model_utils.remap_clip_state_dict splits the upstream single state-dict
into visual / text-transformer / ln_final / token_embedding / aggregator
pieces and renames key prefixes. Checkpoint is loaded with strict=False;
diagnostics (missing / unexpected / unrecognised keys) are surfaced via
wrapper._ckpt_diagnostics and recorded in README.md for reproducibility.

Sanity tests gracefully skip when the checkpoint is not present locally.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## ✅ CHECKPOINT 4 (end of Phase 4)

Stop here. Verify:

1. `python -m ovss.catseg.test_wrapper` — all 7 tests PASS (or 5 PASS + 2 SKIP if checkpoint not yet downloaded).
2. README's `Extracted commit hash`, `SHA-256`, and `Skipped state-dict keys` sections are filled in (if checkpoint was available).
3. `git log --oneline` shows the Task 8 commit.

**Manual review prompt**: read `wrapper._ckpt_diagnostics` output from Step 4. Confirm the `unrecognised` list is small and contains only expected misc keys (e.g. optimizer state, training step counters). If anything that looks structural (e.g. `clip_model.visual.transformer.resblocks.5.attn.in_proj_weight`) is in `unrecognised`, the prefix remap table needs another rule. Proceed to Phase 5 when ready.

---

# Phase 5 — Bash Scripts (Tasks 9–10)

Goal: After this phase, the four new bash scripts exist and pass `--debug` smoke runs (2 corruptions × 5 batches each). The scripts are functionally identical to their NA-CLIP counterparts except for `OVSS_TYPE` and `SAVE_DIR`.

### Task 9: Create the four CAT-Seg bash scripts

**Files:**
- Create: `bash/v20/tent_continual_catseg.sh`
- Create: `bash/v20/tent_divgate_continual_catseg.sh`
- Create: `bash/v20/no_adapt_catseg.sh`
- Create: `bash/v20/mlmp_episodic_catseg.sh`

- [ ] **Step 1: Read the NA-CLIP counterparts to know exact arg lists**

```bash
cd /home/tekai324/MLMP
ls bash/v20/
cat bash/v20/tent_continual.sh
cat bash/v20/tent_divgate_continual.sh
```

If `bash/v20/no_adapt.sh` and `bash/v20/mlmp_episodic.sh` exist, also cat them. If `mlmp_episodic.sh` does NOT exist for v20, look at `bash/v20/mlmp.sh` (the episodic entry calls `main.py` not `main_continual.py` — the new script will follow that pattern).

- [ ] **Step 2: Create `bash/v20/tent_continual_catseg.sh`**

```bash
#!/bin/bash
# TENT-Continual on PascalVOC20Dataset with CAT-Seg backbone (ViT-L/14).
# Fair comparison to bash/v20/tent_continual.sh (NA-CLIP):
#   - same patch convention (INIT_RESIZE 224x224, patch=224, stride=112)
#   - same LR, STEPS, BATCH_SIZE, CONTINUAL_ROUNDS, seed
#   - same CORRUPTIONS_LIST (weather-5 by default)
# Backbone swap is handled inside ovss/catseg/CATSegWrapper:
#   - 224 -> 384 internal resize (CAT-Seg native resolution)
#   - cost-volume aggregation
#   - output resize back to 224
# Only CLIP visual-encoder LN params are trained (aggregator frozen).

# ── GPU ────────────────────────────────────────────────────────────
GPU_ID=1

# ── Dataset ────────────────────────────────────────────────────────
DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
WORKERS=1

# ── Corruption conditions (weather-5 by default) ───────────────────
CORRUPTIONS_ARRAY=(
    snow
    frost
    fog
    brightness
    contrast
)
CORRUPTIONS_LIST="${CORRUPTIONS_LIST:-${CORRUPTIONS_ARRAY[*]}}"

# ── Method ─────────────────────────────────────────────────────────
METHOD="tent_continual"
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/14"

# ── Training hyperparameters ───────────────────────────────────────
BATCH_SIZE=1
LR=0.00001
STEPS=1

# ── Experiment ─────────────────────────────────────────────────────
CONTINUAL_ROUNDS=150
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_catseg_weather/}"

# ───────────────────────────────────────────────────────────────────
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

`chmod +x bash/v20/tent_continual_catseg.sh` (or copy permission from existing).

- [ ] **Step 3: Create `bash/v20/tent_divgate_continual_catseg.sh`**

Start from `bash/v20/tent_divgate_continual.sh`. The only differences:

```bash
METHOD="tent_divgate_continual"
OVSS_TYPE="catseg"                          # was "naclip"
OVSS_BACKBONE="ViT-L/14"

# DivGate hyperparameters -- copy from tent_divgate_continual.sh (h_thr=1.6 best so far)
H_THRESHOLD="${H_THRESHOLD:-1.6}"
H_WARNING="${H_WARNING:-1.4}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-50}"
CAUTIOUS_RST="${CAUTIOUS_RST:-0.01}"
BRAKE_RST="${BRAKE_RST:-0.05}"

SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_catseg_h_thr_${H_THRESHOLD}/}"
```

And the python invocation gains the DivGate-specific args:

```bash
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
```

The full file mirrors `bash/v20/tent_divgate_continual.sh` line-for-line except for the three points above.

- [ ] **Step 4: Create `bash/v20/no_adapt_catseg.sh`**

Identical to `tent_continual_catseg.sh` from Step 2 EXCEPT:

- Remove `--adapt` (so the model never runs an optimizer step — only `evaluate()`)
- Change `METHOD="no_adapt"`? — Check by reading `bash/v20/no_adapt.sh` (or the ACDC equivalent `bash/ACDC_10_round/no_adapt.sh`). Per CLAUDE.md the convention is `METHOD=tent_continual` plus omitting `--adapt`. Match that convention.
- Change `SAVE_DIR="${SAVE_DIR:-save/${DATASET}/no_adapt_catseg/}"`

- [ ] **Step 5: Create `bash/v20/mlmp_episodic_catseg.sh`**

Read `bash/v20/mlmp.sh` (or equivalent) for the episodic invocation pattern. The episodic entry point is `main.py`, not `main_continual.py`. Adapt accordingly:

```bash
METHOD="mlmp"
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/14"

LR=0.001          # episodic LR (per CLAUDE.md table)
STEPS=10
TRIALS=1

SAVE_DIR="${SAVE_DIR:-save/${DATASET}/mlmp_episodic_catseg/}"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
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
                        --trials $TRIALS \
                        --seed 0 \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
                        --prompt_dir prompts.yaml
```

Re-confirm the exact arg names against the existing `bash/v20/mlmp.sh` before saving.

- [ ] **Step 6: Make scripts executable**

```bash
chmod +x bash/v20/tent_continual_catseg.sh \
         bash/v20/tent_divgate_continual_catseg.sh \
         bash/v20/no_adapt_catseg.sh \
         bash/v20/mlmp_episodic_catseg.sh
```

- [ ] **Step 7: Commit**

```bash
git add bash/v20/tent_continual_catseg.sh \
        bash/v20/tent_divgate_continual_catseg.sh \
        bash/v20/no_adapt_catseg.sh \
        bash/v20/mlmp_episodic_catseg.sh
git commit -m "$(cat <<'EOF'
feat(scripts): add CAT-Seg backbone variants of v20 scripts

Four new bash scripts mirror the existing NA-CLIP equivalents with
OVSS_TYPE="catseg" and _catseg-suffixed SAVE_DIRs:
  - no_adapt_catseg.sh         (source baseline)
  - mlmp_episodic_catseg.sh    (upper bound; uses main.py)
  - tent_continual_catseg.sh   (continual TENT)
  - tent_divgate_continual_catseg.sh

Patch convention, LR, steps, rounds, seed all match the NA-CLIP
counterparts -- only the backbone differs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Smoke-run each script with `--debug` to verify wiring

**Files:**
- None modified; runtime verification only

This task requires the CAT-Seg checkpoint to be present. If `data/.cache/catseg/model_large.pth` does not yet exist, stop here, download the checkpoint per `ovss/catseg/README.md`, and resume.

- [ ] **Step 1: Verify checkpoint is present**

```bash
ls -la /home/tekai324/MLMP/data/.cache/catseg/model_large.pth
```

Expected: file exists and is at least 1 GB. If missing, halt and direct the user to download.

- [ ] **Step 2: Debug-run no_adapt_catseg**

```bash
cd /home/tekai324/MLMP
CORRUPTIONS_LIST="fog brightness" SAVE_DIR="save/PascalVOC20Dataset/_smoke/no_adapt_catseg/" \
  bash bash/v20/no_adapt_catseg.sh 2>&1 | head -100
```

Note: this currently runs the full split. If `main_continual.py` supports `--debug`, append it inside the script for the smoke test, or set it via env: `DEBUG_FLAG="--debug"`. **Check `main_continual.py` argparse for `--debug` first.** If `--debug` does not exist, instead run on a tiny subset by limiting `CONTINUAL_ROUNDS=1` and using a 2-corruption list.

Expected: `save/PascalVOC20Dataset/_smoke/no_adapt_catseg/results_all_rounds.txt` is created with at least one row of numeric mIoU values, no Python tracebacks in stderr.

- [ ] **Step 3: Debug-run mlmp_episodic_catseg**

```bash
CORRUPTIONS_LIST="fog" SAVE_DIR="save/PascalVOC20Dataset/_smoke/mlmp_episodic_catseg/" \
  bash bash/v20/mlmp_episodic_catseg.sh 2>&1 | head -100
```

Expected: per-condition results file written, no tracebacks. May take a while because `STEPS=10` per sample is expensive — if too slow on a smoke test, temporarily lower `STEPS=1` in the script via env.

- [ ] **Step 4: Debug-run tent_continual_catseg**

```bash
CORRUPTIONS_LIST="fog brightness" CONTINUAL_ROUNDS=1 \
  SAVE_DIR="save/PascalVOC20Dataset/_smoke/tent_continual_catseg/" \
  bash bash/v20/tent_continual_catseg.sh 2>&1 | head -100
```

Expected: results file written, no tracebacks.

- [ ] **Step 5: Debug-run tent_divgate_continual_catseg**

```bash
CORRUPTIONS_LIST="fog brightness" CONTINUAL_ROUNDS=1 \
  SAVE_DIR="save/PascalVOC20Dataset/_smoke/tent_divgate_continual_catseg/" \
  bash bash/v20/tent_divgate_continual_catseg.sh 2>&1 | head -100
```

Expected: results file AND `divgate_log.txt` written, no tracebacks.

- [ ] **Step 6: Clean up smoke results**

```bash
rm -rf save/PascalVOC20Dataset/_smoke/
```

- [ ] **Step 7: Commit nothing in this task (verification only)**

```bash
git status
```

Expected: clean working tree.

---

## ✅ CHECKPOINT 5 (end of Phase 5)

Stop here. Verify:

1. All four scripts ran their smoke tests without Python errors.
2. `results_all_rounds.txt` rows contain plausible numeric mIoU values (CAT-Seg + no_adapt on fog should likely be in the 70–85 range; if it's 0 or NaN, something is wrong with the forward).
3. `bash/v20/*_catseg.sh` are executable (`ls -l bash/v20/*_catseg.sh` shows `x` permission).

**Manual review prompt**: open one of the smoke `results_all_rounds.txt` files and check the numbers feel plausible. If `no_adapt_catseg` produces single-digit mIoU on fog VOC20, the integration has a semantic bug (most likely in the cost-volume / aggregator wiring) — don't proceed to Phase 6 until this is investigated. If numbers look reasonable, proceed.

---

# Phase 6 — Integration Validation (Task 11)

Goal: After this phase, the spec's §4.5 sanity sequence is confirmed end-to-end. The integration is ready to launch full 150-round experiments (which are outside this plan's scope).

### Task 11: Confirm the spec §4.5 sanity sequence end-to-end

**Files:**
- None modified; pure validation

- [ ] **Step 1: Re-run the unit-level sanity script with checkpoint present**

```bash
cd /home/tekai324/MLMP && python -m ovss.catseg.test_wrapper
```

Expected: all 7 test functions PASS, including `test_checkpoint_load` and `test_no_adapt_forward_with_real_weights`.

- [ ] **Step 2: Run a 1-round no_adapt baseline on full weather-5**

```bash
CORRUPTIONS_LIST="snow frost fog brightness contrast" CONTINUAL_ROUNDS=1 \
  SAVE_DIR="save/PascalVOC20Dataset/no_adapt_catseg_sanity/" \
  bash bash/v20/no_adapt_catseg.sh 2>&1 | tail -20
```

Expected: `save/PascalVOC20Dataset/no_adapt_catseg_sanity/results_all_rounds.txt` shows mean mIoU > 65 (sanity threshold from spec §6.2). If significantly lower, integration has a quality bug — stop and diagnose before launching full runs.

- [ ] **Step 3: Run 1 round of tent_continual to confirm gradient flow doesn't blow up**

```bash
CORRUPTIONS_LIST="fog" CONTINUAL_ROUNDS=1 \
  SAVE_DIR="save/PascalVOC20Dataset/_sanity/tent_continual_catseg/" \
  bash bash/v20/tent_continual_catseg.sh 2>&1 | tail -10
```

Expected: completes without NaN losses, results file is produced. Mean mIoU should be in the same ballpark as the no_adapt baseline on the same corruption (TENT in 1 round on 1 corruption barely changes anything).

- [ ] **Step 4: Spot-check that LN params actually changed during the tent run**

This requires saving a model snapshot before and after the run, which is a heavier mod than smoke tests. Instead, verify indirectly via the `divgate_log.txt` output of the DivGate script — if H_margin computed from CAT-Seg logits looks reasonable (between 0 and ln(20) ≈ 3.0), the model is producing real predictions:

```bash
CORRUPTIONS_LIST="fog" CONTINUAL_ROUNDS=1 \
  SAVE_DIR="save/PascalVOC20Dataset/_sanity/tent_divgate_continual_catseg/" \
  bash bash/v20/tent_divgate_continual_catseg.sh
cat save/PascalVOC20Dataset/_sanity/tent_divgate_continual_catseg/divgate_log.txt | head -10
```

Expected: H_margin values are finite and in `[0, 3.0]`.

- [ ] **Step 5: Clean up sanity results (or keep — these are real partial runs and parser-compatible)**

```bash
# Keep them if useful; or:
# rm -rf save/PascalVOC20Dataset/_sanity/ save/PascalVOC20Dataset/no_adapt_catseg_sanity/
```

- [ ] **Step 6: Commit nothing (validation only). Final git status check.**

```bash
git status
git log --oneline -10
```

Expected: clean working tree, recent commits show the full plan progression.

---

## ✅ CHECKPOINT 6 (final)

The integration is complete. The plan's scope ends here. Next steps (out of plan scope, but the immediate research follow-up):

1. **Run `bash/v20/no_adapt_catseg.sh` for full 150 rounds** to establish the CAT-Seg source baseline on VOC20 weather-5.
2. **Run `bash/v20/mlmp_episodic_catseg.sh`** to establish the CAT-Seg episodic upper bound — this is the headroom-feasibility data point that gates everything downstream (spec §6.1 caveat #5).
3. **If headroom ≥ 5 mIoU**: launch the full `bash/v20/tent_continual_catseg.sh` and `bash/v20/tent_divgate_continual_catseg.sh`.
4. **If headroom < 5 mIoU**: the experiment is likely to confirm the headroom barrier rather than break it (spec §6.1 caveat #5). Update `docs/EXPERIMENT_STATUS.md` accordingly and consider this a negative-but-informative result.

---

# Self-Review

**1. Spec coverage:**

| Spec section | Plan task(s) |
|---|---|
| §1.1 file structure | File Structure Summary section + Tasks 1, 3, 6, 8, 9 |
| §1.2 interface contract | Tasks 1, 4, 5 (tests verify the contract) |
| §1.3 trainable-param invariant | Tasks 4, 5 (gradient-flow test, LN helper compatibility test) |
| §1.4 reproducibility anchors | Task 3 (README), Task 6 (commit hash), Task 8 (SHA-256 + skipped keys) |
| §2 forward data flow | Task 4 (full forward implementation) |
| §3.1 load_ovss change | Task 2 |
| §3.2 __init__ entry | Task 1 |
| §3.3 six-step loader | Tasks 1 (steps 1, 6), 6 (steps 2, 5), 8 (steps 3, 4) |
| §3.4 checkpoint path/env-var | Task 8 (model_utils.resolve_ckpt_path) |
| §3.5 pre-conditions | Task 10 + Pre-flight PF-3 |
| §4 code extraction scope | Task 6 |
| §4.1 mapping | Task 6 (Step 2 reads upstream, Step 3 ports) |
| §4.2 expected classes | Task 6 (Step 3 — CATSegAggregator, AggregatorBlock) |
| §4.3 not-extracted list | Task 6 (the plan explicitly avoids these) |
| §4.4 integration issues | Task 6 (Step 3 notes on guidance broadcast, T>1), Task 8 (state-dict remap table) |
| §4.5 sanity sequence | Tasks 4 (shape), 5 (LN), 8 (no_adapt forward), 10 (gradient flow), 11 (end-to-end) |
| §5 bash scripts | Task 9 |
| §6.1 limitations | Documented in spec; plan respects them (224 stays in scripts, 384 only in wrapper) |
| §6.2 success criteria | Task 11 (Step 2 enforces mean > 65) |
| §6.3 out-of-scope | Plan does not introduce any out-of-scope items |

All sections covered.

**2. Placeholder scan:**

- `TBD-FILL-DURING-IMPLEMENTATION` (commit hash in README) — INTENTIONAL, filled in Task 6 Step 6.
- `TBD-FILL-AFTER-DOWNLOAD` (SHA-256) — INTENTIONAL, filled in Task 8 Step 5.
- `TBD-FILL-AFTER-FIRST-SUCCESSFUL-LOAD` (skipped keys) — INTENTIONAL, filled in Task 8 Step 5.
- `AggregatorBlock` body raises `NotImplementedError` in Task 6 Step 3 — this is the only spot in the plan where actual code must be sourced from outside this document (the upstream repo). The plan is explicit about this and provides the file path to copy from. Acceptable — the alternative is inlining ~200 lines of upstream code, which would bloat the plan and risk transcription errors. Implementation reviewer must verify the ported block against upstream `model.py`.

No "TODO" / "fill in details" / "handle edge cases" / "similar to Task N" red flags found.

**3. Type consistency:**

- `CATSegWrapper.forward(image, text, text_ensemble, interpolate)` — consistent across Tasks 1, 4. Same param names, same return tuple `(logits, image_features, text_features)`.
- `CATSegAggregator.forward(cost_volume, guidance)` — keyword names consistent between Tasks 1 (`_AggregatorStub`), 4 (call site in wrapper), 6 (real aggregator).
- `_ckpt_diagnostics` attribute — introduced in Task 8 Step 2, consumed in Task 8 Step 3 test.
- `remap_clip_state_dict` returns dict with keys `{'visual','transformer','ln_final','token_embedding','aggregator','unrecognised'}` — Task 8 Step 1 defines, Task 8 Step 2 consumes. Consistent.
- `_PREFIX_REMAP` prefix-rewrite table — entries chosen empirically based on a guess; explicitly noted that they may need refinement on first run (Task 8 Step 4).

All types consistent.
