# CMA-Layered-Continual: Cross-Modal Alignment + Layer-Stratified Stochastic Restoration

**Status**: Design approved, implementation pending
**Created**: 2026-04-25
**Owner**: Tekai-Yen
**Related docs**: `proposal_after_cma.md` (Direction A), `docs/cma_continual_spec.md` (CMA baseline), `docs/cma_proto_continual_spec.md` (D1+D2, failed)

---

## 1. Motivation

`cma_continual` and `cma_proto_continual` both collapsed around R35–R54. The post-mortem in `proposal_after_cma.md §0.3` identified the root cause: **any loss whose class-index target `ĉ_i` is chosen by the current model's predictions participates in a confirmation-bias feedback loop, regardless of whether the target vector (text, frozen source prototype, EMA target prototype) is fixed**. Loss-level external anchors are not enough.

Direction A (`proposal_after_cma.md §1`) proposes a **structural** defense instead of a loss-level one: exploit the natural depth hierarchy of ViT-L/14 so that different layers have different freedoms to adapt. Early blocks handle low-level signals (colour, illumination, texture) where the fog/night/rain/snow domain shift actually lives; late blocks hold the semantic class concepts that confirmation bias corrupts. By applying **layer-dependent stochastic restoration** (CoTTA-style restore toward source weights), early blocks are left almost free while late blocks are strongly anchored.

This spec describes the minimal version (A-Minimal): CMA loss unchanged, plus layer-stratified restoration after each gradient step. Teacher models and augmentation-averaged pseudo-labels from the full Direction A proposal are deliberately out of scope — this design isolates the "layer protection" axis for clean attribution.

---

## 2. Design Principle

- **Loss**: identical to `cma_continual` (top-K% cross-modal alignment toward predicted class text embedding). Same tensor mechanics, same hyperparameters.
- **Update target**: visual encoder LayerNorm γ, β (same as all existing continual methods).
- **New mechanism**: after each `optimizer.step()`, stochastically restore a fraction of each LN parameter toward its source value, with the restoration probability determined by which transformer block the parameter belongs to.

Three groups with a shared source-state snapshot:

| Group | Block indices | Default rst | Intuition |
|---|---|---|---|
| early | `[0, early_cutoff)` + `ln_pre` | 0.001 | Domain statistics are allowed to drift |
| mid | `[early_cutoff, late_cutoff)` | 0.01 | Standard CoTTA-level anchoring |
| late | `[late_cutoff, num_blocks)` + `ln_post` | 0.05 | Strong protection of semantic layer |

Defaults follow `proposal_after_cma.md §1.4` (`early_cutoff=8`, `late_cutoff=16` for ViT-L/14's 24 blocks). Setting any rate to `0` disables restoration for that group; setting it to `1` fully freezes those parameters.

---

## 3. Algorithm

### 3.1 Per-batch adapt step

```
for _ in range(steps):
    logits, image_features, text_features = model(x, text_x, True, interpolate=False)
    loss = cma_loss(logits, image_features, text_features)      # identical to cma_continual
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    stochastic_layered_restore()                                # <-- NEW
```

### 3.2 Layer classification

Every LayerNorm parameter of the visual encoder is classified by its state-dict name. The full state-dict name has the form `visual.transformer.resblocks.{i}.ln_{1|2}.{weight|bias}`, plus `visual.ln_pre.{weight|bias}` and `visual.ln_post.{weight|bias}`.

```python
def _get_restoration_rate(name):
    if 'ln_pre'  in name: return early_rst          # pre-block norm → early
    if 'ln_post' in name: return late_rst           # post-block norm → late
    m = re.search(r'resblocks\.(\d+)\.', name)
    if m is None:
        return 0.0                                  # defensive: should never fire for visual LN
    idx = int(m.group(1))
    if idx < early_cutoff: return early_rst
    if idx < late_cutoff:  return mid_rst
    return late_rst
```

### 3.3 Stochastic restoration

Matches CoTTA's restore pattern ([adapt/cotta.py:237-242](adapt/cotta.py#L237-L242)) but with a per-parameter rst:

```python
@torch.no_grad()
def stochastic_layered_restore(self):
    for name, p in self.named_ln_params:             # precomputed: (state-dict name, param tensor)
        rst = self._get_restoration_rate(name)
        if rst <= 0.0:
            continue
        mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
        src  = self.model_state[name].to(p.device, dtype=p.dtype)
        p.data.mul_(1.0 - mask).add_(src * mask)
```

- `self.named_ln_params` is built once in `__init__` by walking the visual encoder modules and emitting `(f"visual.{module_name}.{param_name}", param_tensor)` for every `nn.LayerNorm` weight/bias — matches the key format used by `copy.deepcopy(model.state_dict())`.
- `self.model_state` is the existing source snapshot that every continual class already keeps for episodic-reset semantics.
- Random mask is per-element per-step (not per-parameter-tensor), so every element has an independent probability `rst` of being restored each step.

---

## 4. File Layout

| Path | Action | Purpose |
|---|---|---|
| `adapt/cma_layered_continual.py` | NEW | `class CMALayeredContinual` — full implementation |
| `adapt/__init__.py` | EDIT | Add `'cma_layered_continual': CMALayeredContinual` entry |
| `main_continual.py` | EDIT | Add method branch in `add_method_specific_args` |
| `bash/ACDC_10_round/cma_layered_continual.sh` | NEW | Runnable experiment script |
| `parse_acdc_results.py` | EDIT | Append `("CMA-Layered-continual", "cma_layered_continual_step_1", False)` to `METHODS` |
| `docs/cma_layered_continual_spec.md` | NEW | This document |

`adapt/cma_layered_continual.py` will be a standalone class (not inheriting from `CMAContinual`), mirroring the "each method is self-contained" pattern already established by `cma_proto_continual.py`. The CMA-loss tensor ops are copied verbatim from `adapt/cma_continual.py`.

---

## 5. Public API

Matches every other continual method:

```python
CMALayeredContinual(
    ovss_type, ovss_backbone, lr, classes,
    steps=1,
    top_k_percent=0.2,
    early_rst=0.001, mid_rst=0.01, late_rst=0.05,
    early_cutoff=8,  late_cutoff=16,
    prompt_dir=None,
    runtime_calculation=False,
    device='cpu',
)

.adapt(x)              # single-sample adapt (no reset)
.continual_adapt(x)    # alias for adapt — main_continual.py protocol
.evaluate(x)           # torch.no_grad inference
.reset()               # load source snapshot (episodic use only)
```

No `obtain_src_*` pre-stream hook is needed — unlike `cma_proto_continual`, this method needs no per-class source statistics.

---

## 6. Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `lr` | `1e-5` | Same as `cma_continual` |
| `steps` | `1` | Online CTTA default |
| `top_k_percent` | `0.2` | Same as `cma_continual` (k=0.5 variant available via script edit) |
| `early_rst` | `0.001` | Proposal default |
| `mid_rst` | `0.01` | Proposal default (== CoTTA baseline rst) |
| `late_rst` | `0.05` | Proposal default (stronger than CoTTA) |
| `early_cutoff` | `8` | ViT-L/14 block index, `[0, 8)` is early |
| `late_cutoff` | `16` | `[8, 16)` is mid, `[16, 24)` is late |
| `continual_rounds` | `150` | Match `cma_continual` baseline run length |

### 6.1 Degradation paths

- `early_rst = mid_rst = late_rst = 0` → equivalent to `cma_continual` (no restoration at all). Useful sanity check.
- `early_rst = mid_rst = late_rst = r` (same value r) → uniform CoTTA-like restoration with rate r.
- `early_rst = 0, mid_rst = 0, late_rst = 1` → late blocks fully frozen; early/mid fully free. Ablation for "is late-layer freezing enough?".
- `early_cutoff = 24` → everything is early.
- `late_cutoff = 0` → everything is late (everything strongly anchored).

---

## 7. CLI Surface

Added to `main_continual.py::add_method_specific_args`:

```python
elif method == 'cma_layered_continual':
    parser.add_argument('--top_k_percent', type=float, default=0.2,
                        help='Top-K%% confidence mask for the CMA loss')
    parser.add_argument('--early_rst', type=float, default=0.001,
                        help='Stochastic restore probability for early blocks '
                             '([0, early_cutoff) and ln_pre)')
    parser.add_argument('--mid_rst', type=float, default=0.01,
                        help='Stochastic restore probability for mid blocks '
                             '([early_cutoff, late_cutoff))')
    parser.add_argument('--late_rst', type=float, default=0.05,
                        help='Stochastic restore probability for late blocks '
                             '([late_cutoff, num_blocks) and ln_post)')
    parser.add_argument('--early_cutoff', type=int, default=8,
                        help='Block index boundary: blocks [0, early_cutoff) are early')
    parser.add_argument('--late_cutoff', type=int, default=16,
                        help='Block index boundary: blocks [early_cutoff, late_cutoff) '
                             'are mid; [late_cutoff, num_blocks) are late')
```

---

## 8. Bash Script

`bash/ACDC_10_round/cma_layered_continual.sh` — based on `cma_continual.sh`, exposes all new parameters as shell variables:

```bash
#!/bin/bash
GPU_ID=3
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

METHOD="cma_layered_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=0.2

# Layer-stratified restoration
EARLY_RST=0.001
MID_RST=0.01
LATE_RST=0.05
EARLY_CUTOFF=8
LATE_CUTOFF=16

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
    --adapt \
    --method $METHOD \
    --ovss_type $OVSS_TYPE \
    --ovss_backbone $OVSS_BACKBONE \
    --dataset $DATASET --data_dir $DATA_DIR --init_resize $INIT_RESIZE \
    --patch_size 224 224 --patch_stride 112 \
    --corruptions_list $CONDITIONS --workers $WORKERS \
    --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \
    --continual_rounds $CONTINUAL_ROUNDS --seed 0 \
    --top_k_percent $TOP_K_PERCENT \
    --early_rst $EARLY_RST --mid_rst $MID_RST --late_rst $LATE_RST \
    --early_cutoff $EARLY_CUTOFF --late_cutoff $LATE_CUTOFF \
    --save_dir $SAVE_DIR \
    --class_extensions
```

---

## 9. Sanity Checks (manual, run before the full experiment)

1. **Source state is captured**: `self.model_state['visual.transformer.resblocks.0.ln_1.weight']` is not None after `__init__`.
2. **`named_ln_params` covers every expected layer**: length equals `2 * num_LN_modules` (weight + bias each). For ViT-L/14: `2 * (24 blocks × 2 LN + ln_pre + ln_post) = 2 * 50 = 100` parameters.
3. **No-restoration path matches `cma_continual`**: with `early_rst = mid_rst = late_rst = 0`, running one adapt step produces the same LN parameter values as `cma_continual` on the same batch / seed.
4. **Full-restoration freezes**: with `early_rst = mid_rst = late_rst = 1`, LN parameters after any adapt step equal the source snapshot.
5. **Layer classification is correct**: unit-style manual check — iterate over `named_ln_params`, print `(name, block_index, assigned_group)` for the first 5 and last 5 entries. Confirm `ln_pre` → early, `resblocks.7.ln_1` → early, `resblocks.8.ln_1` → mid, `resblocks.15.ln_2` → mid, `resblocks.16.ln_1` → late, `resblocks.23.ln_2` → late, `ln_post` → late, with default cutoffs.
6. **Debug run completes**: `bash bash/ACDC_10_round/cma_layered_continual.sh` with `--debug` flag (add manually or truncate conditions) processes a few batches without NaN losses, produces `results_all_rounds.txt`.

---

## 10. Success Criteria

From `proposal_after_cma.md §5`:

| Tier | Criterion | Meaning |
|---|---|---|
| Basic | R150 mean mIoU ≥ 15 | Avoided the dead-state basin (vs CMA's 1.20 at R40+) |
| Target | Overall mean ≥ 25 over 150 rounds | Beats CoTTA (23.4) and is stable |
| Ideal | Peak mean ≥ 26.82 with no later collapse | Matches or beats CMA peak AND stays there |
| Stretch | Overall mean ≥ 30.6 | Beats MLMP-episodic (step=10) — the research goal |

### 10.1 Expected interpretation paths

- **All tiers fail (still collapses)**: either (a) late blocks also need stronger rst → raise `late_rst` to 0.1–0.2, or (b) the loss-level confirmation bias dominates the layer protection → combine with Direction B (diversity gate) in a follow-up experiment.
- **Basic passes, target fails (stable-low)**: restoration is too strong globally → halve all three rates.
- **Target passes, ideal fails (stable-mid)**: plasticity/stability balance is good but needs more adaptation signal → lower `early_rst` toward 0, or raise `top_k_percent` to 0.5.
- **Ideal passes**: log the rst/cutoff sweep needed to find this regime, then run the cutoff ablation listed in `proposal_after_cma.md §1.5`.

---

## 11. Out of Scope (for this spec)

- EMA teacher for pseudo-label generation (A-Full / A-Teacher from the brainstorming session).
- Augmentation-averaged predictions (CoTTA-style `aug_n` forward passes per sample).
- Direction B (Diversity-Gated CMA) — separate spec, planned next.
- Direction C (Two-Timescale Meta-Adaptation) — separate spec, deferred.
- Combining Direction A + B — only meaningful after both are independently verified.

---

*End of design. Implementation plan to be produced by `writing-plans`.*
