# SAR-MLMP-SmoothAnchor-Continual — Design Spec

**Date**: 2026-06-07
**Method name**: `sar_mlmp_smooth_anchor_continual`
**Class**: `SARMLMPSmoothAnchorContinual`
**File**: `adapt/sar_mlmp_smooth_anchor_continual.py`

---

## 1. Goal

Combine three existing ingredients into one continual-TTA method for NA-CLIP OVSS:

1. **SAR** ([adapt/sar_continual.py](../adapt/sar_continual.py)) — sample-level reliable
   filter + pixel-level reliable filter + **SAM** (sharpness-aware) optimizer.
2. **MLMP / UAML** ([adapt/mlmp_continual.py](../adapt/mlmp_continual.py)) — multi-prompt
   (7 templates, loss-averaged) **and** multi-layer feature fusion across 18 ViT-L/14
   layers (`vision_out_type="mean"` during adapt, `"adaptive_weighted_mean"` during eval).
3. **Smooth anchor** ([adapt/tent_divgate_smooth_anchor.py](../adapt/tent_divgate_smooth_anchor.py)) —
   continuous `lag(H)` mapping from marginal-diversity entropy to restoration anchor depth,
   over a rotating fp16-CPU LN-snapshot buffer.

The nearest existing precedent is
[adapt/deyo_mlmp_smooth_anchor_continual.py](../adapt/deyo_mlmp_smooth_anchor_continual.py)
(filter-loss + full MLMP + smooth anchor). **This method is structurally identical to it,
with the DeYO filter+reweight loss replaced by SAR's reliable filter + SAM two-step.**

CTTA hard-rule compliance: no per-sample reset; anchors are partial stochastic restores
toward past/source snapshots, never a full reset.

---

## 2. Component ownership

| Concern | Source | Detail |
|---|---|---|
| Sample-level filter | SAR | Skip the whole image if mean per-pixel entropy `> e_margin`. |
| Pixel-level filter | SAR | Loss computed only on pixels with entropy `< e_margin`. |
| Optimizer | SAR | `SAM` (two forward+backward passes per step) over `optim.Adam`. |
| Hard recovery | **DROPPED** | SAR's loss-EMA → full source reset is removed; smooth-anchor's `H ≤ h_floor → restore-to-source` is the gentler replacement (same choice as `sar_divgate_continual`). |
| Adapt forward | MLMP | 7 prompts (`text_x[:-1]`) + multi-layer `vision_out_type="mean"`. |
| Evaluate forward | MLMP | `text_x[-1]` (averaged prompt) + `vision_out_type="adaptive_weighted_mean"`, `save_weights=True`. |
| ILE (optional) | MLMP | `alpha_cls * H(cls_logits)`; off by default (`alpha_cls=0.0`). |
| Restoration gate | SmoothAnchor | `lag(H)=lag_scale/(H−h_floor)`, fixed `rst`, fp16-CPU deque. |

---

## 3. Constructor signature

```python
SARMLMPSmoothAnchorContinual(
    ovss_type, ovss_backbone, lr, classes, steps=1,
    vision_outputs=tuple(range(-1, -19, -1)),   # 18 ViT layers for UAML
    prompt_dir='prompts.yaml',
    alpha_cls=0.0,                # ILE weight; >0 enables the CLS-entropy term
    uaml_in_adapt=True,           # FLAG: multi-layer fusion in the adapt loss
    # --- SAR ---
    e_margin=1.8, sam_rho=0.05,
    # --- SmoothAnchor ---
    h_ceil=2.9, h_floor=2.2,
    lag_scale=150.0, max_lag=3000, rst=0.005,
    monitor_interval=50,
    save_dir=None,
    runtime_calculation=False, device='cpu')
```

LN selection is **SAR-faithful**: train *all* visual-encoder LayerNorm (γ, β) including
`ln_pre`/`ln_post` (SAR's `set_ln_grads` / `collect_ln_params`, ≈100 tensors on ViT-L/14).
No `top_block_exclude` (that is a DeYO-specific trick and is intentionally omitted).

### 3.1 `--uaml_in_adapt` flag (the requested toggle)

- `uaml_in_adapt=True` (default): `_adapt_forward` reads all `vision_outputs` layers, fuses
  with `vision_out_type="mean"`. UAML drives **both** adaptation and inference.
- `uaml_in_adapt=False`: `_adapt_forward` uses the last layer only (`vision_outputs=(-1,)`).
  `evaluate()` **still** uses the full multi-layer entropy-weighted fusion. This is the
  "eval-only UAML" ablation (mirrors `tent_uaml_eval_continual`), and halves the per-step
  cost because each SAM forward reads one layer instead of 18.

The flag is plumbed as `--uaml_in_adapt {0,1}` (argparse `type=int`, constructor `bool()`),
matching the `reweight_ent`/`reweight_plpd` convention.

---

## 4. Adaptation algorithm (`perform_adaptation`)

```
perform_adaptation(x):
    _push_current_to_anchor_buf()          # snapshot LN BEFORE this batch's step

    for _ in range(steps):
        # ---- forward 1 (real entry state) ----
        logits, cls = _adapt_forward(x)        # logits (T,B,C,h,w); cls (T,B,C,1,1) or None
        ent_map = softmax_entropy(logits)      # reduce class dim, mean over T -> (B,h,w)

        # ---- gate signal: every batch, even if filtered ----
        with no_grad:
            probs_ens = logits.softmax(-3).mean(0)         # (B,C,h,w)
            marginal_buf.append(probs_ens.mean([0,2,3]).cpu())

        # ---- SAR sample filter ----
        if ent_map.mean() <= e_margin:
            pixel_mask = ent_map < e_margin
            if pixel_mask.sum() > 0:
                loss1 = ent_map[pixel_mask].mean()
                if cls is not None: loss1 += alpha_cls * H_cls(cls)
                loss1.backward(); optimizer.first_step(zero_grad=True)

                # ---- SAM step 2 at perturbed weights ----
                logits2, cls2 = _adapt_forward(x)
                ent_map2 = softmax_entropy(logits2)
                pixel_mask2 = ent_map2 < e_margin
                if pixel_mask2.sum() > 0:
                    loss2 = ent_map2[pixel_mask2].mean()
                    if cls2 is not None: loss2 += alpha_cls * H_cls(cls2)
                    loss2.backward(); optimizer.second_step(zero_grad=True)
                    loss_report.append(loss2.item())
                    if current_rst > 0: _stochastic_restore(current_rst, current_lag)
                else:
                    optimizer.second_step(zero_grad=True)   # undo perturbation, no real update, no restore
            else:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        batch_count += 1; total_batches += 1
        if batch_count >= monitor_interval: _update_lag()

    return loss_report
```

Notes:
- **Restoration fires only after a real `second_step`** (a genuine update). On either filter
  miss or the perturbation edge-case, no snapshot pull-back happens. (Same discipline as
  `sar_divgate_continual`.)
- **The gate buffer is pushed every batch**, including filtered ones, so `H_margin` has an
  uninterrupted signal even during stretches where SAR skips adaptation.
- The control-flow uses nested `if`s (not `continue`) so `batch_count`/`total_batches`/monitor
  always advance — mirrors `deyo_mlmp_smooth_anchor_continual`.

### 4.1 `_adapt_forward`

```python
def _adapt_forward(self, x):
    vo = self.vision_outputs if self.uaml_in_adapt else (-1,)
    if self.alpha_cls > 0:
        logits, _, _, cls = self.model(
            x, self.text_x[:-1], True, interpolate=False,
            vision_outputs=vo, return_vanilla_cls=True, vision_out_type="mean")
        return logits, cls
    logits, _, _ = self.model(
        x, self.text_x[:-1], True, interpolate=False,
        vision_outputs=vo, vision_out_type="mean")
    return logits, None
```

### 4.2 `evaluate` (UAML, unchanged from MLMP)

```python
logits, _, _ = self.model(
    x, self.text_x[-1], True,
    vision_outputs=self.vision_outputs,
    interpolate=True,
    vision_out_type="adaptive_weighted_mean",
    save_weights=True)
return logits[0]
```

---

## 5. Smooth-anchor gate (verbatim from `deyo_mlmp_smooth_anchor_continual`)

```
lag(H) = lag_scale / (H - h_floor)     for h_floor < H < h_ceil
lag(H) = None (-> frozen source)       for H <= h_floor  OR  raw lag > max_lag
no restore (rst=0)                     for H >= h_ceil
```

- `_lag_from_h(h)`: returns `0` (off), `None` (source), or `int` in `[1, max_lag]`.
- `_anchor_buf`: `deque(maxlen=max_lag+1)` of fp16-CPU LN snapshots; index `-1` is newest.
- `_stochastic_restore(rst, lag)`: per-element Bernoulli(`rst`) blend toward the anchor
  (`_source_ln_snapshot` if `lag is None`, else `_anchor_buf[-1-lag]` with oldest-entry fallback).
- `_update_lag()` every `monitor_interval` batches: aggregate `marginal_buf`, normalize,
  `H = -Σ p log p`, set `current_lag`/`current_rst`, append a row to `gate_log.csv`, clear buffer.

**Threshold scale**: the gate `H_margin` here is the **adapt-time 7-prompt ensemble**
(mean over T of softmax → mean over batch+space), which sits ~0.9 above evaluate-based
diversity. Healthy ≈ 3.0. Hence defaults `h_ceil=2.9`, `h_floor=2.2`, `lag_scale=150`
(same anchoring as `deyo_mlmp_smooth_anchor_continual`), **not** the 1.6/1.4 of single-prompt
TENT. These are starting points; `gate_log.csv` records the realized H so they can be retuned.

---

## 6. Logging

- `{save_dir}/gate_log.csv` — header `total_batches,h_margin,lag,rst`, one row per `_update_lag()`.
- `{save_dir}/sar_log.txt` — header `total_batches,mean_entropy,was_filtered,was_reset`.
  `was_reset` is always 0 (hard recovery dropped) but kept for column compatibility with
  `sar_continual`'s log; **`mean_entropy` + `was_filtered` are the key diagnostics** for tuning
  `e_margin` — see §8 risk.
- `save_dir` is auto-injected by `get_method()` because `--save_dir` is in the base argparse.

---

## 7. Integration points

1. **`adapt/__init__.py`**: `from .sar_mlmp_smooth_anchor_continual import SARMLMPSmoothAnchorContinual`
   and add `'sar_mlmp_smooth_anchor_continual': SARMLMPSmoothAnchorContinual` to `METHOD_CLASSES`.
2. **`main_continual.py` → `add_method_specific_args`**: new `elif` branch registering
   `--vision_outputs` (default 18 layers), `--prompt_integration` not needed, `--alpha_cls`,
   `--uaml_in_adapt`, `--e_margin`, `--sam_rho`, `--h_ceil`, `--h_floor`, `--lag_scale`,
   `--max_lag`, `--rst`, `--monitor_interval`.
3. No source-stat / pre-stream dispatch needed (unlike DPCore/EATA/CMA-Proto).

---

## 8. Risks & open questions

- **`e_margin` scale under fusion (PRIMARY RISK)**: SAR's `e_margin=1.8` was tuned on
  single-layer single-prompt per-pixel entropy. Multi-layer `mean` fusion + 7-prompt averaging
  tends to *flatten* the distribution → higher per-pixel entropy → the filter may reject almost
  every pixel and starve adaptation. **Mitigation**: the `sar_log.txt` records `mean_entropy`
  and `was_filtered` per batch; the smoke test must confirm a non-trivial pass rate, and
  `e_margin` is fully `--`-overridable. If the pass rate is ~0, raise `e_margin` (e.g. toward
  `0.6·ln(C) ≈ 1.77`→ higher, or to a fraction of the observed mean entropy).
- **Cost**: with `uaml_in_adapt=True`, each step is 2 multi-layer (×18) vision forwards. This is
  the heaviest method in the repo. `uaml_in_adapt=False` halves it for quick iteration.
- **Gate threshold transfer across datasets**: 19–20-class datasets share `ln(C)≈3.0`, so the
  2.9/2.2 anchoring should transfer; confirm against `gate_log.csv` per dataset and retune if H
  lives in a different band.

---

## 9. Bash runners (6 datasets)

One runner per dataset, mirroring the corresponding `tent_divgate_smooth_anchor.sh`/
`deyo_mlmp_smooth_anchor` configs but with `--prompt_dir prompts.yaml`, `--vision_outputs`
(18 layers), `--e_margin`, `--sam_rho`, and MLMP-family `LR=5e-6`. Gate defaults
`h_ceil=2.9 h_floor=2.2 lag_scale=150 rst=0.005 monitor_interval=50`. All hyperparameters
env-overridable; `GPU_ID` defaults to 3.

| Folder | DATASET | DATA_DIR | CONDITIONS | INIT_RESIZE | ROUNDS |
|---|---|---|---|---|---|
| `bash/ACDC_10_round/` | ACDCDataset | `data/ACDC/` | fog night rain snow | 1120 560 | 150 |
| `bash/v20_acdc_matched/` | PascalVOC20Dataset | `data/VOC/VOC2012/` | snow frost fog contrast (ACDC-matched) | 224 224 | 150 |
| `bash/cityscapes_continual/` | CityscapesDataset | `data/Cityscape/` | snow frost fog brightness contrast | 1120 560 | 150 |
| `bash/dark_zurich/` | DarkZurichDataset | `data/Dark_Zurich_val_anon/` | night | 1120 560 | 1200 |
| `bash/nighttime_driving/` | NighttimeDrivingDataset | `data/NighttimeDrivingTest/` | night | 1120 560 | 1200 |
| `bash/dz_nd_combined/` | DZ_ND_Combined | `data/` (ignored) | dark_zurich nighttime_driving | 1120 560 | 600 |

The v20 runner lives in `bash/v20_acdc_matched/` (NOT plain `bash/v20/`): a deterministic
101-img × 4-corruption = 404/round subset (`--ann_file val_subset_101_seed0.txt`, auto-generated
via `scripts/make_voc_subset.py`) matching ACDC's 406/round for direct trajectory overlay; the 4
corruptions map to ACDC conditions (snow↔snow, fog↔fog, frost↔rain, contrast↔night). It uses the
MLMP patch convention (`INIT_RESIZE 224 224`, patch 224 stride 112 → 1 patch/img). All scripts pass
`--class_extensions`.

---

## 10. Smoke test

```bash
GPU_ID=3 SAVE_DIR=save/ACDCDataset/sar_mlmp_smooth_anchor_smoke/ \
  bash bash/ACDC_10_round/sar_mlmp_smooth_anchor_continual.sh   # then Ctrl-C after ~1 round
```
Pass criteria: (1) instantiates & prints the lag(H) preview table; (2) `results_all_rounds.txt`
gets a Round-1 row with non-degenerate mIoU; (3) `sar_log.txt` shows a **non-zero fraction of
unfiltered batches** (the e_margin sanity check); (4) `gate_log.csv` shows H in a plausible band.
For a true config-only smoke, run with `--debug`.
```
