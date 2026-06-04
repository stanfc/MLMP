# SAR-DivGate-Continual: SAM-Optimized Entropy with Diversity-Gated Restoration

**Status**: Design approved, implementation pending
**Created**: 2026-05-21
**Owner**: Tekai-Yen
**Related docs**: `docs/tent_divgate_continual_spec.md` (sibling, TENT-base DivGate), `docs/2026-05-17-sar-eata-v20-design.md` (SAR base implementation), `adapt/sar_continual.py`, `adapt/tent_divgate_continual.py`

---

## 1. Motivation: Two Mechanisms with Disjoint Failure Modes

The current ACDC SAR-Continual and TENT-DivGate results trade off in complementary ways:

| Method | Mean | Peak | R150 | Failure pattern |
|---|---|---|---|---|
| **SAR-continual** (sam_rho=0.05, recovery) | 30.32 | **33.38** @ ~R15 | 25.31 | Periodic "W-shape" dips at R50/R100/R150 — full hard reset cycles |
| **TENT-DivGate** (h_thr=1.6, cau_rst=0.01) | **31.59** | 32.96 @ R27 | 31.34 | Stable to R150, but peak capped by TENT's natural ceiling |
| MLMP-episodic (target) | 30.60 | — | — | Requires per-sample reset |
| No Adapt | 23.34 | — | — | Baseline |

**Two signals, two failure modes**:
- SAR monitors **per-pixel entropy EMA** (sample-level confidence). Hard reset fires too late — model already collapsed.
- DivGate monitors **marginal class entropy `H_margin`** (population-level diversity). Cautious-mode partial restore engages **mid-drift**, before collapse.

**Two defenses, two strengths**:
- SAM finds flatter LN minima → genuinely better generalization, not just "less drift". Lifts the peak from TENT's 32.96 to 33.38.
- DivGate's three-tier stochastic restore is **continuous** (mode update every 50 batches, per-element Bernoulli mask) instead of binary on-off — smooth defense without losing accumulated adaptation.

The hypothesis: **SAR's SAM gives the peak; DivGate's gradual brake holds it**. SAR's recovery is a strictly worse mechanism than DivGate's gate when both are available, and should be removed.

Predicted result: mean ≥ 32 on ACDC (above both SAR alone 30.32 and TENT-DivGate 31.59), peak retained at 33+, R150 ≥ 31.

---

## 2. Design Principle

Start from `adapt/sar_continual.py`. **Keep**: reliability filter, SAM optimizer, pixel-level filter. **Drop**: loss EMA tracking, hard `_model_recovery`. **Add**: H_margin marginal buffer + 3-tier stochastic restore (copied verbatim from `adapt/tent_divgate_continual.py`).

Each retained mechanism solves a problem the others don't:

| Mechanism | Source | Role |
|---|---|---|
| Reliability filter (sample-level) | SAR | Skip unreliable samples — no gradient pollution |
| Pixel-level reliable filter | SAR | Compute loss only on reliable pixels — local SNR |
| **SAM optimizer (ascent + descent)** | SAR | **Find flat LN minima — the only mechanism that *raises* the peak** |
| H_margin monitoring (every 50 batches) | DivGate | Detect class-diversity collapse before per-pixel confidence does |
| 3-tier stochastic restore | DivGate | Continuous graded defense; replaces SAR's hard reset |

---

## 3. Loss

Identical to SAR's reliable-pixel TENT loss:

```python
# x: (T=1, B, C, w, h)
@staticmethod
def softmax_entropy(x):
    ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)   # (T, B, w, h)
    return ent.mean(dim=0)                                # (B, w, h)
```

`logits` has `T=1` (single prompt template — matches `tent_continual` / `sar_continual` convention).

---

## 4. SAM Step (verbatim from SAR)

```python
# First forward, sample-level reliability check
logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
ent_map = self.softmax_entropy(logits)                    # (B, w, h)
ent_mean = ent_map.mean()
if ent_mean.item() > self.e_margin:
    was_filtered = True
    self.optimizer.zero_grad()
    # IMPORTANT: still push H_margin signal (see §5.1)
    continue

# Pixel-level filter
pixel_mask = ent_map < self.e_margin
if pixel_mask.sum().item() == 0:
    was_filtered = True
    self.optimizer.zero_grad()
    continue
loss = ent_map[pixel_mask].mean()
loss.backward()
self.optimizer.first_step(zero_grad=True)

# Second forward at perturbed weights
logits2, _, _ = self.model(x, self.text_x, True, interpolate=False)
ent_map2 = self.softmax_entropy(logits2)
pixel_mask2 = ent_map2 < self.e_margin
if pixel_mask2.sum().item() == 0:
    self.optimizer.second_step(zero_grad=True)
    continue
loss2 = ent_map2[pixel_mask2].mean()
loss2.backward()
self.optimizer.second_step(zero_grad=True)
```

The `loss_ma` EMA from SAR is **not** maintained. No `_model_recovery` call exists.

---

## 5. Diversity Gate

Bit-for-bit identical to `tent_divgate_continual_spec.md` §4. Reproduced here for self-containment.

### 5.1 Per-batch buffer push

`H_margin` is computed from the **first forward pass** (before SAM's perturbation), because that represents the actual model state at batch entry. The perturbed-weight forward is a virtual position that should not pollute the diversity signal.

The push happens **regardless of the reliability filter outcome** — we want collapse detection even when adaptation is skipped:

```python
with torch.no_grad():
    probs = logits[0].softmax(dim=1)             # (B, C, w, h)
    batch_marginal = probs.mean(dim=[0, 2, 3])   # (C,)
    self.marginal_buf.append(batch_marginal.detach().float().cpu())
```

Rationale: skipping a sample because it's unreliable is a gating decision on the loss path. The diversity monitor needs uninterrupted signal — if cautious mode only updates buffer on adapted samples, the gate's reaction time grows arbitrarily long under a heavy filter regime.

### 5.2 Mode update every `monitor_interval` batches

```python
agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
agg = agg / agg.sum().clamp(min=1e-8)
h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

new_mode = pick_mode(h_margin, h_threshold, h_warning)
if new_mode != self.current_mode:
    print(f"[DivGate-S] B{total_batches}: H_margin={h_margin:.3f}  {old} -> {new}")
self.current_mode = new_mode
self.current_rst  = mode_to_rst(new_mode)
self.marginal_buf.clear()
self.batch_count = 0
```

Log tag `[DivGate-S]` (S for SAR) to distinguish from `[DivGate-T]` (TENT) and `[DivGate]` (CMA) in mixed logs.

### 5.3 Three-tier mode → restoration rate

```
H_margin >= h_threshold              → "aggressive"  (rst = 0)
h_warning <= H_margin < h_threshold  → "cautious"    (rst = cautious_rst)
H_margin < h_warning                  → "brake"       (rst = brake_rst)
```

Restoration is **flat across all visual-encoder LN params** — same `_stochastic_restore_flat(rst)` helper as `tent_divgate`. Fires **after** SAM's `second_step`:

```python
if self.current_rst > 0.0:
    self._stochastic_restore_flat(self.current_rst)
```

Initial mode is `aggressive` (`current_rst = 0`); buffer fills for the first `monitor_interval` batches before the first re-evaluation.

---

## 6. Adapt Step (complete)

```
forward(x) → logits  (1, B, C, w, h)
ent_map = softmax_entropy(logits)
ent_mean = ent_map.mean()

# DivGate: push marginal signal regardless of filter outcome
with no_grad:
    self.marginal_buf.append(logits[0].softmax(1).mean([0,2,3]).cpu())

if ent_mean > e_margin:
    optimizer.zero_grad()
    # skip adaptation, but still tick gate counters below
else:
    pixel_mask = ent_map < e_margin
    if pixel_mask.sum() == 0:
        optimizer.zero_grad()
    else:
        loss = ent_map[pixel_mask].mean()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward at perturbed weights
        logits2 = model(x)
        ent_map2 = softmax_entropy(logits2)
        pixel_mask2 = ent_map2 < e_margin
        if pixel_mask2.sum() == 0:
            optimizer.second_step(zero_grad=True)
        else:
            loss2 = ent_map2[pixel_mask2].mean()
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            if self.current_rst > 0:
                _stochastic_restore_flat(self.current_rst)

batch_count += 1
total_batches += 1
if batch_count >= monitor_interval:
    _update_mode()
```

Key: stochastic restore only fires when a real SAM `second_step` happened. Filtered or degenerate batches do not trigger restore (no parameters changed, nothing to restore). Buffer push and mode update still happen unconditionally.

---

## 7. Public API

```python
SARDivGateContinual(
    ovss_type, ovss_backbone, lr, classes,
    steps=1,
    # SAR knobs
    e_margin=1.8, sam_rho=0.05,
    # DivGate knobs
    h_threshold=1.6, h_warning=1.4,
    monitor_interval=50,
    cautious_rst=0.01, brake_rst=0.05,
    prompt_dir=None,
    save_dir=None,
    runtime_calculation=False,
    device='cpu',
)

.adapt(x) / .continual_adapt(x) / .evaluate(x) / .reset()
```

No `obtain_src_*` hook. No `e_0`, `ema_factor`, `recovery_warmup` (SAR recovery removed). No `top_k_percent`.

`save_dir` is auto-injected by `main_continual.py` via inspect-based dispatch (same as `tent_divgate_continual`); used for `divgate_log.txt` if present.

---

## 8. Hyperparameters

| Parameter | Default | Origin | Notes |
|---|---|---|---|
| `lr` | `1e-5` | shared | Match all DivGate / SAR scripts |
| `steps` | `1` | shared | Online CTTA |
| `e_margin` | `1.8` | SAR-ACDC tuned | Paper default 0.4·ln(19)≈1.18; we use the SAR-ACDC tuned value 1.8 because that matches the SAR baseline we're improving on |
| `sam_rho` | `0.05` | SAR paper | Perturbation radius |
| `h_threshold` | `1.6` | TENT-DivGate ACDC best | Aggressive boundary |
| `h_warning` | `1.4` | TENT-DivGate ACDC best | Cautious / brake boundary |
| `monitor_interval` | `50` | shared | Batches between H_margin checks |
| `cautious_rst` | `0.01` | TENT-DivGate ACDC best | Mid restoration rate |
| `brake_rst` | `0.05` | TENT-DivGate ACDC best | Strong restoration rate |
| `continual_rounds` | `150` | shared | Match other DivGate runs |

### 8.1 Degradation paths

- `sam_rho = 0` → SAM degenerates to plain SGD/Adam step → equivalent to `tent_divgate_continual` with reliability filter. **Useful ablation**: isolates whether SAM contributes the peak gain.
- `e_margin = 999` → reliability filter never triggers → all samples enter SAM step → tests how much filtering matters
- `h_threshold = 999` → gate permanently aggressive → equivalent to `sar_continual` minus recovery (i.e., "SAR-Filter+SAM only")
- `cautious_rst = brake_rst = 0` → no restoration ever; gate diagnostic-only; tests whether SAR's reliable-pixel TENT loss alone is collapse-resistant

---

## 9. CLI Surface

Combined SAR + DivGate args in `main_continual.py` `add_method_specific_args`:

```python
elif method == 'sar_divgate_continual':
    # SAR-side
    parser.add_argument('--e_margin', type=float, default=1.8,
                        help='Sample mean-pixel entropy threshold; skip if exceeded')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM perturbation radius')
    # DivGate-side
    parser.add_argument('--h_threshold', type=float, default=1.6,
                        help='H_margin >= this -> aggressive mode (rst=0)')
    parser.add_argument('--h_warning', type=float, default=1.4,
                        help='h_warning <= H_margin < h_threshold -> cautious; '
                             '< h_warning -> brake')
    parser.add_argument('--monitor_interval', type=int, default=50,
                        help='Batches between H_margin re-evaluations')
    parser.add_argument('--cautious_rst', type=float, default=0.01,
                        help='Stochastic restore probability in cautious mode')
    parser.add_argument('--brake_rst', type=float, default=0.05,
                        help='Stochastic restore probability in brake mode')
```

No `--e_0`, `--ema_factor`, `--recovery_warmup` (deleted from SAR). No `--top_k_percent`.

---

## 10. Files

| Action | Path | Purpose |
|---|---|---|
| Create | `adapt/sar_divgate_continual.py` | `SARDivGateContinual` class |
| Modify | `adapt/__init__.py` | Register `'sar_divgate_continual': SARDivGateContinual` in METHOD_CLASSES |
| Modify | `main_continual.py` | Add `elif method == 'sar_divgate_continual'` branch to `add_method_specific_args` (see §9) |
| Create | `bash/ACDC_10_round/sar_divgate_continual.sh` | ACDC runner (4 conditions × 150 rounds) |
| Create | `bash/v20/sar_divgate_continual.sh` | VOC20 runner (weather subset default) |
| Create | `bash/cityscapes_continual/sar_divgate_continual.sh` | Cityscapes runner (weather subset default) |

Reuse `adapt/sam.py` unchanged (SAR's SAM wrapper).

No changes to `ovss/`, `utils/`, `main.py`, or any other adapt method.

---

## 11. Bash Scripts

All three follow the existing per-dataset SAR template (`bash/{ACDC_10_round,v20,cityscapes_continual}/sar_continual.sh`) with these substitutions:

```bash
METHOD="sar_divgate_continual"

# SAR (unchanged from sar_continual.sh)
E_MARGIN=1.8
SAM_RHO=0.05
# NOTE: E_0, EMA_FACTOR, RECOVERY_WARMUP removed — recovery replaced by DivGate

# DivGate (TENT-DivGate ACDC best, copied)
H_THRESHOLD=1.6
H_WARNING=1.4
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.01
BRAKE_RST=0.05

SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}_weather/}"
```

CLI block adds:
```
--e_margin $E_MARGIN --sam_rho $SAM_RHO \
--h_threshold $H_THRESHOLD --h_warning $H_WARNING \
--monitor_interval $MONITOR_INTERVAL \
--cautious_rst $CAUTIOUS_RST --brake_rst $BRAKE_RST \
```

Removes SAR-specific `--e_0 --ema_factor --recovery_warmup` lines.

---

## 12. Sanity Checks

1. **`named_ln_params` count = 100** on ViT-L/14 (same as `tent_divgate_continual` and `sar_continual`)
2. **Initial mode = aggressive, current_rst = 0**, buffer empty
3. **`loss_ma` field does NOT exist** on the instance (regression guard against accidentally keeping SAR's recovery state)
4. **SAM uses Adam under the hood**: `self.optimizer.base_optimizer` should be Adam, lr matches CLI arg
5. **Buffer push fires every batch, even when filtered** — log a flag and grep `sar_log.txt` (if added) to confirm
6. **Smoke run** — `--debug` mode, 4 corruption × 5 batch × 1 round with `monitor_interval=5`. Expected: finite mIoUs, at least one mode update logged, no crash
7. **Degeneracy check (optional)** — `sam_rho=0, h_threshold=999, h_warning=-1, cautious_rst=0, brake_rst=0` → trajectory should match `tent_continual` bit-for-bit modulo the reliability filter
8. **Determinism vs `sar_continual` (optional)** — `h_threshold=-1, h_warning=-1, cautious_rst=0, brake_rst=0` (gate effectively off) → trajectory should match `sar_continual` with `e_0 = -inf` (recovery disabled)

---

## 13. Success Criteria

| Tier | Criterion | Meaning |
|---|---|---|
| Basic | R150 mean ≥ 28, no W-shape dip below 25 | DivGate successfully replaces SAR's recovery |
| Target | Overall mean ≥ 31.59 | Matches TENT-DivGate (SAM didn't hurt) |
| **Ideal** | **Overall mean ≥ 32.5 with peak ≥ 33** | **SAM lifts the peak, DivGate holds it — research win** |
| Stretch | Mean ≥ 33.0 with R150 ≥ 32 | Full complementarity realized |

### 13.1 Interpretation paths (ACDC)

- **Basic fails (W-shape remains)**: DivGate thresholds too lenient for SAR's faster collapse. Raise `h_warning` to 1.5, or lower `monitor_interval` to 25 for faster reaction.
- **Basic passes, target fails (mean 28-31)**: SAM + DivGate over-constrains adaptation. Lower `cautious_rst` to 0.005, or raise `h_threshold` to 1.7 so aggressive mode persists longer.
- **Target passes, ideal fails**: SAM contributes minimally beyond TENT. Confirm via degeneracy check `sam_rho=0` — if mean drops noticeably, SAM helps; if not, sweep `sam_rho ∈ {0.02, 0.05, 0.1}`.
- **Ideal passes**: confirm reproducibility across seeds (seed 0, 1, 2). Log `[DivGate-S]` mode-transition timeline; commit `divgate_log.txt` to `save/ACDCDataset/sar_divgate_continual/`.

### 13.2 VOC20 / Cityscapes expectations

Headroom-limited (synthetic corruptions, see CLAUDE.md). Targets are relative:

- **VOC20 weather**: SAR alone hits ~67 (No Adapt 70.79). Goal: ≥ 70 (closes the gap to No Adapt), peak ≥ 71. TENT-DivGate hits ~57 here, so SAM contribution is expected to be largest on this dataset.
- **Cityscapes weather**: no SAR-base baseline yet. Run SAR-alone first (bash script just landed), use that as the comparison point.

---

## 14. Out of Scope

- **MLMP-base variant** (`mlmp_sar_divgate_continual`) — would multiply the design surface; defer until TENT-base SAR-DivGate has a confirmed result
- **EATA-DivGate hybrid** — different anti-forgetting structure (Fisher EWC), separate experiment
- **`sam_rho` sweep / `e_margin` sweep** — defer until base behavior is characterized at defaults
- **Two-signal gate** (use both loss_ma AND H_margin to pick mode) — would re-introduce the SAR EMA logic we just deleted; defer until single-signal version shows a clear bottleneck
- **Per-batch H_margin logging** — `divgate_log.txt` already covers per-monitor logging; per-batch is needed only if calibration matters
- **EATA-style condition-aware Fisher reweighting** — orthogonal to this design

---

*End of design.*
