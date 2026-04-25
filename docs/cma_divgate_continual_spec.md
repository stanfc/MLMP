# CMA-DivGate-Continual: Diversity-Gated Cross-Modal Alignment

**Status**: Design approved, implementation pending
**Created**: 2026-04-25
**Owner**: Tekai-Yen
**Related docs**: `proposal_after_cma.md` (Direction B), `docs/cma_continual_spec.md` (CMA baseline), `docs/cma_layered_continual_spec.md` (Direction A, parallel design)

---

## 1. Motivation

CMA-continual demonstrated genuine adaptation but collapsed at R30–40 because confirmation bias is unbounded. CoTTA never collapses but also never improves past source. CMA-Layered (Direction A) traded plasticity for stability and ended up flat at baseline. None of these find the bridge.

The key observation in `proposal_after_cma.md §2`: **collapse has a label-free early-warning signal**. Before the model locks into predicting 1–2 dominant classes, the batch-level marginal class distribution becomes increasingly concentrated. This is observable without ground truth and several rounds before catastrophic mIoU drop.

Direction B (Diversity-Gated CMA) exploits this: keep CMA's aggressive adaptation when the model is healthy, but apply CoTTA-style stochastic restoration only when the diversity signal indicates impending collapse. This is a **state-dependent brake**: zero overhead in the healthy regime, automatic intervention when needed.

---

## 2. The Diversity Metric

### 2.1 Marginal Class Entropy

For prompt-averaged logits over a batch:

```
probs    = softmax(avg_logits, dim=1)        # (B, C, w, h)
marginal = probs.mean(dim=[0, 2, 3])         # (C,) — average prob per class
H_margin = -Σ_c marginal[c] · log(marginal[c])
```

Reference values for the ACDC 19-class label set:
- `H_margin = log(19) ≈ 2.94` — perfectly uniform predictions
- `H_margin = 0` — every pixel predicted as a single class (dead state)
- `H_margin ≈ 1.5–2.0` — healthy diverse predictions
- `H_margin < 1.0` — predictions concentrating on 1–2 classes (early collapse signal)

This is **not** per-pixel entropy (which is what TENT minimizes and is the cause of collapse). It is the entropy of the **marginal class distribution** averaged over a batch — a measure of how diverse the predictions are *across* pixels and samples.

### 2.2 Buffer-N Aggregation

Per `proposal_after_cma.md §2.3`, H_margin is recomputed every N samples (default N = 50). The implementation:

1. On every adapt batch: append `marginal = probs.mean(dim=[0, 2, 3])` (shape `(C,)`) to a buffer.
2. After N batches: aggregate `agg_marginal = mean(buffer)`, re-normalize to a distribution, compute `H_margin`, decide mode, **clear buffer**, reset counter.
3. The chosen mode persists for the next N batches.

Aggregating marginals first then taking entropy (rather than per-batch entropy then averaging) is intentional — this is the entropy of the **average prediction distribution** across N samples, which is what diversity actually means at this scale.

---

## 3. Three-Tier Brake Mode

```
H_margin >= h_threshold              → "aggressive"  (current_rst = 0)
h_warning <= H_margin < h_threshold  → "cautious"    (current_rst = cautious_rst)
H_margin < h_warning                  → "brake"       (current_rst = brake_rst)
```

Defaults: `h_threshold=1.8`, `h_warning=1.2`, `cautious_rst=0.005`, `brake_rst=0.05`.

`current_rst` is a single scalar restoration probability applied uniformly to every visual-encoder LayerNorm parameter (γ, β) — flat, not layer-stratified. Same masking pattern as CoTTA's stochastic restore.

**Initial mode**: `aggressive` (rst = 0). The first `monitor_interval` batches run with no restoration so the gate has data to evaluate before intervening.

**Mode transitions are logged**: `[DivGate] B{total_batches}: H_margin={value:.3f}  {old_mode} → {new_mode}`. Same-mode windows produce no log line — only transitions.

---

## 4. Adapt Step

```
forward(x) → logits, image_features, text_features

cma_loss(...) → scalar loss
   inside: in no_grad branch, append batch_marginal to self.marginal_buf

loss.backward(); optimizer.step(); optimizer.zero_grad()

if self.current_rst > 0:
    _stochastic_restore_flat()        # CoTTA-style, single rate

self.batch_count += 1
self.total_batches += 1
if self.batch_count >= self.monitor_interval:
    self._update_mode()               # compute H_margin, switch mode, reset buffer
```

CMA loss tensor ops are copied verbatim from `adapt/cma_continual.py` — same Top-K% mask, same prompt-averaging, same `image_features[:, 1:, :]` CLS handling. The only addition inside `cma_loss` is one line in the existing `no_grad` block to compute and buffer the per-batch marginal.

---

## 5. Public API

```python
CMADivGateContinual(
    ovss_type, ovss_backbone, lr, classes,
    steps=1,
    top_k_percent=0.2,
    h_threshold=1.8,
    h_warning=1.2,
    monitor_interval=50,
    cautious_rst=0.005,
    brake_rst=0.05,
    prompt_dir=None,
    runtime_calculation=False,
    device='cpu',
)

.adapt(x)              # adaptation (no reset)
.continual_adapt(x)    # alias for adapt — main_continual.py protocol
.evaluate(x)           # torch.no_grad inference
.reset()               # reload source snapshot (episodic only; not used in CTTA)
```

No `obtain_src_*` pre-stream hook needed.

---

## 6. Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `lr` | `1e-5` | Match `cma_continual` |
| `steps` | `1` | Online CTTA |
| `top_k_percent` | `0.2` | Match `cma_continual` |
| `h_threshold` | `1.8` | Tunable via bash. Higher → more time in aggressive (riskier) |
| `h_warning` | `1.2` | Tunable via bash. Lower → harder to trigger brake |
| `monitor_interval` | `50` | Batches per H_margin re-evaluation. Lower → more responsive but noisier |
| `cautious_rst` | `0.005` | Mid restoration probability |
| `brake_rst` | `0.05` | Strong restoration probability (matches CoTTA default) |
| `continual_rounds` | `150` | Match CMA baseline |

### 6.1 Degradation paths

- `h_threshold = 999` → permanently aggressive → equivalent to `cma_continual`. Sanity baseline.
- `h_threshold = -1` (always below threshold) and `h_warning = -1` → both modes never triggered, gate effectively off.
- `cautious_rst = brake_rst = 0` → gate logs mode changes but never restores → equivalent to `cma_continual` with diagnostics.
- `cautious_rst = brake_rst = 0.05` → CoTTA-style flat restore, gate becomes irrelevant. Useful as a baseline for what flat-rst CMA looks like.

---

## 7. CLI Surface

Added to `main_continual.py::add_method_specific_args`:

```python
elif method == 'cma_divgate_continual':
    parser.add_argument('--top_k_percent', type=float, default=0.2,
                        help='Top-K%% confidence mask for the CMA loss (default 0.2)')
    parser.add_argument('--h_threshold', type=float, default=1.8,
                        help='H_margin >= this -> aggressive mode (rst=0)')
    parser.add_argument('--h_warning', type=float, default=1.2,
                        help='h_warning <= H_margin < h_threshold -> cautious; '
                             '< h_warning -> brake')
    parser.add_argument('--monitor_interval', type=int, default=50,
                        help='Batches between H_margin re-evaluations (default 50)')
    parser.add_argument('--cautious_rst', type=float, default=0.005,
                        help='Stochastic restore probability in cautious mode')
    parser.add_argument('--brake_rst', type=float, default=0.05,
                        help='Stochastic restore probability in brake mode')
```

---

## 8. Bash Script

`bash/ACDC_10_round/cma_divgate_continual.sh` — based on the CMA / Layered scripts:

```bash
TOP_K_PERCENT=0.2

# Diversity gate thresholds (proposal_after_cma.md §2.3 defaults)
H_THRESHOLD=1.8       # aggressive boundary
H_WARNING=1.2         # cautious boundary

# Monitor cadence and brake strengths
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.005
BRAKE_RST=0.05

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"
```

Full script wires all knobs to `python main_continual.py` flags.

---

## 9. Sanity Checks (manual, run before the full experiment)

1. **H_margin formula correctness**: hand-construct three marginal distributions and check
   - uniform `[1/19] * 19` → `H ≈ log(19) ≈ 2.944`
   - one-hot `[1, 0, ...]` → `H ≈ 0` (with the `clamp(min=1e-12)` guard)
   - 50/50 two-class `[0.5, 0.5, 0, ...]` → `H ≈ log(2) ≈ 0.693`
2. **Mode picker**: feed H values 2.0, 1.5, 0.5 with default thresholds → expect `aggressive`, `cautious`, `brake`.
3. **Buffer accounting**: simulate 50 marginal pushes → buffer length 50; after `_update_mode()` → length 0, `batch_count = 0`.
4. **Initial state**: a freshly-constructed model should have `current_mode == 'aggressive'` and `current_rst == 0.0`.
5. **`named_ln_params` count = 100 on ViT-L/14**: matches the layered method's check (24 blocks × 2 LN × 2 params + ln_pre + ln_post = 100).
6. **Smoke test**: debug-mode `--debug` run, 4 conditions × 5 batches × 1 round, completes with finite mIoUs. `monitor_interval=50` will likely not fire in debug mode (only 20 batches total), which is fine — verifies the gate path doesn't break the basic adapt loop. For a fast gate-firing test, override `--monitor_interval 5` in the smoke command.

---

## 10. Success Criteria

| Tier | Criterion | Meaning |
|---|---|---|
| Basic | R150 mean mIoU ≥ 15 | Avoids the dead state |
| Target | Overall mean ≥ 25 with at least one observed brake transition | Gate genuinely activates and produces a stable mid-range result |
| Ideal | Peak ≥ 27 (matches CMA peak) AND R150 ≥ 25 | Captures CMA's adaptation gain without losing it |
| Stretch | Overall mean ≥ 30.6 | Beats MLMP-episodic (step=10) — research goal |

### 10.1 Diagnostic outputs to inspect post-run

- `[DivGate]` mode-transition log lines — count, timing within each round, which conditions tend to trigger them.
- Round-by-round mIoU trajectory vs CMA baseline — does the curve diverge from CMA at the same point CMA peaks (which is when we'd expect first brake)?
- If the mIoU collapses without any brake transition firing → thresholds are too low or `monitor_interval` is too long. Tune.

### 10.2 Interpretation paths

- **All tiers fail (still collapses)**: `h_warning` is too low — collapse signal arrives later than thresholds expect. Raise `h_warning` to 1.5, or lower `monitor_interval` to 20–30 for faster reaction.
- **Basic passes, target fails (stable-low)**: gate triggers too aggressively, restoration is too strong. Lower `cautious_rst`/`brake_rst`, or raise `h_threshold` to keep aggressive mode longer.
- **Target passes, ideal fails**: balance is decent but plasticity is bottlenecked. Try a 4-tier gate (separate `h_danger`) or combine with Direction A.
- **Ideal passes**: log the threshold sweep and report.

---

## 11. Out of Scope

- A+B combined (layer-stratified + diversity-gated) — separate spec, planned.
- Brake-mode loss/gradient scaling (`proposal_after_cma.md §2.3` mentions "gradient reduced" but YAGNI — restoration alone is the brake; revisit only if brake is empirically too weak).
- Adaptive thresholds (auto-calibrating `h_threshold` from the early-round H_margin distribution) — interesting but adds another control loop. Defer.
- Logging per-batch `H_margin` for plotting — easy to add later, not load-bearing for the experiment.

---

*End of design. Implementation plan to be produced by `writing-plans`.*
