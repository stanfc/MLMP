# TENT-DivGate-Continual: Diversity-Gated TENT

**Status**: Design approved, implementation pending
**Created**: 2026-04-26
**Owner**: Tekai-Yen
**Related docs**: `docs/cma_divgate_continual_spec.md` (sibling design, CMA-based), `proposal_after_cma.md` (Direction B)

---

## 1. Motivation: Why TENT, Not CMA

The CMA-DivGate experiment (`save/ACDCDataset/cma_divgate_continual/`) successfully demonstrated that the H_margin gate prevents the dead-state collapse that plain CMA-continual suffers (mean 21.21 vs CMA's 5.54). However, the absolute level fell short:

| Method | Peak | Round | R150 | Overall mean |
|---|---|---|---|---|
| **TENT-continual step=10** | **32.44** | R2 | 5.10 | crashes |
| TENT-continual step=1 | 30.92 | R10 | crashes | crashes |
| MLMP-episodic step=10 (target) | 30.6 | — | — | 30.6 |
| CMA-continual k=0.5 | 27.40 | R18 | 0.84 | 6.65 |
| CMA-DivGate (default) | 26.82 | R16 | 19.32 | 21.21 |

The CMA family's peak is **structurally lower than TENT's peak**: 27.40 vs 32.44. No anti-collapse brake mechanism can produce a higher mean than the base loss's peak — the brake can only stabilize, not amplify. So CMA-DivGate is bounded above by ~27.

**TENT-continual step=10 already exceeds MLMP-episodic before crashing**. If the same H_margin gate that stabilized CMA-continual can stabilize TENT-continual at a similar fraction of peak, the mean might stay in the 28-32 range — and that would beat the episodic upper bound.

This is the key remaining hypothesis test: **can a diversity-gated brake hold TENT in its high-plasticity regime?**

---

## 2. Design Principle

Reuse the diversity gate from `cma_divgate_continual` verbatim. Replace the base loss with **pure pixel-wise softmax entropy** (TENT). Restoration is flat (single rate per gate-state, applied to every visual-encoder LayerNorm parameter), as in `cma_divgate`.

The base loss is **not** masked by Top-K confidence: TENT minimizes entropy across all spatial positions and the batch (matching `adapt/tent_continual.py`). Adding a Top-K filter would introduce a confounding variable that obscures the comparison between CMA-DivGate (capped at 27) and TENT-DivGate (target: 30+).

---

## 3. Loss

Identical to `adapt/tent_continual.py`:

```python
@staticmethod
def softmax_entropy(x):
    # x: (T=1, B, C, w, h) — class dim is -3
    return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

# inside perform_adaptation:
logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
loss = self.softmax_entropy(logits).mean()
```

`logits` has shape `(T, B, C, w, h)` with `T=1` (single prompt template by default — matches `tent_continual`'s convention).

---

## 4. Diversity Gate

Bit-for-bit identical to `cma_divgate_continual_spec.md` §2-§3. For ease of reading:

### 4.1 Per-batch buffer push (inside the existing `no_grad` block adjacent to loss)

```python
with torch.no_grad():
    probs = logits[0].softmax(dim=1)             # (B, C, w, h)
    batch_marginal = probs.mean(dim=[0, 2, 3])   # (C,)
    self.marginal_buf.append(batch_marginal.detach().float().cpu())
```

`logits[0]` removes the prompt dim; `T=1` makes this safe and equivalent to "prompt-averaged" for the multi-prompt case.

### 4.2 Mode update every `monitor_interval` batches

```python
agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
agg = agg / agg.sum().clamp(min=1e-8)
h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

new_mode = pick_mode(h_margin, h_threshold, h_warning)
if new_mode != self.current_mode:
    print(f"[DivGate-T] B{total_batches}: H_margin={h_margin:.3f}  {old} -> {new}")
self.current_mode = new_mode
self.current_rst  = mode_to_rst(new_mode)
self.marginal_buf.clear()
self.batch_count = 0
```

### 4.3 Three-tier mode → restoration rate

```
H_margin >= h_threshold              → "aggressive"  (rst = 0)
h_warning <= H_margin < h_threshold  → "cautious"    (rst = cautious_rst)
H_margin < h_warning                  → "brake"       (rst = brake_rst)
```

Defaults: `h_threshold=1.8`, `h_warning=1.2`, `cautious_rst=0.005`, `brake_rst=0.05`. Initial mode is `aggressive`. Restoration is flat — same `_stochastic_restore_flat(rst)` helper used in `cma_divgate`.

---

## 5. Adapt Step

```
forward(x) → logits  (1, B, C, w, h)

loss = softmax_entropy(logits).mean()
   in no_grad: append batch_marginal to self.marginal_buf

loss.backward(); optimizer.step(); optimizer.zero_grad()

if self.current_rst > 0:
    _stochastic_restore_flat(self.current_rst)

batch_count += 1
total_batches += 1
if batch_count >= monitor_interval:
    _update_mode()
```

The only difference from `cma_divgate.perform_adaptation` is the loss: pure entropy instead of CMA cosine alignment.

---

## 6. Public API

```python
TENTDivGateContinual(
    ovss_type, ovss_backbone, lr, classes,
    steps=1,
    h_threshold=1.8, h_warning=1.2,
    monitor_interval=50,
    cautious_rst=0.005, brake_rst=0.05,
    prompt_dir=None,
    runtime_calculation=False,
    device='cpu',
)

.adapt(x) / .continual_adapt(x) / .evaluate(x) / .reset()
```

No `obtain_src_*` hook. No `top_k_percent` parameter (deliberate — pure TENT).

---

## 7. Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `lr` | `1e-5` | Match `tent_continual.sh` |
| `steps` | `1` | Online CTTA. step=10 reaches peak earlier (R2 vs R10) but crashes faster — try if step=1 result is promising |
| `h_threshold` | `1.8` | aggressive boundary |
| `h_warning` | `1.2` | cautious / brake boundary. **Empirical caveat**: in CMA-DivGate the gate never triggered brake mode in 30+ rounds at default thresholds; if the same is true here, the gate behaves as a 2-tier (aggressive / cautious) controller. Add per-batch H_margin logging in a follow-up if calibration matters. |
| `monitor_interval` | `50` | Batches between H_margin checks |
| `cautious_rst` | `0.005` | Mid restoration rate |
| `brake_rst` | `0.05` | Strong restoration rate |
| `continual_rounds` | `150` | Match other DivGate runs |

### 7.1 Degradation paths

- `h_threshold = 999` → permanently aggressive → equivalent to plain `tent_continual` (collapses ~R80 at step=1)
- `cautious_rst = brake_rst = 0` → no restoration ever, gate becomes diagnostic-only
- `cautious_rst = brake_rst = 0.05` → CoTTA-style flat restore, gate irrelevant

---

## 8. CLI Surface

```python
elif method == 'tent_divgate_continual':
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

## 9. Bash Script

`bash/ACDC_10_round/tent_divgate_continual.sh` — based on `cma_divgate_continual.sh`, with the loss-specific knobs removed (no `TOP_K_PERCENT`).

---

## 10. Sanity Checks

H_margin formula and mode-picker logic are unchanged from `cma_divgate` and were already validated. New/repeated checks:

1. **`named_ln_params` count = 100** on ViT-L/14 (24 blocks × 2 LN × 2 + ln_pre + ln_post = 100)
2. **Initial mode = aggressive, current_rst = 0**
3. **Loss = TENT entropy** — verify by computing on uniform vs concentrated logits and checking expected sign/magnitude
4. **Smoke run** — debug-mode 4-condition × 5-batch × 1-round with `monitor_interval=5`, finite mIoUs, no crash
5. **Determinism check (optional)** — with `h_threshold=999, h_warning=-1, cautious_rst=brake_rst=0`, the trajectory should match `tent_continual` bit-for-bit (gate becomes a no-op)

---

## 11. Success Criteria

| Tier | Criterion | Meaning |
|---|---|---|
| Basic | R150 mean mIoU ≥ 15 | Gate prevents catastrophic dead state, like cma_divgate did |
| Target | Overall mean ≥ 25 with peak ≥ 30 | Reaches the high-plasticity regime and stays alive |
| **Ideal** | **Overall mean ≥ 30.6** | **Beats MLMP-episodic step=10 — the research goal** |
| Stretch | Peak ≥ 32 with R150 ≥ 28 | Captures most of TENT's natural ceiling without losing it |

### 11.1 Interpretation paths

- **All tiers fail (still collapses)**: gate is too lenient — TENT crashes faster than CMA, the default thresholds may not catch it. Lower `monitor_interval` (e.g. 20) or raise `h_warning` (e.g. 1.6).
- **Basic passes, target fails (mean ≤ 25)**: gate clamps too hard, model can't reach high regime. Lower `cautious_rst` and `brake_rst`. Or relax `h_threshold` (e.g. 1.6) so aggressive mode persists longer.
- **Target passes, ideal fails**: gate captures some of TENT's plasticity; tune around (h_threshold, monitor_interval) to extend the aggressive window. Consider step=10 for higher peak.
- **Ideal passes**: this is the headline result — log the gate's mode-transition timeline and make sure it's reproducible.

---

## 12. Out of Scope

- step=10 default — covered by changing `STEPS` in the bash script; not a new method
- top_k masking on TENT — would muddy the comparison with CMA-DivGate and pure TENT
- Combining DivGate with layer-stratified restoration — separate experiment if A is revisited
- Per-batch H_margin logging — useful for calibration but not load-bearing for this experiment

---

*End of design.*
