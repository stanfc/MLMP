# AdaGate — self-calibrating trigger and lag for GDG-PA

**Method**: `deyo_mlmp_adagate_continual` ([adapt/deyo_mlmp_adagate_continual.py](../adapt/deyo_mlmp_adagate_continual.py))
**Base**: GDG-PA = `deyo_mlmp_hmgate2_continual` (current best; ACDC mean 31.06, VOC20 mean 77.11)
**Sweep**: [bash/sweep_adagate.sh](../bash/sweep_adagate.sh) · **Collector**: [scripts/collect_adagate.py](../scripts/collect_adagate.py)
**Status**: implemented, `ctrl` equivalence verified bit-exact, 11-arm sweep launched 2026-08-02.

---

## 1. Problem — two hyperparameters that do not do what they claim

GDG-PA's gate makes three decisions every `monitor_interval=50` batches (one *window*):
whether to restore (**trigger**), how far back to restore (**lag**), and how much
(`base_rst=0.01`). The first two are controlled by absolute constants, and measurement on
`save/*/hmgate2_prompt_S0_baseline/gate_log.csv` shows **neither is doing its stated job**.

### (A) `slope_deadzone = 0.002` — a deadzone that filters nothing

Intent: reject noise-level rises of the windowed grad-norm before triggering restoration.
The gate computes `slope` = OLS slope of the last `slope_window=10` window-mean grad-norms,
in units of grad-norm per window.

| | grad_norm median | slope p10 | slope p90 | deadzone as % of grad_norm |
|---|---|---|---|---|
| ACDC | 5.25 | −0.027 | +0.030 | 0.038 % / window |
| VOC20 | 6.64 | −0.347 | +0.299 | 0.030 % / window |

The slope noise band is 15× (ACDC) to 150× (VOC20) larger than the deadzone, so the test
degenerates to `slope > 0`:

```
P(slope > 0.002) = 0.463 (ACDC), 0.507 (VOC20)      vs      P(slope > 0) ≈ 0.5
```

It is also an absolute constant applied to two slope distributions that differ by 10× in
scale — it transfers across datasets only because it is too small to bind.

### (B) `lag_gain = 1500` — a continuous lag that is binary in practice

Intent: `lag = round(lag_gain · slope)`, clamped to `cap = min(maxlag_shallow=6,
windows_since_min)`, so that faster degradation restores to an older snapshot.

Saturation requires only `slope > 6/1500 = 0.004`, which holds for **93.2 %** (ACDC) and
**99.7 %** (VOC20) of active windows. The realised lag histogram is binary:

```
ACDC  {0: 322, 6: 162, other: 16}
VOC20 {0: 741, 6: 709, other: 50}
```

Spreading lag over 1..6 would need `lag_gain ≈ 153` (ACDC) vs `≈ 17` (VOC20) — a 9× gap, so
**no single absolute gain can work**. (In the early windows the graded lags 1..6 that do
appear come from the `cap = windows_since_min` clamp, not from `lag_gain`.)

---

## 2. Fix — both decisions become unitless functions of the stream's own statistics

### (A) `trend_stat` — normalise the slope before thresholding

| mode | z | notes |
|---|---|---|
| `abs` | `slope` | GDG-PA behaviour; deadzone is `slope_deadzone` |
| `rel` | `slope / grad_norm` | fractional growth — **insufficient**, see below |
| `mad` | `slope / (1.4826 · MAD(recent slopes))` | robust z vs the signal's own spread |
| `tstat` | `slope / SE(slope)` | OLS significance of the upward trend |

The MAD scale is estimated over `trend_hist = 50` preceding windows; `z` is always scored
against windows *before* the current one.

Offline replay of the S0_baseline grad-norm streams (exact — reconstructed slope matches the
logged value to 5.5e-7) gives:

| statistic | ACDC p75 / p90 | VOC20 p75 / p90 | transfers? |
|---|---|---|---|
| `rel` | 0.003 / 0.005 | 0.025 / 0.042 | **no** — still 8× apart |
| `tstat` | 0.451 / 0.869 | 1.187 / 2.246 | partly — 2.6× apart |
| `mad` | **0.635 / 1.125** | **0.656 / 1.114** | **yes — distributions coincide** |

`rel` fails because it normalises the slope's *magnitude* but not its *noise level*, which is
what differs between the datasets. Under `mad`, one unitless `trend_thr` yields the same
firing rate on both (`thr=0.5` → 0.305 / 0.311; `thr=1.0` → 0.135 / 0.139), so `trend_thr`
is a genuine signal-to-noise deadzone rather than a scale-bound constant.

### (B) `lag_mode` — restore depth as a fraction of the budget

```
cap = min(maxlag_shallow, windows_since_min)
lag = max(1, ceil(cap · u))
```

| mode | u | tunables |
|---|---|---|
| `gain` | (GDG-PA: `lag = round(lag_gain·slope)`) | `lag_gain` |
| `sat` | `clamp(z / lag_sat, 0, 1)` | `lag_sat` |
| `ecdf` | rank of `z` among the last `trend_hist` **active** z | **none** |

`ecdf` is rank-based, hence invariant to any monotone rescaling of the trend statistic — it
is scale-free regardless of which `trend_stat` feeds it, and it deletes `lag_gain` outright.
Replayed lag histograms (vs GDG-PA's `{0, 6}`):

```
ACDC  mad0.5 + ecdf : {1:20, 2:24, 3:26, 4:21, 5:15, 6:12}
VOC20 mad0.5 + ecdf : {1:81, 2:75, 3:90, 4:77, 5:65, 6:78}
```

### Reduction to the base method

`trend_stat=abs, trend_thr=slope_deadzone, lag_mode=gain` consumes no extra RNG and
reproduces GDG-PA exactly. **Verified**: a 3-round ACDC `ctrl` run matched
`hmgate2_prompt_S0_baseline` on every per-condition mIoU digit (R1 29.21 / R2 29.25 / R3
29.28) and on all 12 gate windows (`grad_norm`, `grad_slope`, `h_margin`, `collapse`,
`windows_since_min`, `lag`, `rst`, `deep`), with `z == grad_slope`.

Everything else — DeYO+PLPD loss, MLMP/UAML evaluation, the H_margin regime switch
(`h_drop_ratio=0.9`, already relative to the running-max H), the permanent best-state anchor,
and `base_rst` — is unchanged.

---

## 3. Sweep design

Testbeds are byte-identical to `bash/sweep_prompt.sh`, so `hmgate2_prompt_S0_baseline` is a
directly comparable same-machine reference. ACDC (sub50, 4 conditions) exercises both regimes
(collapse fires in 48.5 % of windows); VOC20 (`v20_acdc_matched`, sub100, 5 corruptions) never
enters the collapse regime, so **100 % of its restores go through the lag path** — it is the
clean testbed for (B).

| arm | trend_stat | trend_thr | lag_mode | isolates |
|---|---|---|---|---|
| `ctrl` | abs | 0.002 | gain | equivalence check |
| `Amad0` | mad | 0.0 | gain | (A) normalisation alone, firing matched |
| `Amad05` | mad | 0.5 | gain | (A) + a real SNR deadzone |
| `Amad10` | mad | 1.0 | gain | (A) + strict deadzone |
| `Atstat10` | tstat | 1.0 | gain | (A) significance-test variant |
| `Becdf` | abs | 0.002 | ecdf | (B) alone, trigger untouched |
| `ABmad0_ecdf` | mad | 0.0 | ecdf | A+B, firing matched |
| `ABmad05_ecdf` | mad | 0.5 | ecdf | **flagship — zero absolute constants** |
| `ABmad10_ecdf` | mad | 1.0 | ecdf | A+B strict |
| `ABtstat10_ecdf` | tstat | 1.0 | ecdf | A+B fully statistical |
| `ABmad05_sat` | mad | 0.5 | sat 1.5 | A+B saturating depth |

Launched wave 1 (11 runs, 150R): ACDC `{Amad0, Amad05, Amad10}` + `{Becdf, ABmad05_ecdf,
ABmad10_ecdf, ABtstat10_ecdf}`, VOC20 `{Becdf, ABmad0_ecdf, ABmad05_ecdf, ABmad05_sat}`.
Deferred to wave 2: ACDC `Atstat10`, `ABmad0_ecdf`, `ABmad0_sat`, `ABmad05_sat`.

---

## 3b. RESULTS — both waves complete (19 runs × 150R, 3 datasets)

Reference on ACDC/VOC20 is `hmgate2_prompt_S0_baseline`; on Cityscapes it is the `ctrl`
arm (bit-equivalent to GDG-PA), since no same-machine GDG-PA Cityscapes run existed.

| arm | ACDC mean (Δ) | VOC20 mean (Δ) | Cityscapes mean (Δ) |
|---|---|---|---|
| GDG-PA / ctrl | 31.06 | 77.11 | 23.85 |
| `Becdf` (B only) | 31.02 (−0.04) | **77.56 (+0.45)** | — |
| `Amad0` (A only, firing matched) | 31.03 (−0.03) | — | — |
| **`ABmad05_ecdf`** | **31.31 (+0.25)** | **77.80 (+0.69)** | **23.97 (+0.12)** |
| `ABmad10_ecdf` | 31.68 (+0.62) | 77.90 (+0.79) | **23.74 (−0.10)** |
| `ABmad15_ecdf` | 31.81 (+0.74) | — | — |
| `ABmad20_ecdf` | 32.12 (+1.06) | — | — |
| `ABtstat10_ecdf` | 32.07 (+1.01) | 77.84 (+0.73) | 23.91 (+0.07) |

**(B) is fixed.** `graded` goes from 0.02–0.07 (all three references) to 0.60–0.83. `Becdf`
— B alone, trigger untouched — earns +0.45 on VOC20, where 100 % of restores take the lag
path, so the gain is attributable to B alone. On ACDC, B alone does nothing (−0.04) and A
alone at matched firing does nothing (−0.03): **the two fixes are complementary, one per
regime**, matching the project's existing collapse-vs-uniform-degradation split.

**(A) shows the base method was over-restoring.** All three references fire in 0.46–0.51 of
windows with `graded` ≈ 0 — the same pathology reproduces independently on a third dataset.

**Transferability — the decisive result.** Firing rate under one unitless threshold:

| threshold | ACDC | VOC20 | Cityscapes | spread | verdict |
|---|---|---|---|---|---|
| `mad 0.5` | 0.292 | 0.315 | 0.299 | **1.1×** | transfers |
| `mad 1.0` | 0.167 | 0.155 | 0.105 | 1.6× | marginal |
| `tstat 1.0` | 0.037 | 0.251 | 0.296 | **8.0×** | **does not transfer** |

`tstat`'s ACDC win (+1.01) came from *accidentally firing at 3.7 % on ACDC*, not from the
statistic being better: once VOC20/Cityscapes push its firing to 0.25–0.30, it performs the
same as `mad 0.5` (77.84 vs 77.80; 23.91 vs 23.97). **The statistic choice is not the lever
— cross-dataset firing-rate stability is.**

**`ABmad05_ecdf` is the recommended configuration** — the only arm positive on all three
datasets with stable tails (peak→last −0.25 / −0.35 / −0.05). `mad 1.0` scores higher on
ACDC and VOC20 *mean* but is **negative on Cityscapes** (−0.10) and decays −1.19 from peak
on VOC20 (78.89@R84 → 77.70@R150). The ACDC-only monotone "lower firing is better" trend
(down to 3.5 % / +1.06) is **dataset-specific**: the inverted-U turns much earlier on
Cityscapes.

**Not yet done**: all runs are seed=0; `trend_hist` has never been swept.

---

## 4. What counts as success

The primary claim is **methodological**, not a leaderboard delta: two absolute, dataset-bound
constants are replaced by unitless quantities derived online from the stream, and one of them
(`lag_gain`) is deleted entirely. So:

- **Sufficient**: any arm matches GDG-PA's mean within noise on both datasets while removing
  `slope_deadzone` and `lag_gain`. The lag histogram must show genuine mass on 2..5 (reported
  as `graded` by the collector) — otherwise (B) is still inert and nothing was fixed.
- **Bonus**: a `trend_thr > 0` arm beats GDG-PA. GDG-PA restores in ~50 % of windows because
  its deadzone is degenerate; a real deadzone (`thr=0.5` → ~30 %, `thr=1.0` → ~14 %) leaves
  more windows free to climb, which may raise the peak on ACDC where the curve is still
  rising at R150.
- **Negative but publishable**: if higher `trend_thr` monotonically degrades both datasets,
  that is evidence the ~50 % firing rate is load-bearing — GDG-PA's restoration is doing
  continuous mild regularisation rather than event-driven repair, which is itself a finding
  worth stating.

---

## 5. Notes / gotchas

- `--trend_thr` is **ignored** when `trend_stat=abs` (that mode uses `--slope_deadzone`), so
  the `ctrl` arm cannot be perturbed by a stray `trend_thr`.
- Warm-up: `mad` needs 5 preceding slopes and `ecdf` needs 10 active z. Before that, `mad`
  returns ±10 for a non-zero slope (fires) and `ecdf` returns `u = 0.5`. This affects ~5–10
  windows out of 600 (ACDC) / 1500 (VOC20) and matches the offline replay.
- `gate_log.csv` gains two columns (`z`, `u`) between `grad_slope` and `h_margin`, so any
  plotting script that indexes gate-log columns positionally must be updated.
- GPU 4 on this machine was taken over mid-launch by another user's 82 GB VLLM engine; the
  ACDC arms were moved to GPU 3. Each ACDC run needs ~13 GB, each VOC20 run ~3 GB.
