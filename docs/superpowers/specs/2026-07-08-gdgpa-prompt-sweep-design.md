# GDG-PA Prompt-Set Sweep — Design

**Date**: 2026-07-08
**Method under test**: GDG-PA = `deyo_mlmp_hmgate2_continual` (current best).
**Goal**: push GDG-PA performance by improving the TEXT PROMPTS only. Pure config
sweep — the ONLY variable is `--prompt_dir`; method + all gate hyperparameters are
pinned to GDG-PA's best config, so any mIoU delta is attributable to prompts alone.

## Why prompts

GDG-PA inherits MLMP's prompt machinery: adapt averages the DeYO+MLMP loss over the
N templates in the yaml; eval uses the mean of the N template embeddings (a single
classifier). The stock `prompts.yaml` has 7 CLIP-generic templates, several of them
OFF-domain for street/adverse-weather segmentation (`a origami {}`, `a {} in a video
game`, `art of the {}`). Hypothesis: domain-matched / de-noised prompts give a better
text anchor and lift mIoU — most plausibly the VOC20 peak (uniform-drift, anchor-
alignment-bound) and the pre-collapse quality on ACDC.

## Prompt sets (prompts_sweep/, each yaml header states its rationale)

Most sets fix N≈7 (matching baseline) so effects attribute to CONTENT, not count.
S1 (N=1) and S6 (N=12) deliberately vary the count axis.

| ID | N | Theme | Question it answers |
|----|---|-------|---------------------|
| S0_baseline | 7 | current prompts.yaml | control / reference |
| S1_minimal | 1 | one canonical `a photo of a {}.` | does the ensemble help at all? |
| S2_photographic | 7 | photographic-only (drop artistic) | are origami/videogame/art hurting? |
| S3_street | 7 | driving/urban-street domain | does domain framing tighten anchors? (off-domain for VOC) |
| S4_weather | 7 | adverse-weather (weather-agnostic union) | condition-aware anchors for collapse regime? |
| S5_street_weather | 7 | street × weather cross | best-of-both for ACDC |
| S6_rich | 12 | photographic+driving, larger ensemble | count axis: does more good templates help? |
| S7_generic | 7 | neutral photographic ensemble | clean generic control; natural in-domain set for VOC |
| S8_artistic | 7 | art/painting/sketch (NEGATIVE CONTROL) | proves the domain axis is real, not run noise |

## Protocol

- **Datasets**: ACDC (1120×560, conds fog/night/rain/snow) + VOC20 (224, v20_acdc_matched
  5 corruptions). Covers both regimes (collapse vs uniform-drift).
- **Screening**: `--subset_size 50` (ACDC) / `100` (VOC20), `--subset_seed 0`, full 150
  rounds kept so collapse dynamics remain visible. Winners get re-run at full size.
- **Config**: GDG-PA best gate params (slope_window 10, slope_deadzone 0.002, lag_gain
  1500, base_rst 0.01, h_drop_ratio 0.9, maxlag_shallow 6, monitor_interval 50), LR 5e-6,
  steps 1, batch 1, seed 0, 18-layer vision outputs. Same for every run.
- **Metrics** (per set × dataset): peak, peak-round, mean(R≥30) stable quality, R_last,
  collapse-onset round (first post-peak round < 0.6·peak). Δ vs S0_baseline.

## Tooling

- `prompts_sweep/S0..S8*.yaml` — the 9 prompt sets.
- `bash/sweep_prompt.sh <ds> "<ID...>"` — DRY per-dataset runner (GDG-PA, varies prompt_dir).
- `bash/run_prompt_queue.sh` — mem-gated queue (9×{acdc,v20}=18 jobs), greedily fills free
  VRAM with a BUFFER so it won't OOM the user's other jobs.
- `scripts/collect_prompt_sweep.py` — table + round-vs-mIoU trajectory overlay per dataset
  → `save/_compare/prompt_sweep_{ACDC,VOC20}.png`.
- Results: `save/{ACDCDataset,PascalVOC20Dataset/v20_acdc_matched}/hmgate2_prompt_<ID>/`.

## Follow-ups (deferred)

- If a large ensemble wins, decouple adapt vs eval prompts (small code change) so eval can
  use a rich ensemble (e.g. CLIP-80) without paying N× adapt cost.
- VOC-tailored domain set (S7 partially covers this) if driving sets confirm domain-specificity.
- Entropy-weighted prompt aggregation (UAML applied to the prompt axis) — the more original,
  loss-side lever discussed separately.
