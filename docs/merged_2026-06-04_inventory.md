# Merge Inventory — `origin/2026.06.04` (學長) merged into `tekai`

**Date merged:** 2026-06-04
**Merge commit:** `e1ce191` (then `6935d7f` adds einops to requirements)
**Fork point (last common ancestor):** `5861a05`

This file catalogs everything that came IN from 學長's `2026.06.04` branch that
was NOT on `tekai` before. Read this together with [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)
(our research arc) and the top of [../CLAUDE.md](../CLAUDE.md).

> ⚠️ **None of 學長's new methods are documented in EXPERIMENT_STATUS.md or
> CLAUDE.md's narrative.** They are code-only ports — there are no spec docs and
> we have not yet folded any of their results into our research story. Treat
> them as an available toolbox, not as validated findings.

---

## 0. Quick facts / gotchas

- **55 registered methods, 293 bash scripts.** The method count (`len(adapt.METHOD_CLASSES)`)
  is 55; scripts are far more numerous because each method has many
  hyperparameter-variant runners (ACDC_10_round alone has 116, v20 has 49).
  `adapt/*.py` = 57 files = 55 methods + 2 helpers (`sam.py`, `prompt_vit.py`).
- **New runtime dependency: `einops`** (used by `deyo_*` etc.). 學長 didn't add it
  to requirements; we did (`6935d7f`, pinned 0.8.2). `pip install einops` if a
  fresh env errors on `from einops import rearrange`.
- **`--severity` was removed, consolidated into `--corruption_severity`.** Both
  flags existed after the naive merge (ours vs theirs, same meaning). No bash
  script used `--severity`; 5 use `--corruption_severity`, so we kept theirs and
  rewired all call sites. If an old note says `--severity`, use `--corruption_severity`.
- **`.pyc` / `figures/` / `results*.{csv,tex}` are now untracked + gitignored.**
  `figures/main.jpg` exists locally (restored) but is gitignore'd — local-only,
  never pushed. `*.png/*.jpg/*.jpeg/*.svg` are globally ignored EXCEPT
  `utils/imagecorruptions/frost/*` (runtime dependency, whitelisted).

---

## 1. New backbone — CAT-Seg (`ovss/catseg/`)

A second OVSS backbone alongside `naclip`. Selected via `load_ovss('catseg', ...)`
(wired in `ovss/__init__.py`). Wrapper at `ovss/catseg/wrapper.py`
(`CATSegWrapper`); third-party CLIP/model code under `ovss/catseg/third_party/`
and `ovss/catseg/modeling/`. New CLI arg `--catseg_checkpoint`.
The method `tent_divgate_continual_catseg` runs TENT-DivGate on this backbone.
(Note: we had started CAT-Seg on `tekai` and reverted it; 學長's version is the
one that shipped.)

## 2. New datasets — ACDC class-merging (`utils/segmentation_datasets.py`)

- `ACDCMerged3Dataset` / `ACDCMerged6Dataset` / `ACDCMerged10Dataset` — ACDC
  images with the 19 Cityscapes classes collapsed into 3 / 6 / 10 OVSS-natural
  super-classes. Driven by `ACDC_{3,6,10}CLASS_REMAP` arrays + a `RemapLabels`
  pipeline step (placed after `ResizeAndPatchify`).
- Class-name files: `utils/class_extensions/acdc_{3,6,10}class.txt`.
- Purpose: test whether fewer/coarser classes give CTTA more headroom on ACDC.

## 3. New CLI args & data-pipeline features (`main*.py`, `prepare_data`)

- `--split` — ACDC only; `"val"` (default) or `"train+val"` (ConcatDataset of
  both splits per condition).
- `--subset_size` / `--subset_seed` — deterministic fixed-size subset per
  corruption (align update count across datasets of different val sizes).
- `--acdc_overlay_corruptions` — layer a SYNTHETIC ImageNet-C corruption on TOP
  of ACDC's real weather (real shift + synthetic noise), paired by index with
  `--corruptions_list`.
- `--corruption_severity` — ImageNet-C severity 1–5 (replaces our `--severity`).
- **Universal `entropy_log.csv`** — written for EVERY method, every batch, in
  both `main.py` and `main_continual.py`. Columns include `h_margin`,
  `h_pixel_mean`, `max_logit`, phase (`pre`/`post`). Plot via `plot_gate_entropy.py`.
- DataLoader now uses a **`spawn`** multiprocessing context + thread-limit
  `worker_init_fn` (avoids the 256-thread fork blowup on big machines).

## 4. New result-tooling (`scripts/`)

學長 moved/added parsing + LaTeX-table generators under `scripts/`:
`parse_results*.py` + `generate_latex_table*.py` with variants for
v20/cityscapes methods, mlmp-variants, catseg-methods, dpcore-variants,
backbone-comparison, plus `restore_results_from_csv.py`. Also `plot_results.py`,
`plot_gate_entropy.py` at repo root.

---

## 5. The 36 new methods (grouped by family)

### 5a. Literature baselines (new SOTA ports)
| method | paper | one-liner |
|---|---|---|
| `deyo_continual` | DeYO, ICLR 2024 Spotlight | entropy + PLPD (patch-shuffle disagreement) dual filter + reweighted entropy |
| `rotta_continual` | RoTTA, CVPR 2023 | CSTU category-balanced memory bank + EMA teacher-student + timeliness reweight |
| `dat_continual` | DAT, CVPR 2024 | grad-ranked selective param tuning (DSP/TRP) split by per-pixel uncertainty + EMA teacher |
| `kff` | KFF, NeurIPS 2025 | class-aware domain-knowledge Fusion & Fission (needs source stats, like DPCore) |

### 5b. TENT gating / anchor variants
| method | what's different from `tent_divgate_continual` |
|---|---|
| `tent_siggate_continual` | sigmoid-shaped H_margin→rst mapping (plateaus both ends) |
| `tent_contgate_continual` | continuous H_margin→rst mapping (no discrete modes) |
| `tent_early_continual` | only early-block LN trained, rest frozen (no gate/restore) |
| `tent_divgate_layered_continual` | hard-freeze late LN layers (`--early_cutoff/--late_cutoff`) |
| `tent_divgate_recent_anchor` | restore target = N batches ago (sliding window), not source |
| `tent_divgate_decoupled` | decouple `marginal_buf_size` from `monitor_interval` |
| `tent_divgate_hybrid_anchor` | cautious→recent anchor, brake→source anchor (discrete switch) |
| `tent_divgate_smooth_anchor` | **continuous anchor-depth** `lag(H)` (see §6) |
| `tent_divgate_continual_catseg` | TENT-DivGate on the CAT-Seg backbone |

### 5c. TENT loss variants
`tent_topk_continual` (entropy on top-K% conf pixels), `tent_minprompt_continual`
(multi-prompt min-margin), and their DivGate versions
`tent_topk_divgate_continual`, `tent_minprompt_divgate_continual`.

### 5d. MLMP loss / eval variants
- Loss filters: `mlmp_topk_continual`, `mlmp_minprompt_continual`,
  `mlmp_topk_minprompt_continual`.
- + DivGate: `mlmp_topk_divgate_continual`, `mlmp_minprompt_divgate_continual`,
  `mlmp_topk_minprompt_divgate_continual`.
- + SmoothAnchor: `mlmp_smooth_anchor_continual`, `mlmp_topk_smooth_anchor_continual`,
  `mlmp_minprompt_smooth_anchor_continual`, `mlmp_topk_minprompt_smooth_anchor_continual`.
- Eval ablations: `mlmp_simple_eval_continual` (MLMP adapt loss, TENT-style
  single-prompt/layer eval), `tent_uaml_eval_continual` (TENT adapt loss, MLMP
  UAML 18-layer eval).

### 5e. DeYO × MLMP hybrids
`deyo_uaml_continual`, `deyo_mlmp_continual`, `deyo_mlmp_divgate_continual`,
`deyo_mlmp_smooth_anchor_continual`.

---

## 6. Smooth Anchor — how it works (you asked)

Implemented in `adapt/tent_divgate_smooth_anchor.py` (TENT base) and the
`*_smooth_anchor_continual` MLMP/DeYO variants. It is an evolution of the
DivGate restoration idea:

- **Plain DivGate** restores LN weights toward the **frozen source** snapshot at
  a discrete rate (aggressive=0 / cautious / brake).
- **Recent/Hybrid anchor** restore toward a snapshot from *N batches ago* instead
  of source — but they switch anchor depth discontinuously at the H thresholds.
- **Smooth anchor** makes the anchor *depth* a **continuous function of model
  health** `H_margin`. "The worse the health, the deeper we reach back":

  ```
  lag(H) = lag_scale / (H - h_floor)     for h_floor < H < h_ceil
  lag(H) = ∞  (→ frozen source)          for H ≤ h_floor
  rst    = 0  (no restore)               for H ≥ h_ceil
  ```

  Defaults `lag_scale=90, h_floor=1.5, h_ceil=1.8, max_lag=3000, rst=0.005`:
  `H=1.80→lag 300`, `1.65→600`, `1.55→1800`, `≤1.50→source`.
  Restoration *rate* is fixed (0.005); only the *anchor depth* varies with H.

- **Memory cost:** keeps a deque of fp16-CPU LN snapshots of size `max_lag+1`
  (~0.9 GB CPU RAM for ViT-L/14 at max_lag=3000). H_margin is recomputed every
  `monitor_interval` batches (same buffer-N aggregation as TENT-DivGate).

**Run it:**
```bash
bash bash/ACDC_10_round/smooth_anchor_default.sh   # method=tent_divgate_smooth_anchor
```
Key args: `--h_ceil --h_floor --lag_scale --max_lag --rst --monitor_interval`.
There is a sigmoid-gate hyperparameter sweep at
`bash/ACDC_10_round/sweep_sigmoid/` (A1..E1, split across GPUs by `run_gpu*.sh`).

---

## 7. Where to look next

- Run scripts: `bash/ACDC_10_round/` (116), `bash/cityscapes/` (44),
  `bash/v20/` (49) hold the per-method runners + variants.
- Results land in `save/{Dataset}/{method}/results_all_rounds.txt` (gitignored).
- No spec docs exist for these methods — read the file docstrings in `adapt/`.
- **Our own validated story is still TENT-DivGate (ACDC mean 31.59).** 學長's
  methods have not been benchmarked into our tables yet — that's open work.
