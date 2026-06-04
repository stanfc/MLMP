# H1 Validation Experiments — Design Spec

**Date**: 2026-05-28
**Status**: design approved, awaiting plan + implementation
**Owner**: tekai
**Related**: [docs/EXPERIMENT_STATUS.md](../../EXPERIMENT_STATUS.md), session conversation 2026-05-28

---

## 1. Goal

Build **two scientific experiments** that quantitatively test the hypothesis:

> **H1**: NA-CLIP (LAION-pretrained) has already seen images visually similar to ImageNet-C-style synthetic corruptions in pretraining. Therefore on synthetic-corruption test sets, source weights are already near-optimal, leaving no coherent adaptation direction for entropy-based CTTA. Native domain shifts (real night/adverse weather) are genuinely out-of-distribution, providing real adaptation headroom that DivGate-style methods can exploit to beat MLMP-episodic.

Empirical pattern motivating H1 (already observed, see [docs/EXPERIMENT_STATUS.md §7-9](../../EXPERIMENT_STATUS.md)):

- **Native (4/4 datasets)**: DivGate variants beat MLMP-episodic mean
- **Synthetic (3/3 datasets)**: DivGate variants fail to reach MLMP-episodic mean and often degrade below source

The two experiments below give **independent quantitative evidence** for H1 without running another 150-round training experiment.

---

## 2. Scope (in vs out)

### In scope
- **Exp 1**: per-image cosine similarity between TENT (entropy) gradient and supervised CE gradient, computed on LayerNorm `{γ, β}` parameters at source weights, across 5 datasets
- **Exp 3**: source mIoU (no-adapt) sweep across corruption severity 1-5 on VOC20 + Cityscapes, with native datasets as horizontal references
- A reusable Python package `experiments/h1_validation/` that supports adding new datasets / new experiments without restructuring
- Bash drivers that loop over (dataset, condition, [severity]) and append to a single CSV per experiment
- Three plot scripts producing 5 figures (3 for exp 1, 2 for exp 3)

### Out of scope
- Experiment 2 (CLIP feature-space distance against pretraining proxy) — deferred; possible follow-up
- Experiment 4 (per-image initial entropy distribution) — deferred
- Touching `adapt/*` files or any existing training pipelines beyond a one-line plumbing of `corruption_severity` through `prepare_data()` and one CLI arg in `main_continual.py`
- Reproducing existing CTTA results — neither experiment runs an optimizer step
- Any conclusion that requires multiple random seeds — both experiments are deterministic given the source model and dataset split

---

## 3. Background — Why these two experiments are decisive

For each (dataset, condition) and each image `x`, with source model `f_θ` and supervised label `y`:

```
g_tent(x) = ∂ L_entropy(f_θ(x)) / ∂{γ, β}
g_sup(x)  = ∂ L_CE(f_θ(x), y) / ∂{γ, β}
cos(x)    = ⟨g_tent, g_sup⟩ / (‖g_tent‖ · ‖g_sup‖)
```

The **per-image cosine `cos(x)`** is the rigorous version of "is TENT pulling toward the right direction":
- `cos > 0` ⇒ entropy minimization shares a positive component with supervised improvement
- `cos ≈ 0` ⇒ TENT gradient is orthogonal to the right answer (random direction)
- `cos < 0` ⇒ TENT actively pulls away from correct adaptation

H1 predicts: **native → cos distribution centered well above 0; synthetic → cos distribution clustered near 0**.

The **severity decay curve** is the rigorous version of "did CLIP already see this corruption":
- If CLIP has seen the corruption in pretraining, source mIoU decays slowly as severity increases (shallow slope)
- If the corruption is genuinely OOD, source mIoU drops steeply

H1 predicts: **synthetic curves flat across severity 1-5; native source mIoU much lower than any synthetic point**.

Both signals are observable **without any optimization** — they are pure forward / single backward pass measurements. This is what makes them decisive and cheap.

---

## 4. Repository layout

```
experiments/h1_validation/
├── README.md                          # How to run, where results land
├── common/
│   ├── __init__.py
│   ├── model.py                       # load_source_model()
│   ├── datasets.py                    # DATASET_REGISTRY + build_loader()
│   ├── ln_utils.py                    # flatten_grads(), cosine()
│   └── io.py                          # append_row(), already_done()
├── exp1_gradient_cosine/
│   ├── __init__.py
│   ├── run.py                         # CLI: --dataset --condition --n
│   ├── plot.py                        # CSV → 3 figures
│   └── README.md
├── exp3_severity_sweep/
│   ├── __init__.py
│   ├── run.py                         # CLI: --dataset --corruption --severity
│   ├── plot.py                        # CSV → 2 figures
│   └── README.md
├── bash/
│   ├── run_exp1.sh                    # loops (dataset, condition) → run.py
│   ├── run_exp3.sh                    # loops (dataset, corruption, severity) → run.py
│   └── plot_all.sh                    # calls both plot.py
└── results/                           # generated; .gitignore'd content (keep folder)
    ├── exp1/
    │   ├── all.csv                    # append-only, schema in §6.2
    │   └── figs/{boxplot,bar,scatter}.{png,svg}
    └── exp3/
        ├── all.csv                    # append-only, schema in §7.2
        └── figs/{decay_curves,relative_drop}.{png,svg}
```

### 4.1 Imports and reuse

`common/model.py` imports from existing code to guarantee alignment with the real CTTA methods:

- `ovss.load_ovss` — model construction (matches `adapt/tent_continual.py:58`)
- `adapt.tent_continual.TENTContinual.collect_ln_params` (or copy its body if a static method import is awkward) — LN parameter selection
- `adapt.tent_continual.TENTContinual.softmax_entropy` — entropy loss formula
- `utils.segmentation_datasets.prepare_data` — data loaders

This is the minimum surface area for "guaranteed to compute the same forward / gradient as the production method". Do **not** reimplement softmax, entropy, or LN selection.

### 4.2 What `common/` does NOT abstract

- Per-experiment metrics (cosine vs mIoU) live in each `run.py` — no `BaseMetric` class
- Plot scripts are independent; they read CSV columns by name. No shared `BasePlotter`.
- Result aggregation is "append rows to a CSV"; no database, no per-run JSON, no run manifest

Keeping abstraction shallow is intentional: each experiment is < 200 lines, and the cost of duplicating one CSV-reading function across two `plot.py`s is much smaller than the cost of an abstract `Experiment` base class that gets in the way later.

---

## 5. Shared module: `common/`

### 5.1 `common/model.py`

Text features depend on the dataset's class list (ACDC=19, VOC20=20, Cityscapes=19), so model loading and text-feature computation are deliberately split into two functions.

```python
def load_source_model(
    device: str = "cuda",
    ovss_type: str = "naclip",
    ovss_backbone: str = "ViT-L/14",
) -> tuple[nn.Module, callable, list[nn.Parameter], list[str]]:
    """
    Load NA-CLIP exactly as TENT-DivGate sees it (frozen text, LN-trainable visual).

    Returns:
        model       — NA-CLIP, eval mode but with LN γ,β requires_grad=True
        tokenize    — text tokenizer
        ln_params   — list of nn.Parameter, fixed iteration order
        ln_names    — parallel list of state_dict names (for debugging)
    """

def compute_text_features(
    model: nn.Module,
    tokenize: callable,
    class_names: list[str],
    prompt_template_idx: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build text features for one prompt template against a class list.

    Returns:
        text_features — (num_classes, D), L2-normalized
    """
```

- LN selection mirrors `adapt/tent_continual.py:72` (`collect_ln_params(model.visual)`)
- `prompt_template_idx=0` → first entry of `prompts.yaml`, single-prompt convention (aligns with TENT/TENT-DivGate). The arg keeps the door open for a later 7-prompt ablation.
- `model.eval()` but `requires_grad_(True)` on LN — needed for backward to populate `.grad`
- `class_names` comes from the registry entry (added below as `class_names` field) so the same model object can be reused across datasets within one Python session if desired

### 5.2 `common/datasets.py`

Source of truth for each registry entry's `prepare_data_kwargs` and `class_names`: the existing bash script for the closest method on that dataset (so we are guaranteed bit-identical preprocessing to the production CTTA runs).

| Registry key       | kwargs source bash script                                                  |
|--------------------|------------------------------------------------------------------------------|
| `ACDC`             | [bash/ACDC_10_round/no_adapt.sh](../../../bash/ACDC_10_round/no_adapt.sh)                |
| `DarkZurich`       | [bash/dark_zurich/no_adapt.sh](../../../bash/dark_zurich/no_adapt.sh)                    |
| `NighttimeDriving` | [bash/nighttime_driving/no_adapt.sh](../../../bash/nighttime_driving/no_adapt.sh)        |
| `VOC20_matched`    | [bash/v20_acdc_matched/no_adapt.sh](../../../bash/v20_acdc_matched/no_adapt.sh)          |
| `Cityscapes`       | [bash/cityscapes_continual/no_adapt.sh](../../../bash/cityscapes_continual/no_adapt.sh)  |

`class_names` for the registry — derived from `utils/segmentation_datasets.py` dataset definitions (each dataset class's `class_extensions` / mmseg `METAINFO['classes']`).

```python
DATASET_REGISTRY = {
    "ACDC": {
        "kind": "native",
        "main_dataset_name": "ACDCDataset",
        "data_dir": "data/ACDC/",
        "conditions": ["fog", "night", "rain", "snow"],
        "class_names": [...19 Cityscapes classes...],     # mirror ACDCDataset METAINFO
        "prepare_data_kwargs": {
            "init_resize": [1120, 560], "patch_size": 560, "patch_stride": 280,
            "batch_size": 1, "num_workers": 1, "shuffle": False,
        },
    },
    "DarkZurich": {
        "kind": "native",
        "main_dataset_name": "DarkZurichDataset",
        "data_dir": "data/Dark_Zurich_val_anon/",
        "conditions": ["night"],
        "class_names": [...19 Cityscapes classes...],
        "prepare_data_kwargs": {
            "init_resize": [1120, 560], "patch_size": 560, "patch_stride": 280,
            "batch_size": 1, "num_workers": 1, "shuffle": False,
        },
    },
    "NighttimeDriving": {
        "kind": "native",
        "main_dataset_name": "NighttimeDrivingDataset",
        "data_dir": "data/NighttimeDrivingTest/",
        "conditions": ["night"],
        "class_names": [...19 Cityscapes classes...],
        "prepare_data_kwargs": {
            "init_resize": [1120, 560], "patch_size": 560, "patch_stride": 280,
            "batch_size": 1, "num_workers": 1, "shuffle": False,
        },
    },
    "VOC20_matched": {
        "kind": "synthetic",
        "main_dataset_name": "PascalVOC20Dataset",
        "data_dir": "data/VOC/VOC2012/",
        "conditions": ["snow", "fog", "frost", "contrast"],
        "class_names": [...20 VOC classes (no background, per PascalVOC20Dataset)...],
        "prepare_data_kwargs": {
            "init_resize": [224, 224], "patch_size": 224, "patch_stride": 112,
            "batch_size": 1, "num_workers": 1, "shuffle": False,
            "ann_file": "ImageSets/Segmentation/val_subset_101_seed0.txt",
        },
    },
    "Cityscapes": {
        "kind": "synthetic",
        "main_dataset_name": "CityscapesDataset",
        "data_dir": "data/Cityscapes/",
        "conditions": ["snow", "frost", "fog", "brightness", "contrast"],
        "class_names": [...19 Cityscapes classes...],
        "prepare_data_kwargs": {
            "init_resize": [1120, 560], "patch_size": 560, "patch_stride": 280,
            "batch_size": 1, "num_workers": 1, "shuffle": False,
        },
    },
}

def build_loader(
    dataset_key: str,
    condition: str,
    severity: int = 5,
    n_samples: int | str | None = None,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Thin wrapper around utils.segmentation_datasets.prepare_data().

    - For native datasets, `severity` is ignored and `condition` is the
      ACDC/DZ/ND sub-condition name passed as `corruption=condition`.
    - For synthetic datasets, `condition` is the ImageNet-C corruption name
      and `severity` ∈ {1,2,3,4,5} is plumbed to prepare_data.
    - n_samples=None → full set; int → truncate via subset Sampler;
      "all" → equivalent to None.
    """
```

- `prepare_data_kwargs` values inside each registry entry are the source of truth — adding a dataset means filling them in once
- `build_loader` is the only place that translates registry entries into `prepare_data` calls; `run.py` files do not call `prepare_data` directly
- Truncation uses a deterministic `torch.utils.data.SubsetRandomSampler(seed=seed)` so the same first-N images are reused across reruns

### 5.3 `common/ln_utils.py`

```python
def flatten_grads(ln_params: list[nn.Parameter]) -> torch.Tensor:
    """Concatenate .grad of each param into a single fp32 1-D tensor.
    Raises if any .grad is None (caller bug)."""

def cosine(g1: torch.Tensor, g2: torch.Tensor) -> tuple[float, float, float]:
    """Returns (cos_similarity, ||g1||, ||g2||). All fp32, all CPU floats."""
```

- fp32 deliberately; the model can be fp16 but the gradient comparison must not lose precision
- norms returned alongside cosine because exp 1 also reports `‖g_sup‖` distributions

### 5.4 `common/io.py`

```python
def append_row(csv_path: str, row: dict) -> None:
    """If file doesn't exist, write header from row.keys() then the row.
    If file exists, validate header subset and append.
    Header is determined by the first writer; later writers may not add keys.
    """

def already_done(csv_path: str, key_cols: dict[str, ...]) -> bool:
    """Returns True if a row matching all key_cols values exists.
    Used by --skip_done in both run.py scripts for resume safety.
    """
```

- CSV chosen over JSONL for cheap viewing / pandas / Excel
- Schema-locked-by-first-writer keeps things simple; if we add a new metric later, we either start a new CSV or migrate (out of scope here)

---

## 6. Experiment 1 — Gradient Cosine

### 6.1 `exp1_gradient_cosine/run.py` CLI

```
python -m experiments.h1_validation.exp1_gradient_cosine.run \
    --dataset {ACDC|DarkZurich|NighttimeDriving|VOC20_matched|Cityscapes} \
    --condition <condition or corruption name> \
    --n {int|all}                          # default 100
    [--out PATH]                           # default results/exp1/all.csv
    [--prompt_idx INT]                     # default 0
    [--seed INT]                           # default 0; affects sample selection only
    [--device STR]                         # default cuda
    [--skip_done]                          # resume-safe; skip rows already in CSV
```

### 6.2 CSV schema (`results/exp1/all.csv`)

| column      | type   | description                                                     |
|-------------|--------|-----------------------------------------------------------------|
| `dataset`   | str    | DATASET_REGISTRY key                                            |
| `kind`      | str    | "native" or "synthetic" (denormalized from registry for plotting) |
| `condition` | str    | sub-condition / corruption name                                 |
| `idx`       | int    | image index within the (dataset, condition) subset (0-based)    |
| `cos`       | float  | `cos(g_tent, g_sup)`                                            |
| `norm_tent` | float  | `‖g_tent‖₂`                                                     |
| `norm_sup`  | float  | `‖g_sup‖₂`                                                      |
| `n_pixels`  | int    | number of GT pixels not equal to 255 (used to weight if needed) |

### 6.3 Algorithm (per image, deterministic)

Setup (once per `run.py` invocation):
- `model, tokenize, ln_params, _ = load_source_model()`
- `text_features = compute_text_features(model, tokenize, DATASET_REGISTRY[args.dataset]["class_names"], args.prompt_idx)`
- `loader = build_loader(args.dataset, args.condition, n_samples=parse_n(args.n))`

Per image:

1. Move `img, gt` to device
2. Compute `g_tent`:
   - `model.zero_grad()`
   - `logits, _, _ = model(img, text_features, True, interpolate=False)`  # patch resolution, matches `adapt/tent_continual.py:150`
   - `L_tent = softmax_entropy(logits).mean()`
   - `L_tent.backward()`
   - `g_tent = flatten_grads(ln_params)` (clone after `.detach()`)
3. Compute `g_sup`:
   - `model.zero_grad()`
   - `logits_full, _, _ = model(img, text_features, True, interpolate=True)`  # GT resolution
   - `L_sup = F.cross_entropy(logits_full, gt, ignore_index=255)`
   - `L_sup.backward()`
   - `g_sup = flatten_grads(ln_params)`
4. `cos, n_tent, n_sup = cosine(g_tent, g_sup)`
5. `append_row(csv, {...})`

### 6.4 Why two forward passes

The TENT loss uses **patch-resolution logits** (`interpolate=False`, identical to `adapt/tent_continual.py:150`). The supervised CE loss needs **GT-resolution logits** (`interpolate=True`) to compute pixel CE against the ground truth.

These two losses both compute gradients wrt the **same set of LN parameters** `{γ, β}`. Their flattened gradient vectors live in the same vector space, so `cos(g_tent, g_sup)` is well-defined and answers the right question: "does pushing entropy down also push CE down".

### 6.5 `bash/run_exp1.sh`

```bash
#!/bin/bash
# usage: N=100 GPU=0 bash experiments/h1_validation/bash/run_exp1.sh
# optional filters: DATASETS="ACDC DarkZurich" CONDITIONS="fog night"
set -e
N=${N:-100}
GPU=${GPU:-0}
CSV=${CSV:-experiments/h1_validation/results/exp1/all.csv}
DATASETS=${DATASETS:-"ACDC DarkZurich NighttimeDriving VOC20_matched Cityscapes"}

mkdir -p "$(dirname $CSV)"
for ds in $DATASETS; do
  conds=$(python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['$ds']['conditions']))")
  for cond in $conds; do
    echo "[exp1] $ds / $cond  (n=$N)"
    CUDA_VISIBLE_DEVICES=$GPU python -m experiments.h1_validation.exp1_gradient_cosine.run \
        --dataset $ds --condition $cond --n $N --out $CSV --skip_done
  done
done
```

### 6.6 `exp1_gradient_cosine/plot.py` — three figures

```
python -m experiments.h1_validation.exp1_gradient_cosine.plot \
    --csv experiments/h1_validation/results/exp1/all.csv \
    --out_dir experiments/h1_validation/results/exp1/figs/
```

1. **`cosine_boxplot.{png,svg}`** — X axis: each `(dataset, condition)` pair; Y axis: per-image `cos`. Native boxes use a blue-green palette, synthetic uses an orange-red palette. Horizontal red line at `y=0`. **Main figure.**
2. **`cosine_bar.{png,svg}`** — mean ± 95% CI per `(dataset, condition)`, same ordering and colors. Paper-ready summary.
3. **`norm_vs_cos_scatter.{png,svg}`** — scatter with `x = log10(norm_sup)`, `y = cos`, colored by `kind`. Each point is one image. Supports the secondary claim that `‖g_sup‖` itself is small on synthetic (the supervised loss has little to fix).

All figures saved as both PNG (for slides) and SVG (for paper).

### 6.7 Expected runtime / cost

- 100 images per (dataset, condition)
- 15 (dataset, condition) pairs total (4 ACDC + 1 DZ + 1 ND + 4 VOC20_matched + 5 Cityscapes)
- ~3 seconds per image on RTX PRO 6000 (two forwards + two backwards at NA-CLIP ViT-L/14)
- Total ≈ 75 minutes

If a (dataset, condition) is interrupted, `--skip_done` resumes from the next un-recorded `idx`.

---

## 7. Experiment 3 — Severity Sweep

### 7.1 Required core-code patch (minimal, backward-compatible)

Two edits, both adding optional kwargs with the existing hardcoded value as default:

**[utils/segmentation_datasets.py:682](../../utils/segmentation_datasets.py#L682)**

```python
def prepare_data(..., corruption_severity: int = 5):
    ...
    corrupt_transform = {
        'type': 'CorruptTransform',
        'corruption_severity': corruption_severity,   # was hardcoded 5
        'corruption_name': corruption,
        ...
    }
```

**[main_continual.py](../../main_continual.py)** — add CLI:

```python
parser.add_argument('--severity', type=int, default=5,
                    help='ImageNet-C corruption severity 1-5 (synthetic only)')
# pass through to prepare_data(...)
```

All existing bash scripts continue to work unchanged because the default is `5`.

### 7.2 CSV schema (`results/exp3/all.csv`)

| column        | type   | description                                                 |
|---------------|--------|-------------------------------------------------------------|
| `dataset`     | str    | DATASET_REGISTRY key                                        |
| `kind`        | str    | "synthetic" (native datasets have no severity; see §7.5)    |
| `corruption`  | str    | corruption name                                             |
| `severity`    | int    | 1-5                                                         |
| `miou`        | float  | source-only Mean_mIoU from `results_all_rounds.txt` (Round 1) |

### 7.3 `exp3_severity_sweep/run.py` — orchestrate one cell via subprocess

```
python -m experiments.h1_validation.exp3_severity_sweep.run \
    --dataset {VOC20_matched|Cityscapes} \
    --corruption <name> \
    --severity {1..5} \
    [--out PATH]                       # default results/exp3/all.csv
    [--save_dir PATH]                  # default results/exp3/raw/<ds>_<corr>_s<sev>/
    [--skip_done]
```

Implementation:

1. Construct `main_continual.py` invocation with:
   - `--dataset <main_dataset_name>` from registry
   - `--method tent_continual` (any method works; we omit `--adapt` so it's source-only)
   - `--corruption <corruption>`, `--severity <severity>`, `--continual_rounds 1`
   - `--ann_file` if registry entry has one (e.g. VOC20_matched)
   - `--save_dir <save_dir>`
2. `subprocess.run(cmd, check=True)`
3. Parse `<save_dir>/results_all_rounds.txt` — first round's `Mean_mIoU`
4. `append_row(csv, {...})`

**Why subprocess instead of import**:
- Isolates corruption-cache failures and CUDA OOM into single cells
- Avoids global-state pollution across consecutive runs of `main_continual.py`
- Cost is ~30 s model-load overhead per cell; we have ≤ 45 cells total

### 7.4 `bash/run_exp3.sh`

```bash
#!/bin/bash
# usage: GPU=2 bash experiments/h1_validation/bash/run_exp3.sh
# optional filters: DATASETS="VOC20_matched" CORRUPTIONS="snow fog" SEVERITIES="1 3 5"
set -e
GPU=${GPU:-0}
CSV=${CSV:-experiments/h1_validation/results/exp3/all.csv}
DATASETS=${DATASETS:-"VOC20_matched Cityscapes"}
SEVERITIES=${SEVERITIES:-"1 2 3 4 5"}

mkdir -p "$(dirname $CSV)"
for ds in $DATASETS; do
  corrs=$(python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['$ds']['conditions']))")
  for corr in $corrs; do
    for sev in $SEVERITIES; do
      echo "[exp3] $ds / $corr / sev=$sev"
      CUDA_VISIBLE_DEVICES=$GPU python -m experiments.h1_validation.exp3_severity_sweep.run \
          --dataset $ds --corruption $corr --severity $sev --out $CSV --skip_done
    done
  done
done
```

### 7.5 Native references on the severity plot

Native datasets do not have an ImageNet-C-style severity axis — they are real captures. They appear on the severity decay plot as **horizontal reference lines** spanning the full x-range, labeled with the dataset name and source mIoU.

The values come from already-recorded experiments and are hard-coded in `plot.py` (not produced by this experiment), with citation comments pointing at their `save/` directories:
- ACDC source mIoU = 23.34 (`save/ACDCDataset/No_Adaptation/`)
- DarkZurich source mIoU = 20.24 (`save/DarkZurichDataset/No_Adaptation/`)
- NighttimeDriving source mIoU = 31.01 (`save/NighttimeDrivingDataset/No_Adaptation/`)

### 7.6 `exp3_severity_sweep/plot.py` — two figures

```
python -m experiments.h1_validation.exp3_severity_sweep.plot \
    --csv experiments/h1_validation/results/exp3/all.csv \
    --out_dir experiments/h1_validation/results/exp3/figs/
```

1. **`severity_decay_curves.{png,svg}`** — X axis: severity 1-5; Y axis: source mIoU. One line per `(dataset, corruption)`: VOC20 solid, Cityscapes dashed. Native reference lines drawn as flat horizontal dotted lines at their source mIoU. **Main figure**.
2. **`relative_drop_bar.{png,svg}`** — bar chart of `(miou@sev=1 − miou@sev=5) / miou@sev=1` per `(dataset, corruption)`, sorted ascending. Small drop ⇒ CLIP already-robust ⇒ H1 supported.

### 7.7 Expected runtime / cost

- VOC20_matched: 4 corruptions × 5 severities = 20 cells × ~3 minutes/cell ≈ 1 hour
- Cityscapes: 5 corruptions × 5 severities = 25 cells. First severity per corruption rebuilds the corruption cache (slow: glass_blur on 100 imgs ≈ 30 min). Subsequent severities reuse cache for the lower severities and rebuild only the differential. Total ≈ 4-5 hours worst case.

Cityscapes is the long pole; VOC20 alone is sufficient as a first pass and the Cityscapes results are nice-to-have for paper.

---

## 8. Result interpretation guide (decision table for H1)

Before running, lock down what each outcome means:

| Exp 1 outcome                                                       | Exp 3 outcome                                                | Verdict on H1                                                                                    |
|---------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Native median `cos` > 0.10; synthetic median `cos` ∈ [−0.05, 0.05]  | Synthetic decay slopes shallow (Δ ≤ 15% from sev 1→5)         | **H1 strongly supported**. Single primary story for paper.                                       |
| Native median `cos` > 0.10; synthetic median `cos` > 0.10           | Synthetic decay slopes steep (Δ > 30%)                       | **H1 refuted**. Source isn't optimal on synthetic; something else explains the DivGate failure.  |
| Native and synthetic `cos` distributions overlap heavily            | Severity decay results inconsistent across corruptions       | **Ambiguous**. Need experiment 2 (feature distance) or experiment 4 (per-image entropy) as tie-breaker. |
| Native `cos` >> 0; synthetic `cos` ≈ 0 (clear)                      | Severity sweep inconclusive                                  | **H1 supported by exp 1 alone**. Exp 3 was confirmatory and not load-bearing.                    |

This table is included in the spec so that the analysis stays disciplined — we know what we'd accept or reject **before** seeing the data.

---

## 9. Extension paths (for future work, not in scope now)

The design intentionally supports these without restructuring:

- **New dataset**: add one entry to `DATASET_REGISTRY`; both bash drivers pick it up automatically via the `python -c` registry query.
- **New experiment** (e.g. exp 2 feature-distance): copy `exp1_gradient_cosine/` to a new sibling folder; reuse `common/`. CSV schema is per-experiment so no migration needed.
- **New corruption / new severity range**: pass `SEVERITIES="1 3 5"` env var to `run_exp3.sh`; no code change.
- **Multi-prompt ablation for exp 1**: `--prompt_idx` already exists; run twice and a third figure can compare distributions.
- **New loss for the gradient cosine** (e.g. MLMP loss with 7-prompt average): add a `--loss {tent|mlmp}` arg in `exp1/run.py`; existing CSV gets a `loss` column.

---

## 10. Done criteria

- [ ] `experiments/h1_validation/` exists with the structure in §4
- [ ] `common/model.py`, `common/datasets.py`, `common/ln_utils.py`, `common/io.py` all importable and used by both experiments
- [ ] `experiments/h1_validation/exp1_gradient_cosine/run.py` runs end-to-end on ACDC fog with `--n 5` (smoke test) and appends a row per image to `all.csv`
- [ ] `experiments/h1_validation/exp3_severity_sweep/run.py` runs end-to-end on VOC20_matched snow severity 1 (smoke test) and appends one row to `all.csv`
- [ ] Two bash drivers loop correctly and respect `--skip_done`
- [ ] Two plot scripts produce all 5 figures from the smoke-test CSVs (figures need not be informative; just must not crash)
- [ ] One-line `corruption_severity` patch landed in `utils/segmentation_datasets.py` and `--severity` arg added to `main_continual.py`, with all existing bash scripts still working unchanged
- [ ] `experiments/h1_validation/README.md` documents how to run, where outputs land, and how to add a new dataset

---

## 11. Open questions / explicit assumptions

- **Assumed**: NA-CLIP ViT-L/14 with `ovss_type=naclip` is the only backbone of interest. If we later want to compare with the vanilla `clip` backbone, `load_source_model` already has the knob.
- **Assumed**: single-prompt convention (`prompt_template_idx=0`) is the right baseline. 7-prompt ablation is a paste-in extension; not needed for the H1 decision.
- **Assumed**: 100 images per (dataset, condition) is enough resolution for the cosine distribution. If post-hoc analysis shows wide CIs we can rerun with `N=300`.
- **Cityscapes computational cost is the main risk**: the corruption cache may dominate wall-clock. If exp 3 on VOC20_matched cleanly supports H1, Cityscapes exp 3 becomes a paper polish item rather than a blocker.

---

## 12. References

- [docs/EXPERIMENT_STATUS.md](../../EXPERIMENT_STATUS.md) — full research arc, §7 SAR results, §9 v20_acdc_matched setup
- [CLAUDE.md](../../../CLAUDE.md) — DZ/ND/Combined infrastructure
- [adapt/tent_continual.py](../../../adapt/tent_continual.py) — reference for LN selection, entropy loss, forward call signature
- [utils/segmentation_datasets.py](../../../utils/segmentation_datasets.py) — `prepare_data` (target of the `corruption_severity` patch in §7.1)
- [scripts/make_voc_subset.py](../../../scripts/make_voc_subset.py) — the deterministic subset machinery used by `VOC20_matched`
