# H1 Validation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/h1_validation/` package with two scientific experiments (gradient cosine, severity sweep) plus shared utilities and plotting scripts, validating H1: "CLIP has seen synthetic-like corruptions in pretraining, so source weights are already near-optimal on synthetic; native shifts are genuinely OOD."

**Architecture:** A Python package `experiments/h1_validation/` containing `common/` (4 helper modules) + two experiment folders (`exp1_gradient_cosine/`, `exp3_severity_sweep/`) each with `run.py` + `plot.py`. Bash drivers loop over (dataset × condition × [severity]) and append rows to a single CSV per experiment. Exp 1 computes per-image gradient cosines with two forward/backward passes. Exp 3 orchestrates `main_continual.py` no-adapt runs at severity 1-5 via subprocess. One minimal patch plumbs `corruption_severity` through `prepare_data()` and `main_continual.py`.

**Tech Stack:** Python 3.10, PyTorch 2.1, NA-CLIP ViT-L/14, mmsegmentation 1.x, matplotlib. No new dependencies. No pytest infrastructure (project convention is inline smoke checks via `python -c`).

**Reference spec:** [docs/superpowers/specs/2026-05-28-h1-validation-experiments-design.md](../specs/2026-05-28-h1-validation-experiments-design.md)

---

## File Map

```
experiments/h1_validation/
├── __init__.py                          # empty marker
├── README.md                            # how to run
├── common/
│   ├── __init__.py                      # re-export public API
│   ├── io.py                            # append_row(), already_done()
│   ├── ln_utils.py                      # flatten_grads(), cosine()
│   ├── datasets.py                      # DATASET_REGISTRY, build_loader(), get_class_names()
│   └── model.py                         # load_source_model(), compute_text_features()
├── exp1_gradient_cosine/
│   ├── __init__.py
│   ├── run.py                           # CLI per (dataset, condition)
│   └── plot.py                          # CSV → 3 figures
├── exp3_severity_sweep/
│   ├── __init__.py
│   ├── run.py                           # CLI per (dataset, corruption, severity)
│   └── plot.py                          # CSV → 2 figures
├── bash/
│   ├── run_exp1.sh                      # (dataset, condition) loop
│   ├── run_exp3.sh                      # (dataset, corruption, severity) loop
│   └── plot_all.sh                      # invoke both plot.py
└── results/                             # generated, .gitkeep
    ├── .gitkeep
    ├── exp1/
    └── exp3/

utils/segmentation_datasets.py           # 1-line patch: corruption_severity kwarg
main_continual.py                        # 1-line patch: --severity CLI
.gitignore                               # add experiments/h1_validation/results/{exp1,exp3}/{all.csv,raw,figs}
```

**Patching strategy**: each task is self-contained — engineer can execute tasks in order and commit after each. No task references undefined symbols from a later task.

---

## Task 1: Core-code patch — plumb `corruption_severity` through `prepare_data()`

**Files:**
- Modify: `utils/segmentation_datasets.py:593` and `:682`

- [ ] **Step 1: Read current `prepare_data` signature and the hardcoded severity line**

Run: `sed -n '593,600p' utils/segmentation_datasets.py && sed -n '678,690p' utils/segmentation_datasets.py`

Expected: signature line at 593 ends in `..., ann_file=None):` and line 682 has hardcoded `'corruption_severity': 5,`.

- [ ] **Step 2: Add `corruption_severity` kwarg to `prepare_data` signature**

Edit `utils/segmentation_datasets.py` line 593:

```python
def prepare_data(dataset, data_dir, init_resize, patch_size, patch_stride, corruption="original", batch_size=128, num_workers=1, shuffle=True, corruption_cache_dir=None, ann_file=None, corruption_severity=5):
```

- [ ] **Step 3: Use the kwarg instead of hardcoded 5**

Edit the `corrupt_transform` dict (around line 682):

```python
corrupt_transform = {
    'type': 'CorruptTransform',
    'corruption_severity': corruption_severity,
    'corruption_name': corruption,
    'cache_dir': corruption_cache_dir or osp.join(osp.dirname(data_dir.rstrip('/')), '.cache', 'corruptions'),
}
```

- [ ] **Step 4: Smoke-check that existing code paths still import**

Run: `python -c "from utils.segmentation_datasets import prepare_data; import inspect; sig = inspect.signature(prepare_data); assert 'corruption_severity' in sig.parameters; assert sig.parameters['corruption_severity'].default == 5; print('OK')"`

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add utils/segmentation_datasets.py
git commit -m "feat(prepare_data): expose corruption_severity as kwarg (default 5)"
```

---

## Task 2: Core-code patch — add `--severity` CLI to `main_continual.py`

**Files:**
- Modify: `main_continual.py` (argparse section + prepare_data call site)

- [ ] **Step 1: Locate argparse block and prepare_data calls**

Run: `grep -n "argparse\|add_argument\|prepare_data(" main_continual.py | head -25`

Note the line numbers for `--corruption` argument (use as anchor for inserting `--severity` right after) and the `prepare_data(...)` call sites.

- [ ] **Step 2: Add `--severity` argument right after `--corruption`**

In the argparse block, after the existing `--corruption` arg, insert:

```python
parser.add_argument('--severity', type=int, default=5, choices=[1,2,3,4,5],
                    help='ImageNet-C corruption severity (synthetic datasets only); 5 matches prior behavior')
```

- [ ] **Step 3: Forward `args.severity` to every `prepare_data` call**

For every `prepare_data(...)` call in `main_continual.py`, append `corruption_severity=args.severity` to the kwargs. Example:

Before:
```python
loader = prepare_data(args.dataset, args.data_dir, args.init_resize, args.patch_size, args.patch_stride, corruption=corruption, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, ann_file=args.ann_file)
```

After:
```python
loader = prepare_data(args.dataset, args.data_dir, args.init_resize, args.patch_size, args.patch_stride, corruption=corruption, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, ann_file=args.ann_file, corruption_severity=args.severity)
```

(If there are multiple `prepare_data` calls — there are; e.g. source-stat loaders for EATA/DPCore/CMA-Proto — patch all of them identically.)

- [ ] **Step 4: Smoke-check by inspecting an existing dataset's source mIoU at default severity unchanged**

Run a single-round no-adapt smoke test that exercises the new code path with the default severity:

```bash
CUDA_VISIBLE_DEVICES=0 python main_continual.py --dataset PascalVOC20Dataset --data_dir data/VOC/VOC2012/ \
    --method tent_continual --ovss_type naclip --ovss_backbone ViT-L/14 \
    --init_resize 224 224 --patch_size 224 --patch_stride 112 \
    --corruption snow --severity 5 --continual_rounds 1 --batch_size 1 \
    --ann_file data/VOC/VOC2012/ImageSets/Segmentation/val_subset_101_seed0.txt \
    --save_dir /tmp/h1_smoke_task2/
```

Expected: completes without crash, writes `/tmp/h1_smoke_task2/results_all_rounds.txt`.

Then also test `--severity 1`:

```bash
# rerun with --severity 1, save to different dir
CUDA_VISIBLE_DEVICES=0 python main_continual.py ...same as above... --severity 1 --save_dir /tmp/h1_smoke_task2_sev1/
```

Expected: completes; sev=1 mIoU should be **higher** than sev=5 mIoU (corruption is milder).

Cleanup: `rm -rf /tmp/h1_smoke_task2 /tmp/h1_smoke_task2_sev1`

- [ ] **Step 5: Commit**

```bash
git add main_continual.py
git commit -m "feat(main_continual): add --severity CLI arg (default 5)"
```

---

## Task 3: Create `experiments/h1_validation/` package skeleton + .gitignore

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/h1_validation/__init__.py`
- Create: `experiments/h1_validation/common/__init__.py`
- Create: `experiments/h1_validation/exp1_gradient_cosine/__init__.py`
- Create: `experiments/h1_validation/exp3_severity_sweep/__init__.py`
- Create: `experiments/h1_validation/results/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create empty `__init__.py` markers and results placeholder**

```bash
mkdir -p experiments/h1_validation/common
mkdir -p experiments/h1_validation/exp1_gradient_cosine
mkdir -p experiments/h1_validation/exp3_severity_sweep
mkdir -p experiments/h1_validation/bash
mkdir -p experiments/h1_validation/results/exp1
mkdir -p experiments/h1_validation/results/exp3
touch experiments/__init__.py
touch experiments/h1_validation/__init__.py
touch experiments/h1_validation/common/__init__.py
touch experiments/h1_validation/exp1_gradient_cosine/__init__.py
touch experiments/h1_validation/exp3_severity_sweep/__init__.py
touch experiments/h1_validation/results/.gitkeep
touch experiments/h1_validation/results/exp1/.gitkeep
touch experiments/h1_validation/results/exp3/.gitkeep
```

- [ ] **Step 2: Update `.gitignore` to skip generated files but keep folder structure**

Read `.gitignore`:

```bash
cat .gitignore
```

Append to `.gitignore`:

```
# H1 validation experiments — keep folders but ignore generated artifacts
experiments/h1_validation/results/exp1/all.csv
experiments/h1_validation/results/exp1/raw/
experiments/h1_validation/results/exp1/figs/
experiments/h1_validation/results/exp3/all.csv
experiments/h1_validation/results/exp3/raw/
experiments/h1_validation/results/exp3/figs/
```

- [ ] **Step 3: Smoke-check imports work**

Run: `python -c "import experiments.h1_validation; import experiments.h1_validation.common; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add experiments/__init__.py experiments/h1_validation/ .gitignore
git commit -m "feat(h1_validation): create package skeleton + .gitignore rules"
```

---

## Task 4: Implement `common/io.py` — CSV append + already_done

**Files:**
- Create: `experiments/h1_validation/common/io.py`

- [ ] **Step 1: Write the smoke test first (project convention: inline script)**

Create the smoke test file at `/tmp/h1_test_io.py`:

```python
"""Smoke test for common.io. Run from project root."""
import os, csv
from experiments.h1_validation.common.io import append_row, already_done

csv_path = "/tmp/h1_test_io.csv"
if os.path.exists(csv_path):
    os.remove(csv_path)

# 1. First write creates header
append_row(csv_path, {"a": 1, "b": "x"})
with open(csv_path) as f:
    lines = f.read().strip().split("\n")
assert lines == ["a,b", "1,x"], f"got {lines}"

# 2. Second write appends without re-writing header
append_row(csv_path, {"a": 2, "b": "y"})
with open(csv_path) as f:
    lines = f.read().strip().split("\n")
assert lines == ["a,b", "1,x", "2,y"], f"got {lines}"

# 3. already_done returns True for existing key, False otherwise
assert already_done(csv_path, {"a": "1", "b": "x"}) is True
assert already_done(csv_path, {"a": "99"}) is False

os.remove(csv_path)
print("OK")
```

- [ ] **Step 2: Run the smoke test and verify it FAILS (import error)**

Run: `python /tmp/h1_test_io.py`

Expected: `ImportError` because `common/io.py` doesn't exist yet.

- [ ] **Step 3: Implement `common/io.py`**

Create `experiments/h1_validation/common/io.py`:

```python
"""CSV append + resume helpers shared by both experiments."""
from __future__ import annotations
import csv
import os
from typing import Any


def append_row(csv_path: str, row: dict[str, Any]) -> None:
    """Append one row to csv_path. Writes header from row.keys() on first call.

    If the file exists, its header must include all keys in row (we tolerate
    extra columns in the file but not missing ones).
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    if not new_file:
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f))
        missing = set(row.keys()) - set(existing_header)
        if missing:
            raise ValueError(
                f"row has columns not in existing header {existing_header}: {missing}"
            )
        fieldnames = existing_header
    else:
        fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        # if row missing some columns we'd fill blanks — but we don't expect that
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def already_done(csv_path: str, key_cols: dict[str, Any]) -> bool:
    """Return True if csv_path contains a row whose values for key_cols all
    match (string-compared, since CSV is text-only).
    """
    if not os.path.exists(csv_path):
        return False
    str_keys = {k: str(v) for k, v in key_cols.items()}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if all(r.get(k, "") == v for k, v in str_keys.items()):
                return True
    return False
```

- [ ] **Step 4: Run smoke test, verify PASS**

Run: `python /tmp/h1_test_io.py`

Expected: `OK`

Cleanup: `rm /tmp/h1_test_io.py`

- [ ] **Step 5: Commit**

```bash
git add experiments/h1_validation/common/io.py
git commit -m "feat(h1_validation/common): add CSV append_row + already_done"
```

---

## Task 5: Implement `common/ln_utils.py` — flatten_grads + cosine

**Files:**
- Create: `experiments/h1_validation/common/ln_utils.py`

- [ ] **Step 1: Write smoke test at `/tmp/h1_test_ln.py`**

```python
"""Smoke test for common.ln_utils."""
import torch
import torch.nn as nn
from experiments.h1_validation.common.ln_utils import flatten_grads, cosine

# 1. flatten_grads concatenates .grad of each param in iteration order
p1 = nn.Parameter(torch.zeros(3))
p2 = nn.Parameter(torch.zeros(2, 2))
p1.grad = torch.tensor([1.0, 2.0, 3.0])
p2.grad = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
g = flatten_grads([p1, p2])
assert g.dtype == torch.float32, f"dtype is {g.dtype}"
assert g.shape == (7,), f"shape is {g.shape}"
assert torch.allclose(g, torch.tensor([1., 2., 3., 4., 5., 6., 7.]))

# 2. cosine of identical vectors = 1.0
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0, 2.0, 3.0])
cos, na, nb = cosine(a, b)
assert abs(cos - 1.0) < 1e-6, f"cos={cos}"
assert abs(na - torch.norm(a).item()) < 1e-6
assert abs(nb - torch.norm(b).item()) < 1e-6

# 3. cosine of orthogonal = 0.0
a = torch.tensor([1.0, 0.0])
b = torch.tensor([0.0, 1.0])
cos, _, _ = cosine(a, b)
assert abs(cos) < 1e-6, f"cos={cos}"

# 4. cosine of opposite = -1.0
a = torch.tensor([1.0, 2.0])
b = torch.tensor([-1.0, -2.0])
cos, _, _ = cosine(a, b)
assert abs(cos + 1.0) < 1e-6, f"cos={cos}"

# 5. flatten_grads raises if any .grad is None
p3 = nn.Parameter(torch.zeros(2))   # no .grad set
try:
    flatten_grads([p3])
    assert False, "should have raised"
except RuntimeError:
    pass

print("OK")
```

- [ ] **Step 2: Run, verify FAIL (ImportError)**

Run: `python /tmp/h1_test_ln.py`

Expected: `ImportError`.

- [ ] **Step 3: Implement `common/ln_utils.py`**

Create `experiments/h1_validation/common/ln_utils.py`:

```python
"""LayerNorm gradient helpers."""
from __future__ import annotations
import torch
import torch.nn as nn


def flatten_grads(ln_params: list[nn.Parameter]) -> torch.Tensor:
    """Concatenate .grad of each param into one fp32 1-D tensor.

    Iteration order is the order given. Raises RuntimeError if any .grad
    is None (the caller forgot to backward, or zero'd grads after backward).
    """
    chunks = []
    for i, p in enumerate(ln_params):
        if p.grad is None:
            raise RuntimeError(f"ln_params[{i}] has .grad=None")
        chunks.append(p.grad.detach().reshape(-1).float())
    return torch.cat(chunks)


def cosine(g1: torch.Tensor, g2: torch.Tensor) -> tuple[float, float, float]:
    """Return (cosine_similarity, ||g1||_2, ||g2||_2) as python floats.

    All computation is fp32 on the input device. Returns floats so the result
    can go straight into a CSV row.
    """
    g1 = g1.float()
    g2 = g2.float()
    n1 = torch.linalg.vector_norm(g1).item()
    n2 = torch.linalg.vector_norm(g2).item()
    if n1 == 0.0 or n2 == 0.0:
        return 0.0, n1, n2
    cos = torch.dot(g1, g2).item() / (n1 * n2)
    return float(cos), float(n1), float(n2)
```

- [ ] **Step 4: Run smoke test, verify PASS**

Run: `python /tmp/h1_test_ln.py`

Expected: `OK`

Cleanup: `rm /tmp/h1_test_ln.py`

- [ ] **Step 5: Commit**

```bash
git add experiments/h1_validation/common/ln_utils.py
git commit -m "feat(h1_validation/common): add flatten_grads + cosine"
```

---

## Task 6: Implement `common/datasets.py` — DATASET_REGISTRY + build_loader

**Files:**
- Create: `experiments/h1_validation/common/datasets.py`

- [ ] **Step 1: Confirm Cityscapes-19 and VOC20 class lists**

Run:

```bash
grep -n "class_extensions\|CLASSES\s*=\|METAINFO\|class_names" utils/segmentation_datasets.py | head -30
```

Note the lines defining `class_extensions` lists for ACDCDataset / DarkZurichDataset / NighttimeDrivingDataset / CityscapesDataset / PascalVOC20Dataset. These are the source of truth for the registry's `class_names`.

- [ ] **Step 2: Write `common/datasets.py`**

Create `experiments/h1_validation/common/datasets.py`:

```python
"""Dataset registry shared by both experiments.

To add a new dataset:
1. Add an entry to DATASET_REGISTRY following the schema below.
2. Both exp1 and exp3 bash drivers pick it up automatically via the
   `python -c "from .datasets import DATASET_REGISTRY..."` query.
"""
from __future__ import annotations
from typing import Any
import torch
from torch.utils.data import Subset, DataLoader

from utils.segmentation_datasets import prepare_data


# 19 Cityscapes classes (shared by ACDC, DarkZurich, NighttimeDriving,
# Cityscapes). Order matches CityscapesDataset's METAINFO.
CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# 20 VOC classes (no background, per PascalVOC20Dataset)
VOC_20 = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


# Common preprocessing kwargs reused across driving-style datasets.
# Sourced from bash/{ACDC_10_round,dark_zurich,nighttime_driving,cityscapes_continual}/no_adapt.sh
_DRIVING_KWARGS = dict(
    init_resize=[1120, 560],
    patch_size=560,
    patch_stride=280,
    batch_size=1,
    num_workers=1,
    shuffle=False,
)

# VOC kwargs sourced from bash/v20_acdc_matched/no_adapt.sh
_VOC_KWARGS = dict(
    init_resize=[224, 224],
    patch_size=224,
    patch_stride=112,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    ann_file="ImageSets/Segmentation/val_subset_101_seed0.txt",
)


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "ACDC": {
        "kind": "native",
        "main_dataset_name": "ACDCDataset",
        "data_dir": "data/ACDC/",
        "conditions": ["fog", "night", "rain", "snow"],
        "class_names": CITYSCAPES_19,
        "prepare_data_kwargs": _DRIVING_KWARGS,
    },
    "DarkZurich": {
        "kind": "native",
        "main_dataset_name": "DarkZurichDataset",
        "data_dir": "data/Dark_Zurich_val_anon/",
        "conditions": ["night"],
        "class_names": CITYSCAPES_19,
        "prepare_data_kwargs": _DRIVING_KWARGS,
    },
    "NighttimeDriving": {
        "kind": "native",
        "main_dataset_name": "NighttimeDrivingDataset",
        "data_dir": "data/NighttimeDrivingTest/",
        "conditions": ["night"],
        "class_names": CITYSCAPES_19,
        "prepare_data_kwargs": _DRIVING_KWARGS,
    },
    "VOC20_matched": {
        "kind": "synthetic",
        "main_dataset_name": "PascalVOC20Dataset",
        "data_dir": "data/VOC/VOC2012/",
        "conditions": ["snow", "fog", "frost", "contrast"],
        "class_names": VOC_20,
        "prepare_data_kwargs": _VOC_KWARGS,
    },
    "Cityscapes": {
        "kind": "synthetic",
        "main_dataset_name": "CityscapesDataset",
        "data_dir": "data/Cityscapes/",
        "conditions": ["snow", "frost", "fog", "brightness", "contrast"],
        "class_names": CITYSCAPES_19,
        "prepare_data_kwargs": _DRIVING_KWARGS,
    },
}


def build_loader(
    dataset_key: str,
    condition: str,
    severity: int = 5,
    n_samples: int | str | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a deterministic DataLoader for one (dataset, condition).

    - native datasets: `severity` is ignored; `condition` is the sub-condition
      name passed as `corruption=` to prepare_data.
    - synthetic datasets: `condition` is the ImageNet-C corruption name;
      severity is plumbed through.
    - n_samples: None or "all" → full loader; int → deterministic prefix of N
      images by index (so reruns hit the same images).
    """
    if dataset_key not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset_key {dataset_key!r}")
    entry = DATASET_REGISTRY[dataset_key]
    if condition not in entry["conditions"]:
        raise ValueError(f"{dataset_key} has no condition {condition!r}; "
                         f"available: {entry['conditions']}")

    kwargs = dict(entry["prepare_data_kwargs"])
    if entry["kind"] == "synthetic":
        kwargs["corruption_severity"] = severity

    loader = prepare_data(
        entry["main_dataset_name"],
        entry["data_dir"],
        corruption=condition,
        **kwargs,
    )

    if n_samples is None or n_samples == "all":
        return loader

    n = int(n_samples)
    ds = loader.dataset
    n = min(n, len(ds))
    # Deterministic prefix — first N images by dataset's native ordering
    # (already deterministic with shuffle=False above).
    subset = Subset(ds, list(range(n)))
    return DataLoader(
        subset,
        batch_size=kwargs.get("batch_size", 1),
        num_workers=kwargs.get("num_workers", 1),
        shuffle=False,
    )


def get_class_names(dataset_key: str) -> list[str]:
    """Convenience accessor used by load_source_model callers."""
    return list(DATASET_REGISTRY[dataset_key]["class_names"])
```

- [ ] **Step 3: Smoke-check registry round-trip + the bash helper expression**

Run:

```bash
python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; \
assert set(R) == {'ACDC','DarkZurich','NighttimeDriving','VOC20_matched','Cityscapes'}; \
assert R['ACDC']['kind'] == 'native'; \
assert R['VOC20_matched']['kind'] == 'synthetic'; \
assert len(R['ACDC']['class_names']) == 19; \
assert len(R['VOC20_matched']['class_names']) == 20; \
print('OK')"
```

Expected: `OK`

Test the exact `python -c` expression the bash drivers will use:

```bash
python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['ACDC']['conditions']))"
```

Expected output: `fog night rain snow`

- [ ] **Step 4: Smoke-check `build_loader` actually builds a loader (using an existing dataset)**

Run:

```bash
python -c "
from experiments.h1_validation.common.datasets import build_loader
loader = build_loader('ACDC', 'fog', n_samples=2)
assert len(loader.dataset) == 2, f'got {len(loader.dataset)}'
batch = next(iter(loader))
print('batch keys/types:', type(batch))
print('OK')
"
```

Expected: completes without error, prints `OK`. (Dataset must be present on disk; if not, this confirms the registry path is right and the error is downstream.)

- [ ] **Step 5: Commit**

```bash
git add experiments/h1_validation/common/datasets.py
git commit -m "feat(h1_validation/common): add DATASET_REGISTRY + build_loader"
```

---

## Task 7: Implement `common/model.py` — load_source_model + compute_text_features

**Files:**
- Create: `experiments/h1_validation/common/model.py`

- [ ] **Step 1: Verify text-feature shape convention used by TENT-DivGate**

Run:

```bash
grep -n "extract_text_embeddings\|self.text_x" adapt/tent_continual.py | head -10
```

Confirm: `tent_continual.py:87-88` calls `self.extract_text_embeddings(self.classes, self.prompt_templates, average=False)`. With single prompt (`prompt_templates=[REFERENCE_PROMPT]` since bash doesn't pass `--prompt_dir`), output is shape `(1, num_classes, D)`. Our `compute_text_features` must produce the same shape.

- [ ] **Step 2: Implement `common/model.py`**

Create `experiments/h1_validation/common/model.py`:

```python
"""Source NA-CLIP model loader + text feature builder."""
from __future__ import annotations
import torch
import torch.nn as nn

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml

# Mirror REFERENCE_PROMPT from adapt/tent.py (single-prompt convention)
REFERENCE_PROMPT = "a photo of a {}"


def load_source_model(
    device: str = "cuda",
    ovss_type: str = "naclip",
    ovss_backbone: str = "ViT-L/14",
) -> tuple[nn.Module, callable, list[nn.Parameter], list[str]]:
    """Load NA-CLIP and prepare it for gradient computation on visual LN.

    - All params frozen except visual encoder's LN γ,β (matches TENT/TENT-DivGate).
    - Model is in eval() mode (no dropout/BN drift), but LN params have grad enabled.

    Returns:
        model      — NA-CLIP nn.Module on `device`
        tokenize   — text tokenizer callable
        ln_params  — list of nn.Parameter (visual LN γ,β); fixed iteration order
        ln_names   — parallel list of state_dict-style names (for debugging)
    """
    model, tokenize = load_ovss(ovss_type, ovss_backbone, device=device)
    model.eval()

    # Freeze everything
    model.requires_grad_(False)
    # Enable LN on visual encoder only
    for m in model.visual.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)

    # Collect LN params in deterministic order (same order as
    # adapt.tent_continual.TENTContinual.collect_ln_params)
    ln_params: list[nn.Parameter] = []
    ln_names: list[str] = []
    for nm, m in model.visual.named_modules():
        if isinstance(m, nn.LayerNorm):
            for pname, p in m.named_parameters():
                if pname in ("weight", "bias"):
                    ln_params.append(p)
                    ln_names.append(f"visual.{nm}.{pname}")
    return model, tokenize, ln_params, ln_names


def compute_text_features(
    model: nn.Module,
    tokenize: callable,
    class_names: list[str],
    prompt_template_idx: int = 0,
    prompts_yaml: str | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Build text features matching TENT-DivGate's runtime layout.

    - With prompts_yaml=None: uses [REFERENCE_PROMPT] (single prompt, matches
      bash/ACDC_10_round/tent_divgate_continual.sh which does not pass
      --prompt_dir).
    - With prompts_yaml='prompts.yaml': uses prompts.yaml[prompt_template_idx]
      as a single template (still single-prompt convention).

    Returns:
        text_features — shape (num_prompts=1, num_classes, D), L2-normalized.
                        Shape matches what `self.text_x` is when fed to
                        model.forward(...) in adapt/tent_continual.py.
    """
    if prompts_yaml is None:
        templates = [REFERENCE_PROMPT]
    else:
        all_templates = load_prompts_from_yaml(prompts_yaml)
        templates = [all_templates[prompt_template_idx]]

    text_features = []
    for class_name in class_names:
        texts = [t.format(class_name) for t in templates]
        tok = tokenize(texts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_features.append(emb)
    # Stack along class dim → (num_prompts=1, num_classes, D)
    return torch.stack(text_features, dim=1).to(device)
```

- [ ] **Step 3: Smoke-check load + shapes**

Run:

```bash
python -c "
import torch
from experiments.h1_validation.common.model import load_source_model, compute_text_features
from experiments.h1_validation.common.datasets import get_class_names
model, tok, ln_params, ln_names = load_source_model(device='cuda')
print(f'#LN params: {len(ln_params)}')
assert len(ln_params) > 0
assert all(p.requires_grad for p in ln_params)
# expect 24 blocks * 2 LN * 2 params + ln_pre*2 + ln_post*2 = 100 for ViT-L/14
assert len(ln_params) == 100, f'expected 100, got {len(ln_params)}'
classes = get_class_names('ACDC')
tf = compute_text_features(model, tok, classes, prompt_template_idx=0, device='cuda')
print(f'text features shape: {tuple(tf.shape)}')
assert tf.shape == (1, 19, tf.shape[-1]), f'shape mismatch: {tf.shape}'
print('OK')
"
```

Expected:
```
#LN params: 100
text features shape: (1, 19, <D>)
OK
```

- [ ] **Step 4: Commit**

```bash
git add experiments/h1_validation/common/model.py
git commit -m "feat(h1_validation/common): add load_source_model + compute_text_features"
```

---

## Task 8: Implement `common/__init__.py` re-exports

**Files:**
- Modify: `experiments/h1_validation/common/__init__.py`

- [ ] **Step 1: Add re-exports for convenience**

Edit `experiments/h1_validation/common/__init__.py`:

```python
"""Public API for shared utilities."""
from .io import append_row, already_done
from .ln_utils import flatten_grads, cosine
from .datasets import DATASET_REGISTRY, build_loader, get_class_names
from .model import load_source_model, compute_text_features

__all__ = [
    "append_row", "already_done",
    "flatten_grads", "cosine",
    "DATASET_REGISTRY", "build_loader", "get_class_names",
    "load_source_model", "compute_text_features",
]
```

- [ ] **Step 2: Smoke-check the package-level imports**

Run: `python -c "from experiments.h1_validation.common import *; print(DATASET_REGISTRY['ACDC']['kind'])"`

Expected: `native`

- [ ] **Step 3: Commit**

```bash
git add experiments/h1_validation/common/__init__.py
git commit -m "feat(h1_validation/common): expose public API via __init__"
```

---

## Task 9: Implement `exp1_gradient_cosine/run.py`

**Files:**
- Create: `experiments/h1_validation/exp1_gradient_cosine/run.py`

- [ ] **Step 1: Write `run.py`**

Create `experiments/h1_validation/exp1_gradient_cosine/run.py`:

```python
"""Exp 1 — Per-image cosine between TENT entropy gradient and supervised CE
gradient over visual LayerNorm parameters.

Usage:
    python -m experiments.h1_validation.exp1_gradient_cosine.run \
        --dataset ACDC --condition fog --n 100 \
        [--out experiments/h1_validation/results/exp1/all.csv] \
        [--prompt_idx 0] [--seed 0] [--device cuda] [--skip_done]
"""
from __future__ import annotations
import argparse
import sys
import time

import torch
import torch.nn.functional as F

from experiments.h1_validation.common import (
    DATASET_REGISTRY, build_loader, get_class_names,
    load_source_model, compute_text_features,
    flatten_grads, cosine,
    append_row, already_done,
)


def parse_n(n: str) -> int | str:
    if n == "all":
        return "all"
    return int(n)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Pixel-wise entropy, identical to adapt.tent_continual.softmax_entropy."""
    return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    ap.add_argument("--condition", required=True)
    ap.add_argument("--n", default="100", help="int or 'all'")
    ap.add_argument("--out", default="experiments/h1_validation/results/exp1/all.csv")
    ap.add_argument("--prompt_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    entry = DATASET_REGISTRY[args.dataset]
    kind = entry["kind"]
    classes = get_class_names(args.dataset)

    print(f"[exp1] Loading source NA-CLIP on {args.device}...", flush=True)
    model, tokenize, ln_params, _ = load_source_model(device=args.device)
    text_features = compute_text_features(
        model, tokenize, classes, prompt_template_idx=args.prompt_idx, device=args.device
    )
    print(f"[exp1] #LN params={len(ln_params)}, text features={tuple(text_features.shape)}", flush=True)

    n_samples = parse_n(args.n)
    print(f"[exp1] Building loader {args.dataset}/{args.condition} n={n_samples}", flush=True)
    loader = build_loader(args.dataset, args.condition, severity=5, n_samples=n_samples, seed=args.seed)

    t0 = time.time()
    for idx, batch in enumerate(loader):
        # Resume support
        if args.skip_done and already_done(
            args.out, {"dataset": args.dataset, "condition": args.condition, "idx": idx}
        ):
            print(f"  [skip done] idx={idx}", flush=True)
            continue

        # Batch unpacking matches the rest of the codebase (img, gt)
        # — mmseg loader yields dict-like batches; adapt to what prepare_data returns.
        # Common pattern in this repo: tuple/list of (img, gt) or dict with 'inputs','data_samples'.
        img, gt = _unpack_batch(batch, args.device)

        # ----- g_tent -----
        for p in ln_params:
            if p.grad is not None:
                p.grad.zero_()
        logits, _, _ = model(img, text_features, True, interpolate=False)
        # tent_continual.py:127 does logits = logits[0] when single prompt
        if logits.dim() == 5:
            logits = logits[0]
        L_tent = softmax_entropy(logits).mean()
        L_tent.backward()
        g_tent = flatten_grads(ln_params).cpu()

        # ----- g_sup -----
        for p in ln_params:
            if p.grad is not None:
                p.grad.zero_()
        logits_full, _, _ = model(img, text_features, True, interpolate=True)
        if logits_full.dim() == 5:
            logits_full = logits_full[0]
        # logits_full: (B, C, H, W); gt: (B, H, W)
        L_sup = F.cross_entropy(logits_full, gt, ignore_index=255)
        L_sup.backward()
        g_sup = flatten_grads(ln_params).cpu()

        cos, n_tent, n_sup = cosine(g_tent, g_sup)
        n_pixels = int((gt != 255).sum().item())
        append_row(args.out, {
            "dataset": args.dataset,
            "kind": kind,
            "condition": args.condition,
            "idx": idx,
            "cos": cos,
            "norm_tent": n_tent,
            "norm_sup": n_sup,
            "n_pixels": n_pixels,
        })
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [progress] idx={idx+1} cos={cos:+.4f} norm_sup={n_sup:.2e} "
                  f"({elapsed/(idx+1):.2f}s/img)", flush=True)

    print(f"[exp1] DONE  {args.dataset}/{args.condition}  total={time.time()-t0:.1f}s", flush=True)


def _unpack_batch(batch, device):
    """Adapt the various batch formats prepare_data may yield to (img, gt).

    Supports:
      - tuple/list (img, gt)
      - dict with 'inputs' and 'data_samples' (mmseg pipeline)
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        img, gt = batch[0], batch[1]
    elif isinstance(batch, dict):
        if "inputs" in batch and "data_samples" in batch:
            img = batch["inputs"]
            # data_samples carries gt_sem_seg; extract first sample's tensor
            ds = batch["data_samples"]
            if isinstance(ds, (list, tuple)):
                ds = ds[0]
            gt = ds.gt_sem_seg.data
        elif "img" in batch and "gt_semantic_seg" in batch:
            img = batch["img"]
            gt = batch["gt_semantic_seg"]
        else:
            raise RuntimeError(f"Unknown batch dict keys: {list(batch)}")
    else:
        raise RuntimeError(f"Unknown batch type: {type(batch)}")

    if isinstance(img, list):
        img = torch.stack(img, dim=0)
    if isinstance(gt, list):
        gt = torch.stack(gt, dim=0)
    img = img.to(device)
    gt = gt.to(device).long()
    if gt.dim() == 4 and gt.shape[1] == 1:
        gt = gt.squeeze(1)
    return img, gt


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on ACDC fog with n=2 (verify pipeline end-to-end)**

Run:

```bash
rm -f experiments/h1_validation/results/exp1/all.csv
CUDA_VISIBLE_DEVICES=0 python -m experiments.h1_validation.exp1_gradient_cosine.run \
    --dataset ACDC --condition fog --n 2 \
    --out experiments/h1_validation/results/exp1/all.csv
```

Expected:
- prints `#LN params=100`, `text features=(1, 19, ...)`, `DONE ACDC/fog`
- `experiments/h1_validation/results/exp1/all.csv` exists with 1 header line + 2 data rows
- `cos` values are finite floats (not NaN); `norm_tent` and `norm_sup` > 0

Inspect: `cat experiments/h1_validation/results/exp1/all.csv`

- [ ] **Step 3: Smoke-test `--skip_done` resume**

Run the same command again:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.h1_validation.exp1_gradient_cosine.run \
    --dataset ACDC --condition fog --n 2 \
    --out experiments/h1_validation/results/exp1/all.csv \
    --skip_done
```

Expected: prints `[skip done] idx=0` and `[skip done] idx=1`; CSV unchanged.

Cleanup: `rm -f experiments/h1_validation/results/exp1/all.csv`

- [ ] **Step 4: Commit**

```bash
git add experiments/h1_validation/exp1_gradient_cosine/run.py
git commit -m "feat(h1_validation/exp1): add gradient cosine runner"
```

---

## Task 10: Implement `exp1_gradient_cosine/plot.py` — 3 figures

**Files:**
- Create: `experiments/h1_validation/exp1_gradient_cosine/plot.py`

- [ ] **Step 1: Write `plot.py`**

Create `experiments/h1_validation/exp1_gradient_cosine/plot.py`:

```python
"""Exp 1 plots: box plot, mean±CI bar, |g_sup| vs cos scatter.

Usage:
    python -m experiments.h1_validation.exp1_gradient_cosine.plot \
        --csv experiments/h1_validation/results/exp1/all.csv \
        --out_dir experiments/h1_validation/results/exp1/figs/
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

NATIVE_COLORS = ["#0ea5e9", "#0891b2", "#0d9488", "#059669", "#16a34a", "#65a30d"]
SYNTHETIC_COLORS = ["#f97316", "#ea580c", "#dc2626", "#db2777", "#c026d3", "#9333ea"]


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cos"] = float(r["cos"])
        r["norm_tent"] = float(r["norm_tent"])
        r["norm_sup"] = float(r["norm_sup"])
        r["idx"] = int(r["idx"])
        r["n_pixels"] = int(r["n_pixels"])
    return rows


def group_by_condition(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Returns {(dataset, condition): {kind, cosines, norms_sup}} preserving native-first order."""
    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["dataset"], r["condition"])
        if key not in groups:
            groups[key] = {"kind": r["kind"], "cosines": [], "norms_sup": []}
        groups[key]["cosines"].append(r["cos"])
        groups[key]["norms_sup"].append(r["norm_sup"])
    # sort: native first, then synthetic; within each kind, preserve insertion order
    sorted_keys = sorted(groups.keys(), key=lambda k: (groups[k]["kind"] != "native", k))
    return {k: groups[k] for k in sorted_keys}


def color_for(idx_within_kind: int, kind: str) -> str:
    palette = NATIVE_COLORS if kind == "native" else SYNTHETIC_COLORS
    return palette[idx_within_kind % len(palette)]


def plot_boxplot(groups, out_path: str):
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(groups) + 4), 6), dpi=160)
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0, zorder=2)
    labels, data, colors = [], [], []
    nidx = sidx = 0
    for (ds, cond), g in groups.items():
        labels.append(f"{ds}\n{cond}")
        data.append(g["cosines"])
        if g["kind"] == "native":
            colors.append(color_for(nidx, "native"))
            nidx += 1
        else:
            colors.append(color_for(sidx, "synthetic"))
            sidx += 1
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("cos(g_tent, g_sup)", fontsize=11)
    ax.set_title("Exp 1 — Per-image cosine of TENT gradient vs supervised CE gradient\n"
                 "(positive ⇒ entropy minimization aligned with correctness)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_bar(groups, out_path: str):
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(groups) + 4), 6), dpi=160)
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0, zorder=2)
    labels, means, ci_lows, ci_highs, colors = [], [], [], [], []
    nidx = sidx = 0
    for (ds, cond), g in groups.items():
        arr = np.array(g["cosines"])
        m = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        labels.append(f"{ds}\n{cond}")
        means.append(m)
        ci_lows.append(1.96 * se)
        ci_highs.append(1.96 * se)
        if g["kind"] == "native":
            colors.append(color_for(nidx, "native"))
            nidx += 1
        else:
            colors.append(color_for(sidx, "synthetic"))
            sidx += 1
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=[ci_lows, ci_highs], color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean cos(g_tent, g_sup) ± 95% CI", fontsize=11)
    ax.set_title("Exp 1 — Mean gradient cosine per (dataset, condition)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_scatter(groups, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0)
    for (ds, cond), g in groups.items():
        col = "#0ea5e9" if g["kind"] == "native" else "#dc2626"
        ax.scatter(np.log10(np.array(g["norms_sup"]) + 1e-12), g["cosines"],
                   c=col, alpha=0.55, s=22, label=f"{ds}/{cond}" if False else None)
    # one legend handle per kind
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#0ea5e9", label="native"),
        Line2D([], [], marker="o", linestyle="", color="#dc2626", label="synthetic"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_xlabel("log10 ||g_sup||", fontsize=11)
    ax.set_ylabel("cos(g_tent, g_sup)", fontsize=11)
    ax.set_title("Exp 1 — Per-image |g_sup| vs cosine alignment\n"
                 "(small |g_sup| on synthetic ⇒ supervised loss has little to fix)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/h1_validation/results/exp1/all.csv")
    ap.add_argument("--out_dir", default="experiments/h1_validation/results/exp1/figs/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    groups = group_by_condition(rows)
    print(f"[plot] {len(rows)} rows across {len(groups)} (dataset, condition) pairs")
    plot_boxplot(groups, os.path.join(args.out_dir, "cosine_boxplot"))
    plot_bar(groups, os.path.join(args.out_dir, "cosine_bar"))
    plot_scatter(groups, os.path.join(args.out_dir, "norm_vs_cos_scatter"))
    print(f"[plot] Saved 3 figure pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test plot.py using a hand-written tiny CSV**

Create a tiny CSV at the expected path:

```bash
mkdir -p experiments/h1_validation/results/exp1/
cat > experiments/h1_validation/results/exp1/all.csv <<'EOF'
dataset,kind,condition,idx,cos,norm_tent,norm_sup,n_pixels
ACDC,native,fog,0,0.12,1.2e-3,4.5e-3,30000
ACDC,native,fog,1,0.08,1.1e-3,4.7e-3,30000
ACDC,native,night,0,0.21,1.5e-3,5.1e-3,30000
VOC20_matched,synthetic,snow,0,-0.01,1.0e-3,1.2e-3,49000
VOC20_matched,synthetic,snow,1,0.02,1.1e-3,1.1e-3,49000
VOC20_matched,synthetic,fog,0,0.005,9.8e-4,9.0e-4,49000
EOF
python -m experiments.h1_validation.exp1_gradient_cosine.plot \
    --csv experiments/h1_validation/results/exp1/all.csv \
    --out_dir experiments/h1_validation/results/exp1/figs/
```

Expected: prints `[plot] 6 rows across 4 (dataset, condition) pairs` and `[plot] Saved 3 figure pairs to ...`. Three PNGs and three SVGs exist in `experiments/h1_validation/results/exp1/figs/`.

Cleanup: `rm -rf experiments/h1_validation/results/exp1/all.csv experiments/h1_validation/results/exp1/figs/`

- [ ] **Step 3: Commit**

```bash
git add experiments/h1_validation/exp1_gradient_cosine/run.py experiments/h1_validation/exp1_gradient_cosine/plot.py
git commit -m "feat(h1_validation/exp1): add plot.py (boxplot + bar + scatter)"
```

---

## Task 11: Implement `exp3_severity_sweep/run.py`

**Files:**
- Create: `experiments/h1_validation/exp3_severity_sweep/run.py`

- [ ] **Step 1: Write `run.py`**

Create `experiments/h1_validation/exp3_severity_sweep/run.py`:

```python
"""Exp 3 — Source-only mIoU sweep across corruption severity 1-5.

For each cell, invokes main_continual.py with the corresponding flags
(no --adapt so it's source-only evaluation), then reads the 1-round
Mean_mIoU out of the resulting results_all_rounds.txt.

Usage:
    python -m experiments.h1_validation.exp3_severity_sweep.run \
        --dataset VOC20_matched --corruption snow --severity 3 \
        [--out experiments/h1_validation/results/exp3/all.csv] \
        [--save_dir experiments/h1_validation/results/exp3/raw/<ds>_<corr>_s<sev>/] \
        [--skip_done]
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

from experiments.h1_validation.common import DATASET_REGISTRY, append_row, already_done


def parse_first_round_miou(results_path: str) -> float:
    """Read the first 'Round NN, ..., Mean_mIoU' line from results_all_rounds.txt."""
    with open(results_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = [x.strip() for x in lines[0].split(",")]
    mean_idx = header.index("Mean_mIoU")
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        if re.match(r"Round\s+\d+", parts[0]):
            return float(parts[mean_idx])
    raise RuntimeError(f"no Round row in {results_path}")


def build_cmd(args, entry, save_dir):
    cmd = [
        "python", "main_continual.py",
        "--dataset", entry["main_dataset_name"],
        "--data_dir", entry["data_dir"],
        "--method", "tent_continual",   # arbitrary; --adapt omitted → source-only
        "--ovss_type", "naclip",
        "--ovss_backbone", "ViT-L/14",
        "--corruption", args.corruption,
        "--severity", str(args.severity),
        "--continual_rounds", "1",
        "--save_dir", save_dir,
    ]
    kw = entry["prepare_data_kwargs"]
    # init_resize: list of two ints
    if "init_resize" in kw:
        cmd += ["--init_resize", str(kw["init_resize"][0]), str(kw["init_resize"][1])]
    if "patch_size" in kw:
        cmd += ["--patch_size", str(kw["patch_size"])]
    if "patch_stride" in kw:
        cmd += ["--patch_stride", str(kw["patch_stride"])]
    if "batch_size" in kw:
        cmd += ["--batch_size", str(kw["batch_size"])]
    if "num_workers" in kw:
        cmd += ["--workers", str(kw["num_workers"])]
    if "ann_file" in kw and kw["ann_file"]:
        # ann_file in registry is relative to data_dir
        ann_path = os.path.join(entry["data_dir"], kw["ann_file"])
        cmd += ["--ann_file", ann_path]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=[k for k, v in DATASET_REGISTRY.items() if v["kind"] == "synthetic"])
    ap.add_argument("--corruption", required=True)
    ap.add_argument("--severity", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/h1_validation/results/exp3/all.csv")
    ap.add_argument("--save_dir", default=None,
                    help="default: experiments/h1_validation/results/exp3/raw/<ds>_<corr>_s<sev>/")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    entry = DATASET_REGISTRY[args.dataset]
    if args.corruption not in entry["conditions"]:
        sys.exit(f"corruption {args.corruption!r} not in {entry['conditions']}")

    key = {"dataset": args.dataset, "corruption": args.corruption, "severity": args.severity}
    if args.skip_done and already_done(args.out, key):
        print(f"[exp3] skip done {args.dataset}/{args.corruption}/sev={args.severity}", flush=True)
        return

    save_dir = args.save_dir or os.path.join(
        "experiments/h1_validation/results/exp3/raw",
        f"{args.dataset}_{args.corruption}_s{args.severity}",
    )
    os.makedirs(save_dir, exist_ok=True)
    cmd = build_cmd(args, entry, save_dir)
    print(f"[exp3] running: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"main_continual.py failed (exit {e.returncode})")

    miou = parse_first_round_miou(os.path.join(save_dir, "results_all_rounds.txt"))
    append_row(args.out, {
        "dataset": args.dataset,
        "kind": entry["kind"],
        "corruption": args.corruption,
        "severity": args.severity,
        "miou": miou,
    })
    print(f"[exp3] DONE  {args.dataset}/{args.corruption}/sev={args.severity}  miou={miou:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on one VOC20_matched cell**

Run:

```bash
rm -f experiments/h1_validation/results/exp3/all.csv
CUDA_VISIBLE_DEVICES=0 python -m experiments.h1_validation.exp3_severity_sweep.run \
    --dataset VOC20_matched --corruption snow --severity 5
```

Expected: invokes `main_continual.py`, completes, writes one row to `experiments/h1_validation/results/exp3/all.csv`. Inspect:

```bash
cat experiments/h1_validation/results/exp3/all.csv
```

Expected schema: `dataset,kind,corruption,severity,miou` with one data row (miou around 71 based on the existing v20_acdc_matched/No_Adaptation result).

- [ ] **Step 3: Test `--skip_done`**

Re-run the same command with `--skip_done`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.h1_validation.exp3_severity_sweep.run \
    --dataset VOC20_matched --corruption snow --severity 5 --skip_done
```

Expected: prints `[exp3] skip done ...`, CSV unchanged.

Cleanup:

```bash
rm -rf experiments/h1_validation/results/exp3/all.csv experiments/h1_validation/results/exp3/raw/
```

- [ ] **Step 4: Commit**

```bash
git add experiments/h1_validation/exp3_severity_sweep/run.py
git commit -m "feat(h1_validation/exp3): add severity-sweep runner"
```

---

## Task 12: Implement `exp3_severity_sweep/plot.py` — 2 figures

**Files:**
- Create: `experiments/h1_validation/exp3_severity_sweep/plot.py`

- [ ] **Step 1: Write `plot.py`**

Create `experiments/h1_validation/exp3_severity_sweep/plot.py`:

```python
"""Exp 3 plots: severity decay curves + relative-drop bar.

Native datasets (no severity axis) appear as horizontal reference lines
with values hard-coded below — sourced from save/{Dataset}/No_Adaptation/
runs at the time of the design spec (2026-05-28).
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Native source mIoU references (constants, see spec §7.5)
NATIVE_REFS = {
    "ACDC": 23.34,
    "DarkZurich": 20.24,
    "NighttimeDriving": 31.01,
}

VOC_COLORS = {
    "snow": "#0ea5e9", "fog": "#0891b2", "frost": "#16a34a",
    "contrast": "#9333ea", "brightness": "#ea580c",
}
CITY_COLORS = {
    "snow": "#0284c7", "fog": "#0e7490", "frost": "#15803d",
    "contrast": "#7e22ce", "brightness": "#c2410c",
}


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["severity"] = int(r["severity"])
        r["miou"] = float(r["miou"])
    return rows


def plot_decay(rows, out_path):
    # group: {(dataset, corruption): [(severity, miou), ...]} sorted by severity
    series = defaultdict(list)
    for r in rows:
        series[(r["dataset"], r["corruption"])].append((r["severity"], r["miou"]))
    for k in series:
        series[k].sort()

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    handles = []
    for (ds, corr), pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        is_voc = ds == "VOC20_matched"
        color = (VOC_COLORS if is_voc else CITY_COLORS).get(corr, "#475569")
        ls = "-" if is_voc else "--"
        ax.plot(xs, ys, marker="o", color=color, linewidth=2.0, linestyle=ls, zorder=4)
        handles.append(Line2D([], [], color=color, linewidth=2.0, linestyle=ls,
                              label=f"{ds}/{corr}"))

    # Native reference lines (horizontal)
    native_handles = []
    for name, v in NATIVE_REFS.items():
        ax.axhline(v, color="#6b7280", linestyle=":", linewidth=1.2, zorder=2)
        native_handles.append(Line2D([], [], color="#6b7280", linestyle=":",
                                     label=f"{name} (native, source={v:.2f})"))

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Corruption severity", fontsize=11)
    ax.set_ylabel("Source mIoU (no adaptation)", fontsize=11)
    ax.set_title("Exp 3 — Source mIoU vs synthetic corruption severity\n"
                 "(flat curve ⇒ CLIP pretraining covered this; steep ⇒ OOD)",
                 fontsize=11, pad=10)
    ax.legend(handles=handles + native_handles, loc="best",
              fontsize=8, frameon=True, facecolor="white", edgecolor="#d1d5db")
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_relative_drop(rows, out_path):
    # build {(dataset, corruption): {severity: miou}}
    by_pair: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        by_pair[(r["dataset"], r["corruption"])][r["severity"]] = r["miou"]
    items = []
    for (ds, corr), sev_map in by_pair.items():
        if 1 not in sev_map or 5 not in sev_map:
            continue
        m1, m5 = sev_map[1], sev_map[5]
        drop = (m1 - m5) / m1 if m1 > 0 else 0.0
        items.append((ds, corr, drop, m1, m5))
    items.sort(key=lambda x: x[2])

    if not items:
        print("[plot] skip relative_drop_bar — need rows at both sev=1 and sev=5")
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(items) + 4), 5.5), dpi=160)
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    labels = [f"{ds}/{corr}" for ds, corr, _, _, _ in items]
    drops = [d for _, _, d, _, _ in items]
    colors = ["#0ea5e9" if ds == "VOC20_matched" else "#dc2626" for ds, _, _, _, _ in items]
    xs = np.arange(len(items))
    ax.bar(xs, drops, color=colors, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("(miou@sev1 − miou@sev5) / miou@sev1", fontsize=11)
    ax.set_title("Exp 3 — Relative mIoU drop from severity 1 → 5\n"
                 "(small drop ⇒ CLIP already robust ⇒ H1 supported)",
                 fontsize=11, pad=10)
    handles = [
        Line2D([], [], marker="s", linestyle="", color="#0ea5e9", label="VOC20_matched"),
        Line2D([], [], marker="s", linestyle="", color="#dc2626", label="Cityscapes"),
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/h1_validation/results/exp3/all.csv")
    ap.add_argument("--out_dir", default="experiments/h1_validation/results/exp3/figs/")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    print(f"[plot] {len(rows)} rows")
    plot_decay(rows, os.path.join(args.out_dir, "severity_decay_curves"))
    plot_relative_drop(rows, os.path.join(args.out_dir, "relative_drop_bar"))
    print(f"[plot] Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with a hand-written CSV**

```bash
mkdir -p experiments/h1_validation/results/exp3/
cat > experiments/h1_validation/results/exp3/all.csv <<'EOF'
dataset,kind,corruption,severity,miou
VOC20_matched,synthetic,snow,1,75.2
VOC20_matched,synthetic,snow,2,74.0
VOC20_matched,synthetic,snow,3,72.5
VOC20_matched,synthetic,snow,4,70.0
VOC20_matched,synthetic,snow,5,67.5
VOC20_matched,synthetic,fog,1,76.0
VOC20_matched,synthetic,fog,3,73.0
VOC20_matched,synthetic,fog,5,69.0
Cityscapes,synthetic,snow,1,21.0
Cityscapes,synthetic,snow,5,18.0
EOF
python -m experiments.h1_validation.exp3_severity_sweep.plot \
    --csv experiments/h1_validation/results/exp3/all.csv \
    --out_dir experiments/h1_validation/results/exp3/figs/
```

Expected: prints `[plot] 10 rows` and `[plot] Saved figures to ...`. Two PNGs + two SVGs in `figs/`.

Cleanup: `rm -rf experiments/h1_validation/results/exp3/all.csv experiments/h1_validation/results/exp3/figs/`

- [ ] **Step 3: Commit**

```bash
git add experiments/h1_validation/exp3_severity_sweep/plot.py
git commit -m "feat(h1_validation/exp3): add plot.py (severity decay + relative drop)"
```

---

## Task 13: Bash drivers — `run_exp1.sh`, `run_exp3.sh`, `plot_all.sh`

**Files:**
- Create: `experiments/h1_validation/bash/run_exp1.sh`
- Create: `experiments/h1_validation/bash/run_exp3.sh`
- Create: `experiments/h1_validation/bash/plot_all.sh`

- [ ] **Step 1: Write `run_exp1.sh`**

Create `experiments/h1_validation/bash/run_exp1.sh`:

```bash
#!/bin/bash
# Exp 1 driver — loops (dataset, condition) and appends to a single CSV.
# usage:
#   bash experiments/h1_validation/bash/run_exp1.sh
# optional env vars:
#   N=100              number of images per (dataset, condition) — or "all"
#   GPU=0              CUDA_VISIBLE_DEVICES
#   CSV=...            output CSV path
#   DATASETS="..."     subset of registry keys (default: all 5)
set -e

N=${N:-100}
GPU=${GPU:-0}
CSV=${CSV:-experiments/h1_validation/results/exp1/all.csv}
DATASETS=${DATASETS:-"ACDC DarkZurich NighttimeDriving VOC20_matched Cityscapes"}

mkdir -p "$(dirname $CSV)"
echo "[exp1] CSV=$CSV  N=$N  GPU=$GPU"
echo "[exp1] DATASETS=$DATASETS"
echo ""

for ds in $DATASETS; do
  conds=$(python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['$ds']['conditions']))")
  for cond in $conds; do
    echo "===== [exp1] $ds / $cond (n=$N) ====="
    CUDA_VISIBLE_DEVICES=$GPU python -m experiments.h1_validation.exp1_gradient_cosine.run \
        --dataset $ds --condition $cond --n $N --out $CSV --skip_done
    echo ""
  done
done

echo "[exp1] All done. CSV at $CSV"
```

- [ ] **Step 2: Write `run_exp3.sh`**

Create `experiments/h1_validation/bash/run_exp3.sh`:

```bash
#!/bin/bash
# Exp 3 driver — loops (synthetic dataset, corruption, severity).
# usage:
#   bash experiments/h1_validation/bash/run_exp3.sh
# optional env vars:
#   GPU=0
#   CSV=...
#   DATASETS="VOC20_matched Cityscapes"   subset of synthetic registry keys
#   SEVERITIES="1 2 3 4 5"
set -e

GPU=${GPU:-0}
CSV=${CSV:-experiments/h1_validation/results/exp3/all.csv}
DATASETS=${DATASETS:-"VOC20_matched Cityscapes"}
SEVERITIES=${SEVERITIES:-"1 2 3 4 5"}

mkdir -p "$(dirname $CSV)"
echo "[exp3] CSV=$CSV  GPU=$GPU"
echo "[exp3] DATASETS=$DATASETS  SEVERITIES=$SEVERITIES"
echo ""

for ds in $DATASETS; do
  corrs=$(python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(' '.join(R['$ds']['conditions']))")
  for corr in $corrs; do
    for sev in $SEVERITIES; do
      echo "===== [exp3] $ds / $corr / sev=$sev ====="
      CUDA_VISIBLE_DEVICES=$GPU python -m experiments.h1_validation.exp3_severity_sweep.run \
          --dataset $ds --corruption $corr --severity $sev --out $CSV --skip_done
      echo ""
    done
  done
done

echo "[exp3] All done. CSV at $CSV"
```

- [ ] **Step 3: Write `plot_all.sh`**

Create `experiments/h1_validation/bash/plot_all.sh`:

```bash
#!/bin/bash
# Generates all 5 figures from the two experiments.
# usage: bash experiments/h1_validation/bash/plot_all.sh
set -e

EXP1_CSV=${EXP1_CSV:-experiments/h1_validation/results/exp1/all.csv}
EXP3_CSV=${EXP3_CSV:-experiments/h1_validation/results/exp3/all.csv}
EXP1_DIR=${EXP1_DIR:-experiments/h1_validation/results/exp1/figs/}
EXP3_DIR=${EXP3_DIR:-experiments/h1_validation/results/exp3/figs/}

if [ -f "$EXP1_CSV" ]; then
  echo "[plot] exp1 ← $EXP1_CSV → $EXP1_DIR"
  python -m experiments.h1_validation.exp1_gradient_cosine.plot --csv $EXP1_CSV --out_dir $EXP1_DIR
else
  echo "[plot] skip exp1 (no $EXP1_CSV)"
fi

if [ -f "$EXP3_CSV" ]; then
  echo "[plot] exp3 ← $EXP3_CSV → $EXP3_DIR"
  python -m experiments.h1_validation.exp3_severity_sweep.plot --csv $EXP3_CSV --out_dir $EXP3_DIR
else
  echo "[plot] skip exp3 (no $EXP3_CSV)"
fi
```

- [ ] **Step 4: Make all bash scripts executable**

```bash
chmod +x experiments/h1_validation/bash/run_exp1.sh
chmod +x experiments/h1_validation/bash/run_exp3.sh
chmod +x experiments/h1_validation/bash/plot_all.sh
```

- [ ] **Step 5: Smoke-test the bash registry expressions actually evaluate**

```bash
bash -c 'python -c "from experiments.h1_validation.common.datasets import DATASET_REGISTRY as R; print(\" \".join(R[\"ACDC\"][\"conditions\"]))"'
```

Expected: `fog night rain snow`

- [ ] **Step 6: Commit**

```bash
git add experiments/h1_validation/bash/
git commit -m "feat(h1_validation/bash): add experiment drivers + plot_all"
```

---

## Task 14: Top-level README

**Files:**
- Create: `experiments/h1_validation/README.md`

- [ ] **Step 1: Write the README**

Create `experiments/h1_validation/README.md`:

```markdown
# H1 Validation Experiments

Tests the hypothesis **H1**: NA-CLIP (LAION-pretrained) has already seen images
visually similar to ImageNet-C-style synthetic corruptions in pretraining. Native
domain shifts (real night/adverse weather) are genuinely OOD.

Two experiments:

- **Exp 1 — Gradient cosine**: per-image cosine between TENT (entropy) gradient
  and supervised CE gradient over visual LayerNorm `{γ, β}`. Positive ⇒ entropy
  minimization aligned with correctness.
- **Exp 3 — Severity sweep**: source mIoU at corruption severity 1-5 on VOC20
  and Cityscapes; native datasets as horizontal references.

Spec: `docs/superpowers/specs/2026-05-28-h1-validation-experiments-design.md`.

## Layout

```
common/                  shared helpers (model, datasets, ln gradients, io)
exp1_gradient_cosine/    runner + plotter
exp3_severity_sweep/     runner + plotter
bash/                    drivers
results/                 generated CSVs and figures (gitignored)
```

## Run everything

```bash
# Exp 1 — 100 imgs / (dataset, condition); ~75 min total on RTX PRO 6000
bash experiments/h1_validation/bash/run_exp1.sh

# Exp 3 — VOC20 (~1 h) + Cityscapes (longer; cache rebuild)
bash experiments/h1_validation/bash/run_exp3.sh

# All 5 figures
bash experiments/h1_validation/bash/plot_all.sh
```

## Run subsets

```bash
# only ACDC and VOC20_matched, 50 imgs per condition, on GPU 2
DATASETS="ACDC VOC20_matched" N=50 GPU=2 bash experiments/h1_validation/bash/run_exp1.sh

# only VOC20 sev 1 and 5
DATASETS="VOC20_matched" SEVERITIES="1 5" bash experiments/h1_validation/bash/run_exp3.sh
```

## Add a new dataset

Edit `common/datasets.py` and add an entry to `DATASET_REGISTRY`:

```python
"MyNewDataset": {
    "kind": "native" | "synthetic",
    "main_dataset_name": "<name in utils/segmentation_datasets.py>",
    "data_dir": "<path>",
    "conditions": ["<sub-condition 1>", ...],
    "class_names": [...],
    "prepare_data_kwargs": {...},
},
```

Both bash drivers pick it up automatically. No other code changes needed.

## Add a new experiment

Copy `exp1_gradient_cosine/` to `expN_<name>/`, reuse `common/`, add a bash driver.
CSV schemas are per-experiment so no migration needed.

## Resume / partial reruns

Both runners support `--skip_done`. The bash drivers always pass this, so
re-launching is safe: completed cells are skipped, only missing cells run.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/h1_validation/README.md
git commit -m "docs(h1_validation): add top-level README"
```

---

## Task 15: End-to-end smoke run (validate the pipeline works on real data)

**Files:** none modified

- [ ] **Step 1: Exp 1 smoke — 2 conditions × 3 samples**

```bash
DATASETS="ACDC VOC20_matched" N=3 GPU=0 bash experiments/h1_validation/bash/run_exp1.sh
```

Expected: ~5 minutes wall-clock. CSV at `experiments/h1_validation/results/exp1/all.csv` has 24 rows (4 ACDC conds × 3 + 4 VOC corruptions × 3 = 24). `cos` values are finite. Inspect:

```bash
wc -l experiments/h1_validation/results/exp1/all.csv
head -5 experiments/h1_validation/results/exp1/all.csv
```

- [ ] **Step 2: Exp 3 smoke — VOC20 / snow / sev 1 and 5**

```bash
DATASETS="VOC20_matched" SEVERITIES="1 5" \
  bash experiments/h1_validation/bash/run_exp3.sh
```

Expected: 4 corruptions × 2 severities = 8 cells, ~30 minutes. CSV has 8 rows. Check that severity=1 mIoU > severity=5 mIoU for each corruption (sanity check on plumbing).

- [ ] **Step 3: Generate all figures from the smoke data**

```bash
bash experiments/h1_validation/bash/plot_all.sh
```

Expected: 3 figure pairs in `results/exp1/figs/`, 2 figure pairs in `results/exp3/figs/`. Open one PNG to eyeball.

- [ ] **Step 4: Clean smoke results so the real run starts fresh**

```bash
rm -rf experiments/h1_validation/results/exp1/all.csv
rm -rf experiments/h1_validation/results/exp1/figs/
rm -rf experiments/h1_validation/results/exp3/all.csv
rm -rf experiments/h1_validation/results/exp3/raw/
rm -rf experiments/h1_validation/results/exp3/figs/
```

- [ ] **Step 5: Commit the (now empty) results dir state if anything tracked changed**

No commit needed if .gitignore was set correctly in Task 3 — generated files are ignored. If `git status` shows nothing tracked changed, skip this step.

```bash
git status
```

---

## Self-Review

### Spec coverage check
- §1 Goal — covered by tasks 9-12 (the runners + plotters); README documents the goal
- §2 Scope — both experiments are implemented; severity plumbing is minimal patch only (tasks 1-2)
- §3 Background — informational; no code task needed
- §4 Repo layout — task 3 creates skeleton; tasks 4-14 fill it
- §5 Common module — tasks 4-8 implement io, ln_utils, datasets, model + re-exports
- §6 Exp 1 — tasks 9 (run.py), 10 (plot.py), 13 (bash)
- §7 Exp 3 — tasks 1-2 (core patch), 11 (run.py), 12 (plot.py), 13 (bash)
- §8 Result interpretation guide — documented in spec, not in code (correct: this is paper analysis, not implementation)
- §9 Extension paths — README (task 14) documents both new-dataset and new-experiment flows
- §10 Done criteria — every line maps to a task: skeleton (3), common modules (4-8), exp1 run+plot (9-10), exp3 run+plot (11-12), severity patch (1-2), bash drivers (13), README (14), smoke validation (15)
- §11 Open questions — informational
- §12 References — informational

All sections covered.

### Placeholder scan
- No "TBD" / "TODO" / "implement later" anywhere in the plan
- Every code step shows the actual code to write
- Every test step shows the exact command and expected output
- No "similar to Task N" deferrals — each task is self-contained

### Type consistency
- CSV schema for exp1: `dataset, kind, condition, idx, cos, norm_tent, norm_sup, n_pixels` — consistent across run.py (Task 9) and plot.py (Task 10)
- CSV schema for exp3: `dataset, kind, corruption, severity, miou` — consistent across run.py (Task 11) and plot.py (Task 12)
- `DATASET_REGISTRY` schema: `kind, main_dataset_name, data_dir, conditions, class_names, prepare_data_kwargs` — consistent between datasets.py (Task 6) and consumers (Tasks 7, 9, 11)
- `build_loader` signature stays `(dataset_key, condition, severity=5, n_samples=None, seed=0)` everywhere it's called
- `load_source_model` returns `(model, tokenize, ln_params, ln_names)` consistently
- `compute_text_features` returns shape `(1, num_classes, D)` matching what `model(x, text_features, True, ...)` expects per `adapt/tent_continual.py:127`

All consistent.
