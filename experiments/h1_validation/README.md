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

Spec: [`docs/superpowers/specs/2026-05-28-h1-validation-experiments-design.md`](../../docs/superpowers/specs/2026-05-28-h1-validation-experiments-design.md).
Plan: [`docs/superpowers/plans/2026-05-28-h1-validation-experiments.md`](../../docs/superpowers/plans/2026-05-28-h1-validation-experiments.md).

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

# Exp 3 — VOC20 (~1 h) + Cityscapes (longer; corruption cache rebuild)
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
    "prepare_data_kwargs": {...},   # match the corresponding bash/<...>/no_adapt.sh
},
```

Both bash drivers pick it up automatically — no other code changes needed.

## Add a new experiment

Copy `exp1_gradient_cosine/` to `expN_<name>/`, reuse `common/`, add a bash driver.
CSV schemas are per-experiment so no migration needed.

## Resume / partial reruns

Both runners support `--skip_done`. The bash drivers always pass this, so
re-launching is safe: completed cells are skipped, only missing cells run.

## Implementation notes

- **fp16 vs fp32**: NA-CLIP weights are fp16; exp 1 casts logits to fp32 before
  loss + divides by `model.logit_scale.exp()` (≈ 100) to avoid softmax/CE
  underflow that otherwise produces zero gradients.
- **CE resolution**: exp 1 uses `interpolate=False` for both losses and downsamples
  GT to patch-token resolution (16×16) with `mode="nearest"`. This is the
  apples-to-apples comparison: same model output, same backward graph.
  `interpolate=True`'s bilinear upsample backward saturates fp16 and zeros the
  gradient — avoid it for gradient-comparison purposes.
- **Single-prompt**: exp 1 uses `"a photo of a {}"` (matches `REFERENCE_PROMPT`
  in `adapt/tent.py`) so the measured gradient corresponds to TENT/TENT-DivGate's
  actual runtime configuration (bash scripts don't pass `--prompt_dir`).
