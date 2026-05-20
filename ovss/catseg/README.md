# CAT-Seg Backbone Integration

This directory provides `CATSegWrapper`, a drop-in OVSS backbone that
matches the public interface of the modified CLIP in `ovss/clip/model.py`.
Selecting it requires only setting `OVSS_TYPE="catseg"` in any bash script
that already passes `--ovss_type` through to `main_continual.py`.

## Source

- Repo: <https://github.com/cvlab-kaist/CAT-Seg>
- Paper: Cho et al., *CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation*, CVPR 2024.
- License: MIT (compatible with this project).
- Extracted commit hash: **TBD-FILL-DURING-IMPLEMENTATION** (record before merging Task 6).

## Checkpoint download (manual user step)

Before running any CAT-Seg experiment, download the ViT-L/14 checkpoint
trained on COCO-Stuff and place it at:

```
data/.cache/catseg/model_large.pth
```

Download link is in the CAT-Seg repo README (Google Drive / HuggingFace).
The wrapper will raise `FileNotFoundError` with a pointer to this README
if the file is missing.

### Override path

Set the env var `CATSEG_CKPT_PATH` to use a different location:

```bash
CATSEG_CKPT_PATH=/data/shared/catseg/model_large.pth \
    bash bash/v20/tent_continual_catseg.sh
```

### SHA-256

Record the SHA-256 of the downloaded checkpoint here for reproducibility:

```
TBD-FILL-AFTER-DOWNLOAD
```

## Skipped state-dict keys

Some CAT-Seg checkpoint entries depend on the training class count
(171 for COCO-Stuff) and are loaded with `strict=False`. Keys reported
as "Unexpected" or "Missing" by `load_state_dict(strict=False)` during
the first successful load are listed here for reproducibility:

```
TBD-FILL-AFTER-FIRST-SUCCESSFUL-LOAD
```

## Interface contract

`CATSegWrapper` provides:

| Attribute / method | Purpose |
|---|---|
| `forward(x, text_x, text_ensemble=True, interpolate=False)` | Main forward. Returns `(logits, image_features, text_features)`. Logits shape `(#templates, B, #class, W, H)`. |
| `encode_text(tokens)` | CLIP text encoder. |
| `.visual` | Underlying CLIP `VisionTransformer`. All LayerNorm params here are trainable. |
| `.transformer`, `.ln_final`, `.token_embedding`, `.logit_scale` | CLIP text-side sub-modules (frozen by `adapt/*.py`). |
| `.aggregator` | Frozen CAT-Seg cost-aggregation transformer (always `.eval()`). |

The cost-aggregation transformer is **not** under `.visual` -- this
ensures `for m in wrapper.visual.modules(): if isinstance(m, nn.LayerNorm)`
naturally skips it, preserving the "CLIP backbone LN only" trainable
invariant.

## Sanity check

```bash
python -m ovss.catseg.test_wrapper
```

Runs the standalone sanity script. Exits 0 on success. Re-run after
any change in this directory.

## Bash scripts that use this backbone

| Script | Method | Save dir |
|---|---|---|
| `bash/v20/no_adapt_catseg.sh` | source baseline | `save/PascalVOC20Dataset/no_adapt_catseg/` |
| `bash/v20/mlmp_episodic_catseg.sh` | episodic upper bound | `save/PascalVOC20Dataset/mlmp_episodic_catseg/` |
| `bash/v20/tent_continual_catseg.sh` | TENT-continual | `save/PascalVOC20Dataset/tent_continual_catseg_weather/` |
| `bash/v20/tent_divgate_continual_catseg.sh` | TENT-DivGate | `save/PascalVOC20Dataset/tent_divgate_continual_catseg_h_thr_1.6/` |
