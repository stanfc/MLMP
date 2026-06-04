"""Dataset registry shared by both experiments.

To add a new dataset:
1. Add an entry to DATASET_REGISTRY following the schema below.
2. Both exp1 and exp3 bash drivers pick it up automatically via the
   `python -c "from .datasets import DATASET_REGISTRY..."` query.
"""
from __future__ import annotations
from typing import Any, Iterator
from torch.utils.data import DataLoader

from utils.segmentation_datasets import prepare_data


# 19 Cityscapes classes (shared by ACDC, DarkZurich, NighttimeDriving,
# Cityscapes). Order matches CityscapesDataset's METAINFO.
CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# 20 VOC classes (no background, per PascalVOC20Dataset METAINFO)
VOC_20 = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Full ImageNet-C corruption set (15 standard corruptions, category order:
# noise / blur / weather / digital). Matches utils.imagecorruptions
# get_corruption_names() and the CORRUPTIONS_LIST in
# bash/cityscapes_continual/*.sh. Used as the `conditions` for synthetic
# datasets so both exp1 (cosine @ sev=5) and exp3 (severity sweep) cover the
# same corruptions.
IMAGENET_C_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise",          # noise
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",  # blur
    "snow", "frost", "fog", "brightness", "contrast",          # weather
    "elastic_transform", "pixelate", "jpeg_compression",       # digital
]


# Driving-style preprocessing — every existing bash script uses the
# 224×224 patch / stride=112 setup even for 1120×560 inputs.
# Source: bash/{ACDC_10_round,dark_zurich,nighttime_driving,cityscapes_continual}/no_adapt.sh
_DRIVING_KWARGS = dict(
    init_resize=[1120, 560],
    patch_size=[224, 224],
    patch_stride=112,
    batch_size=1,
    num_workers=1,
    shuffle=False,
)

# VOC kwargs sourced from bash/v20_acdc_matched/no_adapt.sh
_VOC_KWARGS = dict(
    init_resize=[224, 224],
    patch_size=[224, 224],
    patch_stride=112,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    ann_file="data/VOC/VOC2012/ImageSets/Segmentation/val_subset_101_seed0.txt",
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
        "conditions": IMAGENET_C_15,
        "class_names": VOC_20,
        "prepare_data_kwargs": _VOC_KWARGS,
    },
    "Cityscapes": {
        "kind": "synthetic",
        "main_dataset_name": "CityscapesDataset",
        # Bash scripts use the singular folder name "Cityscape/" — preserved here
        "data_dir": "data/Cityscape/",
        "conditions": IMAGENET_C_15,
        "class_names": CITYSCAPES_19,
        "prepare_data_kwargs": _DRIVING_KWARGS,
    },
}


def build_loader(
    dataset_key: str,
    condition: str,
    severity: int = 5,
    seed: int = 0,
) -> DataLoader:
    """Build a DataLoader for one (dataset, condition).

    - native datasets: `severity` is ignored; `condition` is the sub-condition
      name passed as `corruption=` to prepare_data.
    - synthetic datasets: `condition` is the ImageNet-C corruption name;
      severity is plumbed through.
    - Iteration ordering is deterministic (shuffle=False in prepare_data_kwargs).
    - Callers cap sample count by `break`-ing after N batches; we deliberately
      do not Subset-wrap because prepare_data installs a custom collate_fn that
      Subset would bypass.
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

    loader, _ = prepare_data(
        entry["main_dataset_name"],
        entry["data_dir"],
        corruption=condition,
        **kwargs,
    )
    return loader


def iterate_limited(loader: DataLoader, n_samples: int | str | None) -> Iterator:
    """Yield (idx, batch) pairs from `loader`, stopping after `n_samples`.

    n_samples: None or "all" → full iteration; int → first N batches.
    """
    if n_samples is None or n_samples == "all":
        for idx, batch in enumerate(loader):
            yield idx, batch
    else:
        n = int(n_samples)
        for idx, batch in enumerate(loader):
            if idx >= n:
                return
            yield idx, batch


def get_class_names(dataset_key: str) -> list[str]:
    """Convenience accessor used by load_source_model callers."""
    return list(DATASET_REGISTRY[dataset_key]["class_names"])
