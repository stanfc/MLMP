#!/usr/bin/env python3
"""Generate the `_instanceTrainIds.png` semantic label maps that
`COCOObjectDataset` (utils/segmentation_datasets.py) expects, from COCO's raw
instance annotations (`instances_val2017.json`).

COCO-Object = 1 background class (train_id 0) + the 80 COCO "thing" categories
(train_id 1..80, in ascending COCO category-id order -- this is exactly the
order of COCOObjectDataset.METAINFO['classes'][1:], verified against
pycocotools' loadCats(sorted(getCatIds()))).

For each image, paint annotations largest-area-first so smaller (typically
closer/more specific) instances end up on top when masks overlap -- the
standard convention used by prior open-vocab segmentation work (TCL/GroupViT)
that this dataset class's docstring borrows from.

Usage:
    python scripts/make_coco_object_labels.py \
        --data_dir .data/coco/ --split val2017
"""
import argparse
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='.data/coco/')
    p.add_argument('--split', type=str, default='val2017')
    args = p.parse_args()

    ann_json = os.path.join(args.data_dir, 'annotations', f'instances_{args.split}.json')
    out_dir = os.path.join(args.data_dir, 'annotations', args.split)
    os.makedirs(out_dir, exist_ok=True)

    coco = COCO(ann_json)
    cat_ids = sorted(coco.getCatIds())
    assert len(cat_ids) == 80, f"expected 80 COCO thing categories, got {len(cat_ids)}"
    cat_id_to_train_id = {cid: i + 1 for i, cid in enumerate(cat_ids)}  # 0 reserved for background

    img_ids = coco.getImgIds()
    written, skipped = 0, 0
    for img_id in tqdm(img_ids, desc='coco-object labels'):
        info = coco.loadImgs(img_id)[0]
        file_name = info['file_name']
        stem = os.path.splitext(file_name)[0]
        out_path = os.path.join(out_dir, f'{stem}_instanceTrainIds.png')
        if os.path.exists(out_path):
            skipped += 1
            continue

        h, w = info['height'], info['width']
        label = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        anns = sorted(anns, key=lambda a: a.get('area', 0), reverse=True)
        for ann in anns:
            train_id = cat_id_to_train_id.get(ann['category_id'])
            if train_id is None:
                continue
            m = coco.annToMask(ann)
            label[m == 1] = train_id

        Image.fromarray(label, mode='L').save(out_path)
        written += 1

    print(f"done: {written} written, {skipped} already existed, {len(img_ids)} total")


if __name__ == '__main__':
    main()
