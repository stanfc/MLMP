import copy
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader, ConcatDataset


def _worker_init_limit_threads(worker_id):
    """DataLoader worker_init_fn — limit each worker's PyTorch thread pool.

    Default behavior: every dataloader worker (a forked process) calls
    `torch.set_num_threads(cpu_count())` and ends up with ~256 threads on a
    256-core machine. With many concurrent experiments × many workers each,
    this saturates the user thread limit (ulimit -u) and triggers
    'res = 11' / 'Connection reset by peer' errors.

    This function caps each worker's PyTorch intra-op thread count to
    OMP_NUM_THREADS (default 4), reducing per-worker thread count from
    ~260 to ~10. Has no effect on dataloading throughput because PyTorch
    is not used inside the worker for matrix ops here.
    """
    n = int(os.environ.get('OMP_NUM_THREADS', '4'))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass

from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import Resize

from mmengine.registry import TRANSFORMS
from mmengine.registry import DATASETS
import mmengine.fileio as fileio

from mmseg.datasets import BaseSegDataset
from mmseg.datasets.transforms.loading import LoadAnnotations
from mmseg.datasets.transforms.formatting import PackSegInputs

from utils import mm_transforms
from utils.misc import custom_collate, get_cls_idx


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    
    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class ACDCDataset(BaseSegDataset):
    """ACDC dataset (Adverse Conditions Dataset with Correspondences).

    Real-world adverse condition images: fog, night, rain, snow.
    Same 19 semantic classes and label convention as Cityscapes.
    img_suffix  = '_rgb_anon.png'
    seg_suffix  = '_gt_labelTrainIds.png'
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/cityscapes.txt")

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


# ACDC label remapping for class-merging experiments.
# Original 19-class index (Cityscapes train IDs):
#   0:road  1:sidewalk  2:building  3:wall  4:fence  5:pole
#   6:traffic light  7:traffic sign  8:vegetation  9:terrain
#   10:sky  11:person  12:rider  13:car  14:truck  15:bus
#   16:train  17:motorcycle  18:bicycle
# Each list maps original-class-idx -> super-class-idx.
ACDC_3CLASS_REMAP = [
    0, 0, 1, 1, 1, 1,   # road sidewalk -> 0;  building wall fence pole -> 1
    1, 1, 1, 0,         # traffic light/sign vegetation -> 1;  terrain -> 0
    1,                  # sky -> 1
    2, 2, 2, 2, 2, 2, 2, 2,  # person..bicycle -> 2
]
# 6-class super-class names (acdc_6class.txt order):
#   0:"road"               ← road, sidewalk, terrain
#   1:"building"           ← building, wall, fence
#   2:"sign and pole"      ← pole, traffic light, traffic sign
#   3:"vegetation"         ← vegetation
#   4:"sky"                ← sky
#   5:"vehicle and person" ← person, rider, car, truck, bus, train, motorcycle, bicycle
ACDC_6CLASS_REMAP = [
    0, 0,                # road, sidewalk        -> 0 road
    1, 1, 1,             # building, wall, fence -> 1 building
    2, 2, 2,             # pole, traffic light, traffic sign -> 2 sign and pole
    3,                   # vegetation -> 3 vegetation
    0,                   # terrain    -> 0 road (grouped with ground)
    4,                   # sky        -> 4 sky
    5, 5, 5, 5, 5, 5, 5, 5,  # person..bicycle -> 5 vehicle and person
]

# 10-class super-class names (acdc_10class.txt order):
#   0:"road"          ← road
#   1:"sidewalk"      ← sidewalk, terrain
#   2:"building"      ← building, wall, fence
#   3:"sign and pole" ← pole, traffic light, traffic sign
#   4:"vegetation"    ← vegetation
#   5:"sky"           ← sky
#   6:"person"        ← person, rider
#   7:"car"           ← car
#   8:"large vehicle" ← truck, bus, train
#   9:"two-wheeler"   ← motorcycle, bicycle
ACDC_10CLASS_REMAP = [
    0,                  # road
    1,                  # sidewalk
    2, 2, 2,            # building, wall, fence -> 2 building
    3, 3, 3,            # pole, traffic light, traffic sign -> 3 sign and pole
    4,                  # vegetation
    1,                  # terrain -> 1 sidewalk
    5,                  # sky
    6, 6,               # person, rider -> 6 person
    7,                  # car
    8, 8, 8,            # truck, bus, train -> 8 large vehicle
    9, 9,               # motorcycle, bicycle -> 9 two-wheeler
]


@DATASETS.register_module()
class ACDCMerged3Dataset(ACDCDataset):
    """ACDC with 19 classes merged into 3 super-classes (flat / vertical / movable).

    Pipeline must include {'type': 'RemapLabels', 'mapping': ACDC_3CLASS_REMAP}
    so the GT label is remapped at load time. Image data is identical to ACDCDataset.
    """
    METAINFO = dict(
        classes=('flat surface', 'vertical structure', 'movable object'),
        palette=[[128, 64, 128], [70, 70, 70], [220, 20, 60]],
    )
    class_extensions, extentions_to_real_class_idx = get_cls_idx(
        "utils/class_extensions/acdc_3class.txt")


@DATASETS.register_module()
class ACDCMerged6Dataset(ACDCDataset):
    """ACDC with 19 classes merged into 6 OVSS-natural super-classes.
    Class names (in order, as used directly as CLIP prompts):
      road / building / sign and pole / vegetation / sky / vehicle and person
    """
    METAINFO = dict(
        classes=('road', 'building', 'sign and pole',
                 'vegetation', 'sky', 'vehicle and person'),
        palette=[[128, 64, 128], [70, 70, 70], [153, 153, 153],
                 [107, 142, 35], [70, 130, 180], [0, 0, 142]],
    )
    class_extensions, extentions_to_real_class_idx = get_cls_idx(
        "utils/class_extensions/acdc_6class.txt")


@DATASETS.register_module()
class ACDCMerged10Dataset(ACDCDataset):
    """ACDC with 19 classes merged into 10 OVSS-natural super-classes.
    Class names (used directly as CLIP prompts):
      road / sidewalk / building / sign and pole / vegetation / sky /
      person / car / large vehicle / two-wheeler
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'sign and pole',
                 'vegetation', 'sky', 'person', 'car',
                 'large vehicle', 'two-wheeler'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [153, 153, 153],
                 [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142],
                 [0, 60, 100], [119, 11, 32]],
    )
    class_extensions, extentions_to_real_class_idx = get_cls_idx(
        "utils/class_extensions/acdc_10class.txt")


@DATASETS.register_module()
class COCOStuffDataset(BaseSegDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=(
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood'),
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]])
    
    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/coco_stuff.txt")


    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class PascalVOC21Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/voc21.txt")

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class COCOObjectDataset(BaseSegDataset):
    """
    Implementation borrowed from TCL (https://github.com/kakaobrain/tcl) and GroupViT (https://github.com/NVlabs/GroupViT)
    COCO-Object dataset.
    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    METAINFO = dict(
        classes=('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        palette=[[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/coco_object.txt")


    def __init__(self, **kwargs):
        super(COCOObjectDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)


@DATASETS.register_module()
class PascalVOC20Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[128, 0, 0], [0, 128, 0], [0, 0, 192],
                 [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 128, 64],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/voc20.txt")


    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class PascalContext60Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bag', 'bed', 'bedclothes',
                 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
                 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
                 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
                 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
                 'sofa', 'table', 'track', 'train', 'tree', 'truck',
                 'tvmonitor', 'wall', 'water', 'window', 'wood'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/context60.txt")

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PascalContext59Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    class_extensions, extentions_to_real_class_idx = get_cls_idx("utils/class_extensions/context59.txt")

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)



CLIP_MEAN = [122.7709, 116.7460, 104.0937]
CLIP_STD  = [68.5005, 66.6322, 70.3232]


# Register the modules with TRANSFORMS
TRANSFORMS.register_module(module=LoadImageFromFile, force=True)
TRANSFORMS.register_module(module=Resize, force=True)
TRANSFORMS.register_module(module=LoadAnnotations, force=True)
TRANSFORMS.register_module(module=PackSegInputs, force=True)


### make them args in the main.py and pass them to the file
data_dir  = ''
batch_size = 2 # number of loaded images
resize = (224, 224) # the size of the image after resizing => it can be vertical or horizontal depending on the image so => (560, 448) or (448, 560)
patch_size = (224, 224) # the size of the patch that will be extracted from the resized image
patch_stride = 112 # the stride of the patch extraction



mm_cocostuff_cfg =  { 
    'type': 'COCOStuffDataset', 
    'data_root': data_dir,  
    'data_prefix': {'img_path': 'images/val2017', 'seg_map_path': 'annotations/val2017'}, 
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_cocoobject_cfg =  { 
    'type': 'COCOObjectDataset', 
    'data_root': data_dir,  
    'data_prefix': {'img_path': 'images/val2017', 'seg_map_path': 'annotations/val2017'}, 
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

mm_cityscapes_cfg =  {
    'type': 'CityscapesDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'leftImg8bit/val', 'seg_map_path': 'gtFine/val'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
    }

# ACDC: data_prefix is filled in dynamically by prepare_data() using corruption name
mm_acdc_cfg_base = {
    'type': 'ACDCDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': '', 'seg_map_path': ''},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

# Class-merging variants: same as base, but with a RemapLabels step right after
# LoadAnnotations so the GT label is collapsed before metric computation.
mm_acdc_3class_cfg_base = {
    'type': 'ACDCMerged3Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': '', 'seg_map_path': ''},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                # IMPORTANT: keep ResizeAndPatchify at index 2 — prepare_data
                # patches that slot to inject init_resize/patch_size/stride.
                # RemapLabels is placed AFTER ResizeAndPatchify (resize uses
                # nearest-neighbor on the GT, so original IDs are preserved
                # and remapping is equivalent on a per-pixel basis).
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'RemapLabels', 'mapping': ACDC_3CLASS_REMAP},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}
mm_acdc_6class_cfg_base = {
    'type': 'ACDCMerged6Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': '', 'seg_map_path': ''},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'RemapLabels', 'mapping': ACDC_6CLASS_REMAP},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}
mm_acdc_10class_cfg_base = {
    'type': 'ACDCMerged10Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': '', 'seg_map_path': ''},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'RemapLabels', 'mapping': ACDC_10CLASS_REMAP},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

mm_pascalvoc20_cfg = {
    'type': 'PascalVOC20Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClass'},
    'ann_file': 'ImageSets/Segmentation/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

mm_pascalvoc21_cfg = {
    'type': 'PascalVOC21Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClass'},
    'ann_file': 'ImageSets/Segmentation/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}    


mm_pascalcontect59_cfg = {
    'type': 'PascalContext59Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClassContext'},
    'ann_file': 'ImageSets/SegmentationContext/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations', 'reduce_zero_label':True},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}    


mm_pascalcontect60_cfg = {
    'type': 'PascalContext60Dataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'JPEGImages', 'seg_map_path': 'SegmentationClassContext'},
    'ann_file': 'ImageSets/SegmentationContext/val.txt',
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}    




def prepare_data(dataset, data_dir, init_resize, patch_size, patch_stride, corruption="original", batch_size=128, num_workers=1, shuffle=True, corruption_cache_dir=None, split="val", subset_size=None, subset_seed=0, corruption_severity=5, acdc_overlay_corruption=None):
    
    # # print everything
    # print("\n+++++++ Data Preparation +++++++")
    # print(f"Dataset:           {dataset}")
    # print(f"Data directory:    {data_dir}")
    # print(f"Initial resize:    {init_resize}")
    # print(f"Patch size:        {patch_size}")
    # print(f"Patch stride:      {patch_stride}")
    # print(f"Corruption:        {corruption}")
    # print(f"Batch size:        {batch_size}")
    # print(f"Number of workers: {num_workers}")
    # print("----------------------------------------")


    if init_resize is None:
        assert batch_size == 1, "Batch size must be 1 if init_resize is None"

    if dataset == "COCOStuffDataset":
        mm_config = copy.deepcopy(mm_cocostuff_cfg)
    elif dataset == "COCOObjectDataset":
        mm_config = copy.deepcopy(mm_cocoobject_cfg)
    elif dataset == "CityscapesDataset":
        mm_config = copy.deepcopy(mm_cityscapes_cfg)
    elif dataset == "ACDCDataset":
        mm_config = copy.deepcopy(mm_acdc_cfg_base)
        # ACDC condition is encoded in the data path, not as a synthetic transform
        mm_config['data_prefix']['img_path'] = f'rgb_anon/{corruption}/{split}'
        mm_config['data_prefix']['seg_map_path'] = f'gt/{corruption}/{split}'
    elif dataset == "ACDCMerged3Dataset":
        mm_config = copy.deepcopy(mm_acdc_3class_cfg_base)
        mm_config['data_prefix']['img_path'] = f'rgb_anon/{corruption}/{split}'
        mm_config['data_prefix']['seg_map_path'] = f'gt/{corruption}/{split}'
    elif dataset == "ACDCMerged6Dataset":
        mm_config = copy.deepcopy(mm_acdc_6class_cfg_base)
        mm_config['data_prefix']['img_path'] = f'rgb_anon/{corruption}/{split}'
        mm_config['data_prefix']['seg_map_path'] = f'gt/{corruption}/{split}'
    elif dataset == "ACDCMerged10Dataset":
        mm_config = copy.deepcopy(mm_acdc_10class_cfg_base)
        mm_config['data_prefix']['img_path'] = f'rgb_anon/{corruption}/{split}'
        mm_config['data_prefix']['seg_map_path'] = f'gt/{corruption}/{split}'
    elif dataset == "PascalVOC20Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc20_cfg)
    elif dataset == "PascalVOC21Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc21_cfg)
    elif dataset == "PascalContext59Dataset":
        mm_config = copy.deepcopy(mm_pascalcontect59_cfg)
    elif dataset == "PascalContext60Dataset":
        mm_config = copy.deepcopy(mm_pascalcontect60_cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    ### add specified configs
    
    mm_config['data_root'] = data_dir
    mm_config['pipeline'][2]['resize'] = init_resize
    mm_config['pipeline'][2]['patch_size'] = patch_size
    mm_config['pipeline'][2]['patch_stride'] = patch_stride


    ### add corruption to the pipline
    # Find the index of 'LoadImageFromFile' in the pipeline
    # ACDC: condition is encoded in the data path already — no synthetic transform needed
    # UNLESS acdc_overlay_corruption is set, in which case we add a synthetic
    # corruption on TOP of the weather condition (real shift + synthetic noise).
    _acdc_conditions = {'fog', 'night', 'rain', 'snow'}
    _acdc_datasets = {"ACDCDataset", "ACDCMerged3Dataset",
                      "ACDCMerged6Dataset", "ACDCMerged10Dataset"}

    # Decide which synthetic corruption (if any) to overlay on top of the image.
    overlay_corruption = None
    if dataset in _acdc_datasets:
        if acdc_overlay_corruption:
            overlay_corruption = acdc_overlay_corruption
            print(f"+ ACDC overlay corruption '{overlay_corruption}' (severity {corruption_severity})"
                  f" on top of condition '{corruption}'")
        else:
            print("No corruption added to the pipeline (ACDC weather only)")
    elif corruption == "original":
        print("No corruption added to the pipeline")
    else:
        overlay_corruption = corruption

    if overlay_corruption is not None:
        load_image_index = next(
            (i for i, transform in enumerate(mm_config['pipeline']) if transform['type'] == 'LoadImageFromFile'),
            None
        )
        if load_image_index is not None:
            corrupt_transform = {
                'type': 'CorruptTransform',
                'corruption_severity': int(corruption_severity),
                'corruption_name': overlay_corruption,
                'cache_dir': corruption_cache_dir or osp.join(osp.dirname(data_dir.rstrip('/')), '.cache', 'corruptions'),
            }
            mm_config['pipeline'].insert(load_image_index + 1, corrupt_transform)
            print(f"+ Corruption '{overlay_corruption}' (sev {corruption_severity}) added to the pipeline")
        else:
            raise ValueError("LoadImageFromFile not found in the pipeline")

    ### bulid the dataset from the config using mmseg registry
    # ACDC supports a combined split like "train+val" — build each part
    # separately and ConcatDataset them. METAINFO/ignore_index are taken
    # from the first part (identical across ACDC splits).
    if "+" in split and dataset in _acdc_datasets:
        parts = split.split("+")
        sub_datasets = []
        for s in parts:
            sub_cfg = copy.deepcopy(mm_config)
            sub_cfg['data_prefix']['img_path'] = f'rgb_anon/{corruption}/{s}'
            sub_cfg['data_prefix']['seg_map_path'] = f'gt/{corruption}/{s}'
            sub_datasets.append(DATASETS.build(sub_cfg))
        dataset = ConcatDataset(sub_datasets)
        dataset.METAINFO = sub_datasets[0].METAINFO
        dataset.ignore_index = sub_datasets[0].ignore_index
        dataset.class_extensions = getattr(sub_datasets[0], 'class_extensions', None)
        dataset.extentions_to_real_class_idx = getattr(
            sub_datasets[0], 'extentions_to_real_class_idx', None)
    else:
        dataset = DATASETS.build(mm_config)

    ### Optional fixed-size subset (deterministic by subset_seed).
    # Use this to align update count across datasets with different val sizes
    # (e.g. ACDC ~100 imgs/condition vs Cityscapes 500 vs VOC 1449).
    if subset_size is not None and subset_size > 0:
        import numpy as _np
        full_len = len(dataset)
        n = min(int(subset_size), full_len)
        if n < full_len:
            rng = _np.random.RandomState(int(subset_seed))
            keep_idx = sorted(rng.choice(full_len, n, replace=False).tolist())
            from torch.utils.data import Subset as _Subset
            # Preserve METAINFO / ignore_index that downstream code may read.
            meta = dataset.METAINFO
            ig = getattr(dataset, 'ignore_index', None)
            cext = getattr(dataset, 'class_extensions', None)
            cext_map = getattr(dataset, 'extentions_to_real_class_idx', None)
            dataset = _Subset(dataset, keep_idx)
            dataset.METAINFO = meta
            if ig is not None: dataset.ignore_index = ig
            if cext is not None: dataset.class_extensions = cext
            if cext_map is not None: dataset.extentions_to_real_class_idx = cext_map
            print(f"+ Subset: kept {n}/{full_len} samples (seed={subset_seed})")

    ### bulid the dataloader
    # if num_workers == 0:
    #     persistent_workers = False
    # else:
    #     persistent_workers = True
    
    persistent_workers = False

    # Use 'spawn' multiprocessing context for DataLoader workers. Default is
    # 'fork', which inherits the main process's PyTorch thread pool (~256 threads
    # on a 256-core box). 'spawn' creates a fresh Python process that re-reads
    # OMP_NUM_THREADS / MKL_NUM_THREADS from the env, so each worker only opens
    # ~10 threads instead of ~260.
    # Trade-off: spawn workers take ~5-10s to start (re-import torch/numpy)
    # vs near-instant for fork. Acceptable for 150-round runs.
    import torch.multiprocessing as _tmp
    _spawn_ctx = _tmp.get_context('spawn') if num_workers > 0 else None

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=custom_collate, persistent_workers=persistent_workers, pin_memory=True,
                            shuffle=shuffle,
                            worker_init_fn=_worker_init_limit_threads,
                            multiprocessing_context=_spawn_ctx)

    classes = dataset.METAINFO['classes']

    return dataloader, classes

    
