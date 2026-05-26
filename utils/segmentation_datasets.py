import copy
import os.path as osp

from torch.utils.data import DataLoader

from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import Resize

from mmengine.registry import TRANSFORMS
from mmengine.registry import DATASETS
from mmseg.registry import DATASETS as MMSEG_DATASETS
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


@MMSEG_DATASETS.register_module(force=True)  # override mmseg builtin (no class_extensions)
@DATASETS.register_module()
class DarkZurichDataset(BaseSegDataset):
    """Dark Zurich dataset (val/night split only).

    Real-world night driving images from Zurich, anonymised (faces/plates blurred).
    Same 19 Cityscapes classes and label convention as ACDC.
    img_suffix     = '_rgb_anon.png'
    seg_map_suffix = '_gt_labelTrainIds.png'

    Registered with force=True to mmseg's DATASETS because mmseg has a builtin
    DarkZurichDataset that gets auto-imported. The builtin lacks class_extensions
    (required by the --class_extensions flag in main_continual.py).
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


@DATASETS.register_module()
class NighttimeDrivingDataset(BaseSegDataset):
    """Nighttime Driving Test dataset (Dai & Van Gool, ICCV 2018).

    Real-world night driving images with coarse Cityscapes-style GT.
    Same 19 Cityscapes classes and label convention as Cityscapes.
    img_suffix     = '_leftImg8bit.png'
    seg_map_suffix = '_gtCoarse_labelTrainIds.png'
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
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtCoarse_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


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

# Dark Zurich: single val/night split; condition is path-encoded (no CorruptTransform).
mm_darkzurich_cfg = {
    'type': 'DarkZurichDataset',
    'data_root': data_dir,  # overridden by prepare_data() per call
    'data_prefix': {'img_path': 'rgb_anon/val/night',
                    'seg_map_path': 'gt/val/night'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': CLIP_MEAN, 'std': CLIP_STD},
                ]
}

# Nighttime Driving: Cityscapes-style suffix + gtCoarse_daytime_trainvaltest folder.
mm_nightdriving_cfg = {
    'type': 'NighttimeDrivingDataset',
    'data_root': data_dir,
    'data_prefix': {'img_path': 'leftImg8bit/test/night',
                    'seg_map_path': 'gtCoarse_daytime_trainvaltest/test/night'},
    'pipeline': [{'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
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




def prepare_data(dataset, data_dir, init_resize, patch_size, patch_stride, corruption="original", batch_size=128, num_workers=1, shuffle=True, corruption_cache_dir=None, ann_file=None):
    
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
        mm_config['data_prefix']['img_path'] = f'rgb_anon/{corruption}/val'
        mm_config['data_prefix']['seg_map_path'] = f'gt/{corruption}/val'
    elif dataset == "PascalVOC20Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc20_cfg)
    elif dataset == "PascalVOC21Dataset":
        mm_config = copy.deepcopy(mm_pascalvoc21_cfg)
    elif dataset == "PascalContext59Dataset":
        mm_config = copy.deepcopy(mm_pascalcontect59_cfg)
    elif dataset == "PascalContext60Dataset":
        mm_config = copy.deepcopy(mm_pascalcontect60_cfg)
    elif dataset == "DarkZurichDataset":
        mm_config = copy.deepcopy(mm_darkzurich_cfg)
    elif dataset == "NighttimeDrivingDataset":
        mm_config = copy.deepcopy(mm_nightdriving_cfg)
    elif dataset == "DZ_ND_Combined":
        # Combined-stream mode: args.data_dir is IGNORED. Sub-paths are
        # hardcoded per-condition because DZ and ND live in separate parents.
        # `corruption` is overloaded to name the sub-dataset.
        if corruption == "dark_zurich":
            mm_config = copy.deepcopy(mm_darkzurich_cfg)
            mm_config['data_root'] = "data/Dark_Zurich_val_anon/"
        elif corruption == "nighttime_driving":
            mm_config = copy.deepcopy(mm_nightdriving_cfg)
            mm_config['data_root'] = "data/NighttimeDrivingTest/"
        else:
            raise ValueError(
                f"DZ_ND_Combined: unknown sub-dataset '{corruption}', "
                f"expected 'dark_zurich' or 'nighttime_driving'")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    ### add specified configs
    
    # DZ_ND_Combined sets data_root per-condition in its dispatch branch;
    # the blanket assignment would clobber it. Other datasets behave as before.
    if dataset != "DZ_ND_Combined":
        mm_config['data_root'] = data_dir
    mm_config['pipeline'][2]['resize'] = init_resize
    mm_config['pipeline'][2]['patch_size'] = patch_size
    mm_config['pipeline'][2]['patch_stride'] = patch_stride


    ### add corruption to the pipline
    # Find the index of 'LoadImageFromFile' in the pipeline
    # Native-shift datasets encode the condition in the data path itself
    # (no synthetic CorruptTransform needed).
    _NATIVE_SHIFT_DATASETS = (
        "ACDCDataset", "DarkZurichDataset", "NighttimeDrivingDataset",
        "DZ_ND_Combined",
    )
    _acdc_conditions = {'fog', 'night', 'rain', 'snow'}
    if dataset in _NATIVE_SHIFT_DATASETS or corruption == "original":
        print("No corruption added to the pipeline")
    else:
        load_image_index = next(
            (i for i, transform in enumerate(mm_config['pipeline']) if transform['type'] == 'LoadImageFromFile'),
            None
        )  
        # Insert the new transform right after 'LoadImageFromFile'
        if load_image_index is not None:
            corrupt_transform = {
                'type': 'CorruptTransform',
                'corruption_severity': 5,
                'corruption_name': corruption,
                'cache_dir': corruption_cache_dir or osp.join(osp.dirname(data_dir.rstrip('/')), '.cache', 'corruptions'),
            }
            mm_config['pipeline'].insert(load_image_index + 1, corrupt_transform)

            print(f"+ Corruption '{corruption}' added to the pipeline")
        else:
            raise ValueError("LoadImageFromFile not found in the pipeline")

    ### override ann_file (used by bash/v20_acdc_matched/ to load a deterministic
    ### subset split for cross-dataset comparability). VOC20/VOC21 only.
    ### mmseg joins data_root with ann_file, so we accept either a relative path
    ### or a full path that includes data_root and normalise to relative.
    if ann_file is not None:
        if dataset not in ("PascalVOC20Dataset", "PascalVOC21Dataset"):
            print(f"+++ ann_file override ignored: {dataset} does not use ann_file")
        else:
            ann_file_rel = ann_file
            if ann_file_rel.startswith(data_dir):
                ann_file_rel = ann_file_rel[len(data_dir):]
            elif osp.isabs(ann_file_rel):
                try:
                    ann_file_rel = osp.relpath(ann_file_rel, data_dir)
                except ValueError:
                    pass
            mm_config['ann_file'] = ann_file_rel
            print(f"+++ ann_file overridden -> {ann_file_rel}")

    ### bulid the dataset from the config using mmseg registry
    dataset = DATASETS.build(mm_config)

    ### bulid the dataloader
    # if num_workers == 0:
    #     persistent_workers = False
    # else:
    #     persistent_workers = True
    
    persistent_workers = False

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                            collate_fn=custom_collate, persistent_workers=persistent_workers, pin_memory=True,
                            shuffle=shuffle)

    classes = dataset.METAINFO['classes']

    return dataloader, classes

    
