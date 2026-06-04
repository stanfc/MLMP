
import os
import numpy as np

import torch

import mmcv
import mmengine.fileio as fileio
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

from mmcv.transforms import BaseTransform, TRANSFORMS
from mmcv.transforms import to_tensor


from utils.imagecorruptions import corrupt
from utils.imagecorruptions import get_corruption_names


@TRANSFORMS.register_module()
class CorruptTransform(BaseTransform):
    def __init__(self, corruption_name: str, corruption_severity: int = 5,
                 cache_dir: str = None):
        super().__init__()
        self.corruption_name = corruption_name
        self.corruption_severity = corruption_severity
        self.cache_dir = cache_dir  # e.g. ".cache/corruptions"

        if self.corruption_name not in get_corruption_names():
            raise ValueError(f"Corruption name {self.corruption_name} is not valid. \nchoose from {get_corruption_names()}")

    def _cache_path(self, img_path: str) -> str:
        import hashlib
        img_hash = hashlib.md5(img_path.encode()).hexdigest()
        fname = f"{img_hash}_{self.corruption_name}_s{self.corruption_severity}.npy"
        return os.path.join(self.cache_dir, fname)

    def transform(self, results: dict) -> dict:
        """
        Args:
            results (dict): The input data dictionary.
        Returns:
            dict: The corrupted data dictionary.

        Note that the input image should be numpy array (bgr or rgb).
        """

        img = results['img']
        img_index = results['sample_idx']

        # Try loading from cache
        if self.cache_dir is not None:
            cache_path = self._cache_path(results.get('img_path', str(img_index)))
            if os.path.exists(cache_path):
                results['img'] = np.load(cache_path)
                return results

        # Save the current RNG state
        rng_state = np.random.get_state()

        # Set the seed based on the index to ensure reproducibility
        np.random.seed(img_index)

        # Corrupt the image
        corrupted = corrupt(img, severity=self.corruption_severity, corruption_name=self.corruption_name)

        # Restore the original RNG state
        np.random.set_state(rng_state)

        # Save to cache
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, corrupted)

        results['img'] = corrupted
        return results
    

@TRANSFORMS.register_module()
class ResizeAndPatchify(BaseTransform):
    """
    ResizeAndPatchify is a transformation class that resizes an image and its corresponding segmentation map, 
    and then extracts patches from the resized images.
    Attributes:
        resize (tuple): Target size to resize the image and segmentation map.
        patch_size (tuple): Size of the patches to be extracted.
        patch_stride (int): Stride for extracting patches.
        backend (str): Backend to use for resizing (default is 'cv2').
    Methods:
        __init__(resize, patch_size, patch_stride, backend):
            Initializes the ResizeAndPatchify class with the given parameters.
        transform(results):
        Resizes the input image and segmentation map, extracts patches, and updates the results dictionary.
    """
    def __init__(self, resize: tuple = (560, 448), patch_size: tuple = (224, 224), patch_stride: int = 112, backend='cv2'):
        super().__init__()
        self.resize = resize
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.backend = backend

        if resize:
            print(f"The number of extracted patches will be: {((resize[0] - patch_size[0]) // patch_stride + 1) * ((resize[1] - patch_size[1]) // patch_stride + 1)}")
        else:
            print("No patch extraction will be performed. The image will be used with its original size.")

    def transform(self, results: dict) -> dict:
        """
        Args:
            results (dict): The input data dictionary with 'img' and 'gt_seg_map' keys.
        
        Returns:
            dict: The modified data dictionary with patches added for 'img' and 'gt_seg_map'.
        
        Note: The input image and segmentation map should be numpy arrays.
        """

        img = results['img']
        gt_seg_map = results['gt_seg_map']  # Ground truth segmentation map
        h, w = img.shape[:2]
        
        if self.resize:
            # Determine target resize dimensions based on orientation
            target_size = self.resize if w > h else self.resize[::-1]
            # Resize the image and segmentation map
            resized_img, w_scale, h_scale = mmcv.imresize(img, 
                                                        target_size,
                                                        interpolation='bilinear',  # Bilinear for image
                                                        return_scale=True,
                                                        backend=self.backend)
            
            resized_seg_map = mmcv.imresize(gt_seg_map,
                                            target_size,
                                            interpolation='nearest',  # Nearest neighbor for segmentation map
                                            backend=self.backend)

            # Update the results dictionary with resized images
            results['img'] = resized_img
            results['img_shape'] = resized_img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['gt_seg_map'] = resized_seg_map

        

            # Calculate the number of patches in each dimension
            target_h, target_w = resized_img.shape[:2]
            num_patches_y = (target_h - self.patch_size[1]) // self.patch_stride + 1
            num_patches_x = (target_w - self.patch_size[0]) // self.patch_stride + 1

            # Define the patch shape and strides for both image and segmentation map
            img_shape = (num_patches_y, num_patches_x, self.patch_size[1], self.patch_size[0], resized_img.shape[2])
            seg_map_shape = (num_patches_y, num_patches_x, self.patch_size[1], self.patch_size[0])

            img_strides = (
                resized_img.strides[0] * self.patch_stride,
                resized_img.strides[1] * self.patch_stride,
                resized_img.strides[0],
                resized_img.strides[1],
                resized_img.strides[2],
            )

            seg_map_strides = (
                resized_seg_map.strides[0] * self.patch_stride,
                resized_seg_map.strides[1] * self.patch_stride,
                resized_seg_map.strides[0],
                resized_seg_map.strides[1],
            )

            # Extract patches for image and segmentation map using strided views
            img_patches = np.lib.stride_tricks.as_strided(resized_img, shape=img_shape, strides=img_strides)
            img_patches = img_patches.reshape(-1, self.patch_size[1], self.patch_size[0], resized_img.shape[2])

            gt_seg_map_patches = np.lib.stride_tricks.as_strided(resized_seg_map, shape=seg_map_shape, strides=seg_map_strides)
            gt_seg_map_patches = gt_seg_map_patches.reshape(-1, self.patch_size[1], self.patch_size[0])

            # Update results with patches for both image and segmentation map
            results['patches'] = img_patches  # Shape (num_patches, 224, 224, channels)
            results['gt_seg_map_patches'] = gt_seg_map_patches  # Shape (num_patches, 224, 224)
            results['num_patches'] = img_patches.shape[0]
            results['patch_shape'] = img_patches.shape[1:3]
            results['patch_grid_shape'] = (num_patches_y, num_patches_x)

        else:
            results['scale_factor'] = (1.0, 1.0)
            results['patches'] = img[None, ...]
            results['gt_seg_map_patches'] = gt_seg_map[None, ...]
            results['num_patches'] = 1
            results['patch_shape'] = img.shape[:2]
            results['patch_grid_shape'] = (1, 1)

        return results
    


@TRANSFORMS.register_module()
class ToTensorAndNormalize(BaseTransform):
    """
    a method to convert the data (dict) to tensor

    Does the following:
    1. Convert to tensor and transpose it to (C, H, W)
    2. Convert the BGR to RGB
    3. Normalize the image with mean and std

    """

    def __init__(self, mean, std,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'patch_shape', 'scale_factor',
                            'patch_grid_shape')
                            ):
        
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:  
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            img = img.transpose(2, 0, 1)
            img = to_tensor(img).contiguous()
            # convert to RGB
            if img.shape[0] == 3:
                img = img[[2, 1, 0], ...]
            # normalize the image
            img = (img - self.mean) / self.std
            packed_results['img'] = img


        if 'patches' in results:
            patches = results['patches']
            patches = patches.transpose(0, 3, 1, 2)
            patches = to_tensor(patches).contiguous()
            # convert to RGB
            if patches.shape[1] == 3:
                patches = patches[:, [2, 1, 0], ...]

            # normalize the image
            patches = (patches - self.mean[None]) / self.std[None]
            packed_results['patches'] = patches

        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            if len(gt_seg_map.shape) == 2:
                gt_seg_map = to_tensor(gt_seg_map[None, ...].astype(np.int64)).contiguous()
                packed_results['gt_seg_map'] = gt_seg_map

            else:
                raise ValueError('Please pay attention your ground truth '
                                'segmentation map, usually the segmentation '
                                'map is 2D, but got '
                                f'{gt_seg_map.shape}')
        
        
        if 'gt_seg_map_patches' in results:
            gt_seg_map_patches = results['gt_seg_map_patches']
            gt_seg_map_patches = to_tensor(gt_seg_map_patches[:,None].astype(np.int64)).contiguous()
            packed_results['gt_seg_map_patches'] = gt_seg_map_patches
            

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        packed_results['meta'] = img_meta

        return packed_results



@TRANSFORMS.register_module()
class RemapLabels(BaseTransform):
    """Remap GT label IDs onto a coarser super-class index space.

    Input:  gt_seg_map (np.ndarray) with original class indices, e.g. [0, 18]
    Output: gt_seg_map with super-class indices, e.g. [0, num_super_class-1]

    Args:
        mapping: list[int] of length = num_original_classes, where
            mapping[i] is the super-class index that original class i belongs to.
            Special value 255 (or anything >= num_super_class) is left untouched
            so it can serve as ignore_index.
        ignore_index: pixels with this value are kept as-is (default 255).
    """

    def __init__(self, mapping, ignore_index=255):
        super().__init__()
        self.mapping = np.asarray(mapping, dtype=np.int64)
        self.ignore_index = int(ignore_index)

    def _remap(self, gt):
        # Build remapped map; preserve ignore_index pixels.
        remapped = np.full_like(gt, fill_value=self.ignore_index)
        valid_mask = (gt != self.ignore_index) & (gt < len(self.mapping))
        remapped[valid_mask] = self.mapping[gt[valid_mask]]
        return remapped

    def transform(self, results):
        # Remap both the (resized) full GT map and any per-patch GT maps,
        # so this transform works regardless of whether it runs before or
        # after ResizeAndPatchify.
        for key in ('gt_seg_map', 'gt_seg_map_patches'):
            if key in results and results[key] is not None:
                results[key] = self._remap(results[key])
        return results


