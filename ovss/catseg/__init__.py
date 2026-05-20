"""CAT-Seg backbone integration for MLMP.

Provides CATSegWrapper, exposed via load_catseg() with the same
interface contract as ovss/__init__.py::load_ovss.

Source: github.com/cvlab-kaist/CAT-Seg (MIT licensed). See README.md
in this directory for commit hash, checkpoint download steps, and any
skipped state-dict keys.
"""
from ovss.catseg.catseg_wrapper import CATSegWrapper
from ovss.clip import tokenize as clip_tokenize


def load_catseg(backbone='ViT-L/14', device='cpu'):
    """Construct a CATSegWrapper and return (wrapper, tokenizer).

    Args:
        backbone: ViT identifier accepted by ovss.clip.load (e.g. 'ViT-L/14').
        device: 'cpu' or a CUDA device string.

    Returns:
        (CATSegWrapper, tokenize_fn)
    """
    wrapper = CATSegWrapper(backbone=backbone, device=device)
    return wrapper, clip_tokenize
