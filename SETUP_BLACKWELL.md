# Environment Setup for Blackwell GPUs (sm_120)

This guide is for machines with NVIDIA Blackwell GPUs (RTX PRO 6000, RTX 50xx series, etc.) which require CUDA 12.8+ and are **not** compatible with the default cu118 setup in the README.

**Tested on:** Ubuntu 24.04, NVIDIA RTX PRO 6000 Blackwell (sm_120), Driver 580.x, System CUDA 13.0

---

## Step 1: Create Conda Environment

```bash
conda create -n mlmp python=3.10 -y
conda activate mlmp
```

---

## Step 2: Install PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

---

## Step 3: Install NVCC 12.8

The system `/usr/bin/nvcc` may be an older version that does not support `sm_120`. Install the correct version into the conda environment:

```bash
conda install -y cuda-nvcc=12.8 -c nvidia/label/cuda-12.8.0
```

---

## Step 4: Downgrade setuptools

setuptools 70+ removed `pkg_resources.packaging`, which is required by mmcv and the bundled CLIP code:

```bash
pip install "setuptools<70"
```

---

## Step 5: Install libjpeg (required by Pillow_SIMD)

```bash
conda install -y libjpeg-turbo -c conda-forge
```

---

## Step 6: Install Remaining Requirements

```bash
pip install -r requirements.txt
```

---

## Step 7: Compile mmcv from Source

OpenMMLab does not yet provide pre-built mmcv wheels for cu128. It must be compiled from source.

> Ubuntu 24.04 ships g++-13 by default, but CUDA 12.8 requires g++ < 13. Use g++-12 explicitly.

```bash
CXX=g++-12 CC=gcc-12 pip install mmcv==2.1.0 --no-binary mmcv --no-build-isolation
```

This step takes approximately 5–10 minutes.

---

## Step 8: Fix NumPy Version

pandas and PyTorch (compiled against NumPy 1.x) are incompatible with NumPy 2.x:

```bash
pip install "numpy<2"
```

---

## Verify the Full Installation

```bash
python -c "
import torch, mmcv
from mmcv.ops import point_sample
print('torch:', torch.__version__)
print('mmcv:', mmcv.__version__)
print('GPU:', torch.cuda.get_device_name(0))
"
```

Expected output:
```
torch: 2.11.0+cu128
mmcv: 2.1.0
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

---

## Known Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: no kernel image is available` | PyTorch cu118 does not support sm_120 | Use cu128 (Step 2) |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools 70+ removed it | Downgrade setuptools (Step 4) |
| `ImportError: cannot import name 'packaging' from 'pkg_resources'` | Same as above | Downgrade setuptools (Step 4) |
| `ImportError: libcudart.so.11.0: cannot open shared object file` | mmcv wheel compiled for cu118, mismatches torch cu128 | Recompile from source (Step 7) |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | System nvcc too old | Install conda nvcc 12.8 (Step 3) |
| `RuntimeError: g++ version > maximum required by CUDA` | g++-13 incompatible with CUDA 12.8 | Use `CXX=g++-12` (Step 7) |
| NumPy 2.x warnings / crashes | torch compiled against NumPy 1.x | Downgrade numpy (Step 8) |
