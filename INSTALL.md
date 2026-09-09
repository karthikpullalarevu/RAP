# RAP Installation Guide

Tested on Ubuntu 22.04, NVIDIA driver 550+, CUDA 12.4, 8x GPUs.

## 1. Install Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /datahdd/karthik/miniforge3
rm /tmp/miniforge.sh
/datahdd/karthik/miniforge3/bin/conda init zsh   # or bash
exec zsh  # reload shell
```

## 2. Create conda env + PyTorch

```bash
conda create -n rap python=3.11 -y
conda activate rap
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

## 3. Install RAP and dependencies

```bash
cd /datahdd/karthik/RAP
pip install --upgrade pip "setuptools<70" wheel

# nuplan-devkit
pip install "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"

# RAP itself
pip install -e .

# Core packages
pip install \
    "numpy<2.0" \
    mmdet==3.3.0 \
    pytorch-lightning==2.2.1 \
    tensorboard==2.13.0 \
    timm==1.0.24 \
    hydra-core==1.3.2 \
    opencv-python==4.9.0.80
```

## 4. Build mmcv from source (needs CUDA toolkit)

This is the slow step (~10 min). Must be compiled from source because prebuilt wheels don't support this PyTorch+CUDA combo.

```bash
pip install ninja
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install "mmcv==2.1.0" \
    --no-binary=mmcv \
    --force-reinstall \
    --no-cache-dir \
    --no-build-isolation
```

**Common issues:**
- `ModuleNotFoundError: pkg_resources` → your setuptools is too new. Run `pip install "setuptools<70"` first.
- `nvcc not found` → make sure `CUDA_HOME` points to your CUDA toolkit (check with `which nvcc`).

## 5. Verify

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import mmcv; print(f'mmcv: {mmcv.__version__}')
import mmdet; print(f'mmdet: {mmdet.__version__}')
import pytorch_lightning; print(f'pl: {pytorch_lightning.__version__}')
import timm; print(f'timm: {timm.__version__}')
import hydra; print(f'hydra: {hydra.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
print('All OK')
"
```

## 6. Environment variables

Add to your shell config or a `.env` file:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/maps"
export NAVSIM_EXP_ROOT="/datahdd/karthik/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export OPENSCENE_DATA_ROOT="/datahdd/karthik/dataset"
```

## Lockfile

If you want the exact same versions without going through the above:

```bash
conda create -n rap python=3.11 -y
conda activate rap
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.lock.txt   # exact pinned versions
```

**Note:** mmcv still needs to be built from source (step 4) even with the lockfile — pip can't install it from a wheel for this setup.

## Updating the lockfile

After adding/upgrading packages, update the lockfile:

```bash
pip freeze > requirements.lock.txt
```

Commit this file so the next person (or next machine) gets the same env.
