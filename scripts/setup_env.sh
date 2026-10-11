#!/usr/bin/env bash
# Stage 0 environment setup. Run AFTER /mnt is an ext4 volume owned by ubuntu.
#   bash scripts/setup_env.sh
# Idempotent: safe to re-run if interrupted.
set -euo pipefail

BASE=/mnt
CONDA_DIR=$BASE/miniforge3
ENV=gsplat
REPO=$(cd "$(dirname "$0")/.." && pwd)

# Keep every cache off the small root disk.
mkdir -p $BASE/cache/{pip,conda-pkgs,torch} $BASE/data $BASE/outputs
export PIP_CACHE_DIR=$BASE/cache/pip
export CONDA_PKGS_DIRS=$BASE/cache/conda-pkgs

# 1. Miniforge
if [ ! -x $CONDA_DIR/bin/conda ]; then
  curl -L -o $BASE/cache/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash $BASE/cache/miniforge.sh -b -p $CONDA_DIR
  $CONDA_DIR/bin/conda init bash
fi
source $CONDA_DIR/etc/profile.d/conda.sh

# 2. Env with Python 3.12, nvcc 12.6 and a gcc that nvcc 12.6 accepts (<=13)
if ! conda env list | grep -q "^$ENV "; then
  conda env create -f $REPO/environment.yml
fi
conda activate $ENV

# Env vars applied on every `conda activate gsplat`
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/gsplat_env.sh <<EOF
export TORCH_CUDA_ARCH_LIST="7.5"
export CUDA_HOME=\$CONDA_PREFIX
export CC=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export PIP_CACHE_DIR=$BASE/cache/pip
export TORCH_HOME=$BASE/cache/torch
export TORCH_EXTENSIONS_DIR=$BASE/cache/torch_extensions
EOF
conda deactivate; conda activate $ENV

# 3. PyTorch (CUDA 12.6 wheels)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

# 4. Python packages
pip install -r $REPO/requirements.txt

# 5. gsplat compiled for sm_75 against the installed torch
pip install --no-build-isolation gsplat==1.5.3

# 6. data/ and outputs/ live on the big disk
[ -e $REPO/data ] || ln -s $BASE/data $REPO/data
[ -e $REPO/outputs ] || ln -s $BASE/outputs $REPO/outputs

python $REPO/scripts/check_env.py
