#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "Setting up HW3 Q1 AutoDL workspace at ${PROJECT_ROOT}"
require_command git
require_command python3
require_command wget
require_command unzip
init_conda

TWODGS_DIR="${EXTERNAL_DIR}/2d-gaussian-splatting"
THREESTUDIO_DIR="${EXTERNAL_DIR}/threestudio"
MAGIC123_DIR="${EXTERNAL_DIR}/Magic123"

clone_or_update "https://github.com/hbb1/2d-gaussian-splatting.git" "${TWODGS_DIR}" "true"
clone_or_update "https://github.com/threestudio-project/threestudio.git" "${THREESTUDIO_DIR}" "false"
clone_or_update "https://github.com/guochengqian/Magic123.git" "${MAGIC123_DIR}" "false"

log "Preparing 2DGS dependencies in base environment"
conda activate base
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install open3d mediapy lpips scikit-image tqdm trimesh plyfile opencv-python
if [[ -d "${TWODGS_DIR}/submodules/diff-surfel-rasterization" ]]; then
  python -m pip install --no-build-isolation --force-reinstall "${TWODGS_DIR}/submodules/diff-surfel-rasterization"
fi
if [[ -d "${TWODGS_DIR}/submodules/simple-knn" ]]; then
  python -m pip install --no-build-isolation --force-reinstall "${TWODGS_DIR}/submodules/simple-knn"
fi
conda deactivate

log "Creating/updating threestudio environment"
if ! conda env list | awk '{print $1}' | grep -qx "hw3_threestudio"; then
  conda create -n hw3_threestudio python=3.10 -y
fi
conda activate hw3_threestudio
python -m pip install --upgrade pip setuptools wheel ninja cmake
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${THREESTUDIO_DIR}/requirements.txt" || true
conda deactivate

log "Creating/updating Magic123 environment"
if ! conda env list | awk '{print $1}' | grep -qx "hw3_magic123"; then
  conda create -n hw3_magic123 python=3.10 -y
fi
conda activate hw3_magic123
python -m pip install --upgrade pip setuptools wheel ninja cmake
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${MAGIC123_DIR}/requirements.txt" || true
python -m pip install 'transformers==4.31.0' 'diffusers==0.19.3' 'huggingface_hub==0.16.4' 'accelerate==0.21.0' 'tokenizers<0.14,>=0.11.1' onnxruntime
conda deactivate

log "Setup complete. If Hugging Face or model downloads require auth, run huggingface-cli login before training."
