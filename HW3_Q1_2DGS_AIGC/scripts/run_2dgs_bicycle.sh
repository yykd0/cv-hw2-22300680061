#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

init_conda
conda activate base
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

TWODGS_DIR="${EXTERNAL_DIR}/2d-gaussian-splatting"
SCENE_DIR="${DATA_DIR}/mipnerf360/${MIPNERF360_SCENE}"
OUT_DIR="${RESULTS_DIR}/2dgs/${MIPNERF360_SCENE}"

[[ -d "${TWODGS_DIR}" ]] || fail "2DGS repo missing. Run scripts/setup_autodl.sh first."
if [[ ! -d "${SCENE_DIR}" ]]; then
  bash "${PROJECT_ROOT}/scripts/download_mipnerf360.sh"
fi

mkdir -p "${OUT_DIR}"
log "Training 2DGS on ${SCENE_DIR}"
(
  cd "${TWODGS_DIR}"
  python train.py \
    -s "${SCENE_DIR}" \
    -i "${TWODGS_IMAGE_DIR}" \
    -m "${OUT_DIR}" \
    --iterations "${TWODGS_ITERATIONS}" \
    --eval
) 2>&1 | tee "${LOG_DIR}/2dgs_train_${MIPNERF360_SCENE}.log"

log "Rendering and extracting 2DGS mesh"
(
  cd "${TWODGS_DIR}"
  python render.py \
    -s "${SCENE_DIR}" \
    -i "${TWODGS_IMAGE_DIR}" \
    -m "${OUT_DIR}" \
    --unbounded \
    --skip_test \
    --mesh_res "${TWODGS_MESH_RES}"
) 2>&1 | tee "${LOG_DIR}/2dgs_render_${MIPNERF360_SCENE}.log"

log "2DGS run complete: ${OUT_DIR}"
