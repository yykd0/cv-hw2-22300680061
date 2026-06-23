#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

init_conda
conda activate hw3_magic123
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MAGIC123_DIR="${EXTERNAL_DIR}/Magic123"
INPUT_DIR="${PROJECT_ROOT}/assets/magic123_input"
MAIN_IMAGE="${INPUT_DIR}/main.png"
RGBA_IMAGE="${INPUT_DIR}/rgba.png"
OUT_ROOT="${RESULTS_DIR}/magic123/dragon_nodepth"
COARSE_DIR="${OUT_ROOT}/coarse"
FINE_DIR="${OUT_ROOT}/fine"

[[ -d "${MAGIC123_DIR}" ]] || fail "Magic123 repo missing. Run scripts/setup_autodl.sh first."
[[ -f "${RGBA_IMAGE}" ]] || fail "Missing ${RGBA_IMAGE}. Generate/prepare the AIGC image first."

mkdir -p "${OUT_ROOT}" "${MAGIC123_DIR}/pretrained/zero123"

if [[ ! -f "${MAGIC123_DIR}/pretrained/zero123/105000.ckpt" ]]; then
  log "Downloading Zero-1-to-3 checkpoint"
  wget -O "${MAGIC123_DIR}/pretrained/zero123/105000.ckpt" \
    "https://hf-mirror.com/cvlab/zero123-weights/resolve/main/105000.ckpt"
fi

log "Skipping MiDaS preprocessing for the verified no-depth Magic123 route; using existing rgba.png."

log "Running Magic123 coarse stage"
(
  cd "${MAGIC123_DIR}"
  CUDA_VISIBLE_DEVICES=0 python main.py -O \
    --text "${MAGIC123_PROMPT}" \
    --sd_version 1.5 \
    --image "${RGBA_IMAGE}" \
    --workspace "${COARSE_DIR}" \
    --optim adam \
    --iters "${MAGIC123_COARSE_ITERS}" \
    --guidance SD zero123 \
    --lambda_guidance ${MAGIC123_LAMBDA_GUIDANCE_COARSE} \
    --guidance_scale 100 5 \
    --latent_iter_ratio 0 \
    --normal_iter_ratio 0.2 \
    --t_range 0.2 0.6 \
    --bg_radius -1 \
    --lambda_depth 0 \
    --dataset_size_test 32 \
    --save_mesh
) 2>&1 | tee "${LOG_DIR}/magic123_dragon_coarse.log"

COARSE_CKPT="$(latest_file "${COARSE_DIR}" "*.pth")"
[[ -f "${COARSE_CKPT}" ]] || fail "Could not find Magic123 coarse checkpoint in ${COARSE_DIR}"
log "Stable HW3 route stops after coarse Magic123. Fine DMTet is optional and was not used for the final result."
log "Magic123 run complete. Artifact candidates:"
find "${OUT_ROOT}" -type f \( -name "*.mp4" -o -name "*.obj" -o -name "*.ply" -o -name "*.pth" \) | tee "${LOG_DIR}/magic123_dragon_artifacts.log"
