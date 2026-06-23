#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

init_conda
conda activate hw3_threestudio
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

THREESTUDIO_DIR="${EXTERNAL_DIR}/threestudio"
OUT_ROOT="${RESULTS_DIR}/threestudio"
OUT_TAG="teapot_sds"

[[ -d "${THREESTUDIO_DIR}" ]] || fail "threestudio repo missing. Run scripts/setup_autodl.sh first."
mkdir -p "${OUT_ROOT}"

log "Training threestudio DreamFusion/SDS object: ${THREESTUDIO_PROMPT}"
(
  cd "${THREESTUDIO_DIR}"
  python launch.py \
    --config "${THREESTUDIO_CONFIG}" \
    --train \
    --gpu 0 \
    exp_root_dir="${OUT_ROOT}" \
    name="dreamfusion-sd15" \
    tag="${OUT_TAG}" \
    seed="${SEED}" \
    trainer.max_steps="${THREESTUDIO_STEPS}" \
    data.batch_size=1 \
    data.width=64 \
    data.height=64 \
    system.background.random_aug=true \
    system.prompt_processor.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    system.guidance.pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    system.prompt_processor.prompt="${THREESTUDIO_PROMPT}"
) 2>&1 | tee "${LOG_DIR}/threestudio_teapot_train.log"

log "threestudio run complete. Artifact candidates:"
find "${OUT_ROOT}" -type f \( -name "*.mp4" -o -name "*.obj" -o -name "*.ply" -o -name "*.ckpt" \) | tee "${LOG_DIR}/threestudio_teapot_artifacts.log"
