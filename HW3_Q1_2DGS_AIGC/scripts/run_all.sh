#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "Starting HW3 Q1 full pipeline"
bash "${PROJECT_ROOT}/scripts/download_mipnerf360.sh"
bash "${PROJECT_ROOT}/scripts/run_2dgs_bicycle.sh"
bash "${PROJECT_ROOT}/scripts/run_threestudio_teapot.sh"
bash "${PROJECT_ROOT}/scripts/run_magic123_dragon.sh"

if command -v blender >/dev/null 2>&1; then
  log "Blender found. Rendering unified scene."
  blender -b --python "${PROJECT_ROOT}/scripts/render_scene.py" -- \
    --project-root "${PROJECT_ROOT}" \
    --output-dir "${RESULTS_DIR}/blender"
else
  log "Blender not found. Skipping unified scene render; keep native videos as evidence."
fi

init_conda
conda activate base
python "${PROJECT_ROOT}/scripts/collect_results.py"
python "${PROJECT_ROOT}/scripts/make_report.py"
python "${PROJECT_ROOT}/scripts/package_submission.py"

log "Full pipeline complete"
