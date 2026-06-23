#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command wget
require_command unzip

RAW_DIR="${DATA_DIR}/raw"
MIP_DIR="${DATA_DIR}/mipnerf360"
ARCHIVE="${RAW_DIR}/360_v2.zip"
SCENE_DIR="${MIP_DIR}/${MIPNERF360_SCENE}"

mkdir -p "${RAW_DIR}" "${MIP_DIR}"

if [[ ! -d "${SCENE_DIR}" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    log "Downloading Mip-NeRF 360 dataset archive"
    wget -O "${ARCHIVE}" "${MIPNERF360_URL}"
  else
    log "Using existing archive ${ARCHIVE}"
  fi

  log "Extracting ${MIPNERF360_SCENE}"
  unzip -q "${ARCHIVE}" -d "${MIP_DIR}"
fi

[[ -d "${SCENE_DIR}" ]] || fail "Scene directory not found after extraction: ${SCENE_DIR}"
log "Mip-NeRF 360 scene ready: ${SCENE_DIR}"
