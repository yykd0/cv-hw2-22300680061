#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/experiments.env"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

EXTERNAL_DIR="${PROJECT_ROOT}/external"
DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/logs"
FIGURES_DIR="${RESULTS_DIR}/figures"
SUBMISSION_DIR="${PROJECT_ROOT}/submission"

mkdir -p "${EXTERNAL_DIR}" "${DATA_DIR}" "${RESULTS_DIR}" "${LOG_DIR}" "${FIGURES_DIR}" "${SUBMISSION_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"
}

init_conda() {
  local conda_base
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
  else
    for candidate in /root/miniconda3 /root/miniconda /opt/conda; do
      if [[ -x "${candidate}/bin/conda" ]]; then
        conda_base="${candidate}"
        break
      fi
    done
  fi
  [[ -n "${conda_base:-}" ]] || fail "Missing conda. Expected it in PATH, /root/miniconda3, /root/miniconda, or /opt/conda."
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
}

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  local recursive="${3:-false}"

  if [[ -d "${target_dir}/.git" ]]; then
    log "Repository already exists: ${target_dir}"
    git -C "${target_dir}" fetch --all --tags
  else
    log "Cloning ${repo_url}"
    if [[ "${recursive}" == "true" ]]; then
      git clone --recursive "${repo_url}" "${target_dir}"
    else
      git clone "${repo_url}" "${target_dir}"
    fi
  fi
}

tee_log() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

latest_file() {
  local root="$1"
  local pattern="$2"
  find "${root}" -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}
