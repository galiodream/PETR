#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-environment.cpu.yml}
CONDA_ROOT=${CONDA_ROOT:-/tmp/miniconda3}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
elif [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="${CONDA_ROOT}"
else
  INSTALLER=/tmp/miniconda-installer.sh
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${INSTALLER}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  else
    echo "Neither curl nor wget is available, cannot download Miniconda" >&2
    exit 1
  fi
  bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
  CONDA_BASE="${CONDA_ROOT}"
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda tos --help >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
fi

ENV_NAME=$(awk '/^name:/{print $2; exit}' "${ENV_FILE}")
if [[ -z "${ENV_NAME}" ]]; then
  echo "Cannot parse env name from ${ENV_FILE}" >&2
  exit 1
fi

conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda env create -f "${ENV_FILE}"
conda activate "${ENV_NAME}"
python -V
python -c "import torch, yaml, numpy; print('deps_ok', torch.__version__)"
