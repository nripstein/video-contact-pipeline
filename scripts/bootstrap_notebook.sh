#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-shan_et_al2}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH."
  exit 1
fi

if [[ ! -x /usr/bin/gcc-10 || ! -x /usr/bin/g++-10 ]]; then
  echo "ERROR: /usr/bin/gcc-10 and /usr/bin/g++-10 are required to build extensions."
  echo "Install with: sudo apt install gcc-10 g++-10"
  exit 1
fi

cd "${REPO_ROOT}"

echo "[1/5] Creating/updating conda env: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f environment.yml
else
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

echo "[2/5] Installing python dependencies"
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt
conda run -n "${ENV_NAME}" python -m pip install -r requirements-notebooks.txt
conda run -n "${ENV_NAME}" python -m pip install -e .

echo "[3/5] Building C++/CUDA extensions"
conda run -n "${ENV_NAME}" bash -lc "cd '${REPO_ROOT}/lib' && CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop"

echo "[4/5] Registering Jupyter kernel"
conda run -n "${ENV_NAME}" python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo "[5/5] Verifying import"
conda run -n "${ENV_NAME}" python -c "from pipeline.config import PipelineConfig; print('pipeline import ok')"

echo
echo "Setup complete."
echo "Use this kernel in Jupyter: Python (${ENV_NAME})"
echo "Launch from repo root:"
echo "  cd '${REPO_ROOT}'"
echo "  conda activate '${ENV_NAME}'"
echo "  jupyter lab"
