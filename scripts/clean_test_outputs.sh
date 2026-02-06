#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="tests/output"
if [[ -d "${OUT_DIR}" ]]; then
  rm -rf "${OUT_DIR}"
  echo "Removed ${OUT_DIR}/"
else
  echo "No ${OUT_DIR}/ to remove."
fi
