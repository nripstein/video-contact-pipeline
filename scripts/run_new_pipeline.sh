#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   scripts/run_new_pipeline.sh /path/to/video.mp4 /path/to/output
#   scripts/run_new_pipeline.sh /path/to/frames_dir /path/to/output
#   scripts/run_new_pipeline.sh /path/to/folder_of_videos /path/to/output
#
# You can also set INPUT and OUTPUT_DIR env vars:
#   INPUT=/path/to/input OUTPUT_DIR=/path/to/output scripts/run_new_pipeline.sh

INPUT="${1:-${INPUT:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"

if [[ -z "${INPUT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 <input> <output_dir>"
  echo "Or set INPUT and OUTPUT_DIR env vars."
  exit 2
fi

python run_pipeline.py --input "${INPUT}" --output-dir "${OUTPUT_DIR}"
