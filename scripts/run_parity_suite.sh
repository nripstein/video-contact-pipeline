#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TEST_INPUT:-}" ]]; then
  echo "TEST_INPUT is required."
  echo "Example: TEST_INPUT=/path/to/frames_or_video scripts/run_parity_suite.sh"
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_OUT="${REPO_ROOT}/tests/output/old"
NEW_OUT="${REPO_ROOT}/tests/output/new"

mkdir -p "${OLD_OUT}" "${NEW_OUT}"

python "${REPO_ROOT}/scripts/run_old_pipeline.py" --input "${TEST_INPUT}" --output-dir "${OLD_OUT}"
# Disable refactor-only filters for legacy parity comparison.
python "${REPO_ROOT}/run_pipeline.py" --input "${TEST_INPUT}" --output-dir "${NEW_OUT}" --inference-only --no-crop --no-flip --no-object-size-filter --no-small-object-filter --condense-priority-strategy portable_first

REPO_ROOT="${REPO_ROOT}" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from tests.compare_csvs import compare_csvs
from tests.test_inference_parity_full import _normalize_old, _normalize_new
from tests.test_condense_parity import _source_hand_map, _parse_frame_number, _map_contact_label

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
old_out = repo_root / "tests" / "output" / "old"
new_out = repo_root / "tests" / "output" / "new"

old_full = pd.read_csv(old_out / "detections_full.csv")
new_full = pd.read_csv(new_out / "detections_full.csv")

old_norm = _normalize_old(old_full)
new_norm = _normalize_new(new_full)

full_expected = new_out / "detections_full_expected_parity.csv"
full_actual = new_out / "detections_full_actual_parity.csv"
old_norm.to_csv(full_expected, index=False)
new_norm.to_csv(full_actual, index=False)

compare_csvs(
    full_expected,
    full_actual,
    sort_keys=["frame_number", "detection_type", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence"],
)

import subprocess
subprocess.run(
    [
        "python",
        str(repo_root / "run_pipeline.py"),
        "--input",
        str(os.environ["TEST_INPUT"]),
        "--output-dir",
        str(new_out),
        "--no-crop",
        "--no-flip",
        "--no-small-object-filter",
        "--condense-priority-strategy",
        "portable_first",
    ],
    check=True,
    cwd=repo_root,
)

old_condensed = pd.read_csv(old_out / "detections_condensed.csv")
new_condensed = pd.read_csv(new_out / "detections_condensed.csv")

source_map = _source_hand_map(old_full)
rows = []
for idx, row in old_condensed.iterrows():
    frame_id = row["frame_id"]
    frame_number = _parse_frame_number(frame_id, idx)
    contact_label = _map_contact_label(row.get("contact_label_pred", "No Contact"))
    source_hand = source_map.get(frame_id, "NA")
    rows.append(
        {
            "frame_id": frame_id,
            "frame_number": frame_number,
            "contact_label": contact_label,
            "source_hand": source_hand,
        }
    )

expected_condensed = pd.DataFrame(rows)
cond_expected = new_out / "detections_condensed_expected_parity.csv"
cond_actual = new_out / "detections_condensed_actual_parity.csv"
expected_condensed.to_csv(cond_expected, index=False)
new_condensed.to_csv(cond_actual, index=False)

compare_csvs(cond_expected, cond_actual, sort_keys=["frame_number", "frame_id", "contact_label", "source_hand"])

print("PASS: parity checks (full + condensed)")
PY
