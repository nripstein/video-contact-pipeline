from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from compare_csvs import compare_csvs


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_OUT_DIR = REPO_ROOT / "tests" / "output" / "old"
NEW_OUT_DIR = REPO_ROOT / "tests" / "output" / "new"


def _parse_frame_number(name: str, fallback: int) -> int:
    stem = Path(name).stem
    num = None
    current = ""
    for ch in stem:
        if ch.isdigit():
            current += ch
        elif current:
            num = int(current)
            break
    if current and num is None:
        num = int(current)
    return num if num is not None else fallback


def _normalize_old(df: pd.DataFrame) -> pd.DataFrame:
    label_to_state = {
        "No Contact": 0,
        "Self Contact": 1,
        "Other Person Contact": 2,
        "Portable Object": 3,
        "Portable Object Contact": 3,
        "Stationary Object Contact": 4,
    }

    rows = []
    for idx, row in df.iterrows():
        frame_id = row["frame_id"]
        frame_number = _parse_frame_number(frame_id, idx)
        det_type = "hand" if row["type"] == "hand" else "object"

        bbox = ast.literal_eval(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        contact_label = row["contact_label_pred"] if det_type == "hand" else None
        contact_state = label_to_state.get(contact_label) if det_type == "hand" else None
        hand_side = row["which"] if det_type == "hand" else None

        rows.append(
            {
                "frame_id": frame_id,
                "frame_number": frame_number,
                "detection_type": det_type,
                "bbox_x1": int(bbox_x1),
                "bbox_y1": int(bbox_y1),
                "bbox_x2": int(bbox_x2),
                "bbox_y2": int(bbox_y2),
                "confidence": float(row["probability"]),
                "contact_state": contact_state,
                "contact_label": contact_label,
                "hand_side": hand_side,
                "offset_x": None,
                "offset_y": None,
                "offset_mag": None,
                "blue_prop": None,
                "blue_glove_status": None,
                "is_filtered": False,
                "filtered_by": "",
                "filtered_reason": "",
            }
        )

    return pd.DataFrame(rows)


def _normalize_new(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "frame_id",
        "frame_number",
        "detection_type",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "confidence",
        "contact_state",
        "contact_label",
        "hand_side",
        "offset_x",
        "offset_y",
        "offset_mag",
        "blue_prop",
        "blue_glove_status",
        "is_filtered",
        "filtered_by",
        "filtered_reason",
    ]
    norm = df[columns].copy()
    # Old pipeline did not expose offsets/blue proportions; normalize to match.
    norm["offset_x"] = None
    norm["offset_y"] = None
    norm["offset_mag"] = None
    norm["blue_prop"] = None
    norm["blue_glove_status"] = None
    # Old pipeline rewrites confidence to 100 for blue-discarded hands.
    if "blue_glove_status" in df.columns:
        mask = df["blue_glove_status"] == "experimenter"
        norm.loc[mask, "confidence"] = 100.0
    return norm


def test_inference_parity_full():
    test_input = os.getenv("TEST_INPUT")
    if not test_input:
        import pytest
        pytest.skip("Set TEST_INPUT to run inference parity test.")

    OLD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    NEW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    old_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_old_pipeline.py"),
        "--input",
        test_input,
        "--output-dir",
        str(OLD_OUT_DIR),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(old_cmd, check=True, cwd=REPO_ROOT, env=env)

    new_cmd = [
        sys.executable,
        str(REPO_ROOT / "run_pipeline.py"),
        "--input",
        test_input,
        "--output-dir",
        str(NEW_OUT_DIR),
        "--inference-only",
        "--no-crop",
        "--no-flip",
    ]
    subprocess.run(new_cmd, check=True, cwd=REPO_ROOT, env=env)

    old_df = pd.read_csv(OLD_OUT_DIR / "detections_full.csv")
    new_df = pd.read_csv(NEW_OUT_DIR / "detections_full.csv")

    old_norm = _normalize_old(old_df)
    new_norm = _normalize_new(new_df)

    tmp_old = NEW_OUT_DIR / "detections_full_old_norm.csv"
    tmp_new = NEW_OUT_DIR / "detections_full_new_norm.csv"
    old_norm.to_csv(tmp_old, index=False)
    new_norm.to_csv(tmp_new, index=False)

    sort_keys = [
        "frame_number",
        "detection_type",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "confidence",
    ]
    compare_csvs(tmp_old, tmp_new, sort_keys=sort_keys)
