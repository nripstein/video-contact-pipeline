from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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


def _normalize_old_full(df: pd.DataFrame) -> pd.DataFrame:
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
                "hand_side": row["which"] if det_type == "hand" else None,
                "offset_x": None,
                "offset_y": None,
                "offset_mag": None,
                "blue_prop": None,
                "blue_glove_status": "NA",
                "is_filtered": False,
                "filtered_by": "",
                "filtered_reason": "",
                "other_detail": row.get("other_detail", "NA"),
            }
        )

    return pd.DataFrame(rows)


def _source_hand_map(old_full_df: pd.DataFrame) -> dict:
    hand_df = old_full_df[old_full_df["type"] == "hand"]
    source_map = {}
    for frame_id, group in hand_df.groupby("frame_id"):
        sides = set(group["which"].dropna().tolist())
        sides = {s for s in sides if s in {"Left", "Right"}}
        if not sides:
            source_map[frame_id] = "NA"
        elif sides == {"Left", "Right"}:
            source_map[frame_id] = "Both"
        else:
            source_map[frame_id] = next(iter(sides))
    return source_map


def test_blue_glove_parity():
    test_input = os.getenv("TEST_INPUT")
    if not test_input:
        import pytest
        pytest.skip("Set TEST_INPUT to run blue glove parity test.")

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
        "--no-crop",
        "--no-flip",
        "--condense-priority-strategy",
        "portable_first",
    ]
    subprocess.run(new_cmd, check=True, cwd=REPO_ROOT, env=env)

    old_full = pd.read_csv(OLD_OUT_DIR / "detections_full.csv")
    new_full = pd.read_csv(NEW_OUT_DIR / "detections_full.csv")

    expected = _normalize_old_full(old_full)

    expected_path = NEW_OUT_DIR / "detections_full_expected_blue.csv"
    actual_path = NEW_OUT_DIR / "detections_full_actual_blue.csv"
    expected.to_csv(expected_path, index=False)
    new_full.to_csv(actual_path, index=False)

    compare_cols = [
        "frame_id",
        "frame_number",
        "detection_type",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "hand_side",
        "contact_state",
        "contact_label",
        "blue_prop",
        "blue_glove_status",
    ]

    expected_trim = expected[compare_cols + ["other_detail"]].copy()
    actual_trim = new_full[compare_cols].copy()

    def _key(df: pd.DataFrame) -> pd.Series:
        def _val(v):
            return "" if pd.isna(v) else str(v)
        return (
            df["frame_id"].map(_val)
            + "|"
            + df["detection_type"].map(_val)
            + "|"
            + df["bbox_x1"].map(_val)
            + "|"
            + df["bbox_y1"].map(_val)
            + "|"
            + df["bbox_x2"].map(_val)
            + "|"
            + df["bbox_y2"].map(_val)
            + "|"
            + df["hand_side"].map(_val)
        )

    expected_trim["row_key"] = _key(expected_trim)
    actual_trim["row_key"] = _key(actual_trim)

    expected_sorted = expected_trim.sort_values(by=["row_key"], kind="mergesort").reset_index(drop=True)
    actual_sorted = actual_trim.sort_values(by=["row_key"], kind="mergesort").reset_index(drop=True)

    # Build expected rows for blue_discard keys and compare against new output.
    blue_expected = expected_sorted[expected_sorted["other_detail"] == "blue_discard"].copy()
    blue_expected = blue_expected[["row_key"]].copy()
    blue_expected["contact_state"] = 0
    blue_expected["contact_label"] = "No Contact"
    blue_expected["blue_glove_status"] = "experimenter"

    actual_blue = actual_sorted[actual_sorted["row_key"].isin(blue_expected["row_key"])][
        ["row_key", "contact_state", "contact_label", "blue_glove_status"]
    ].copy()

    expected_path = NEW_OUT_DIR / "detections_full_expected_blue.csv"
    actual_path = NEW_OUT_DIR / "detections_full_actual_blue.csv"
    blue_expected.to_csv(expected_path, index=False)
    actual_blue.to_csv(actual_path, index=False)

    compare_csvs(expected_path, actual_path, sort_keys=["row_key"])

    # Condensed parity: normalize old to new schema before comparison.
    old_condensed = pd.read_csv(OLD_OUT_DIR / "detections_condensed.csv")
    source_map = _source_hand_map(old_full)
    rows = []
    for idx, row in old_condensed.iterrows():
        frame_id = row["frame_id"]
        frame_number = _parse_frame_number(frame_id, idx)
        label = row.get("contact_label_pred", "No Contact")
        if label in ("Portable Object Contact", "Portable Object"):
            label = "Portable Object"
        elif label in ("Stationary Object Contact", "Stationary Object"):
            label = "Stationary Object"
        else:
            label = "No Contact"
        rows.append(
            {
                "frame_id": frame_id,
                "frame_number": frame_number,
                "contact_label": label,
                "source_hand": source_map.get(frame_id, "NA"),
            }
        )
    old_norm = pd.DataFrame(rows)
    old_norm_path = NEW_OUT_DIR / "detections_condensed_old_norm.csv"
    new_norm_path = NEW_OUT_DIR / "detections_condensed_new_norm.csv"
    old_norm.to_csv(old_norm_path, index=False)
    new_condensed = pd.read_csv(NEW_OUT_DIR / "detections_condensed.csv")
    new_condensed.to_csv(new_norm_path, index=False)
    compare_csvs(
        old_norm_path,
        new_norm_path,
        sort_keys=["frame_number", "frame_id", "contact_label", "source_hand"],
    )
