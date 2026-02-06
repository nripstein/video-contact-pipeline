from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def _map_contact_label(label: str) -> str:
    if label in ("Portable Object", "Portable Object Contact"):
        return "Portable Object"
    if label in ("Stationary Object", "Stationary Object Contact"):
        return "Stationary Object"
    if label == "No Contact":
        return "No Contact"
    return "No Contact"


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


def test_condense_parity():
    test_input = os.getenv("TEST_INPUT")
    if not test_input:
        import pytest
        pytest.skip("Set TEST_INPUT to run condense parity test.")

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
    ]
    subprocess.run(new_cmd, check=True, cwd=REPO_ROOT, env=env)

    old_full = pd.read_csv(OLD_OUT_DIR / "detections_full.csv")
    old_condensed = pd.read_csv(OLD_OUT_DIR / "detections_condensed.csv")
    new_condensed = pd.read_csv(NEW_OUT_DIR / "detections_condensed.csv")

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

    expected = pd.DataFrame(rows)
    expected_path = NEW_OUT_DIR / "detections_condensed_expected.csv"
    expected.to_csv(expected_path, index=False)

    actual_path = NEW_OUT_DIR / "detections_condensed_actual.csv"
    new_condensed.to_csv(actual_path, index=False)

    compare_csvs(
        expected_path,
        actual_path,
        sort_keys=["frame_number", "frame_id", "contact_label", "source_hand"],
    )
