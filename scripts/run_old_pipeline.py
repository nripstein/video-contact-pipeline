"""
Wrapper around the current (pre-refactor) detection pipeline.

Behavior:
- Calls `my_demo_fn.main` to run the existing detector on a video file or a
  directory of frames.
- Saves the raw dataframe to `detections_full.csv`.
- Applies the historical "condense_dataframe" logic from the notebooks to
  collapse duplicate frames and writes `detections_condensed.csv`.

Note: This intentionally mirrors the legacy workflow; no refactoring of the
original code is performed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPO_ROOT / "archive"
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import my_demo_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run legacy 100-DOH pipeline")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to video file or directory of frames to process.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write detections_full.csv and detections_condensed.csv",
    )
    parser.add_argument(
        "--no-blue-refine",
        action="store_true",
        help="Disable blue-glove refinement (legacy flag stays default on).",
    )
    return parser.parse_args()


def condense_dataframe(df: pd.DataFrame, priority: Dict[str, int] | None = None) -> pd.DataFrame:
    """
    Legacy logic copied from the notebooks:
    - For duplicate frame_ids, keep the highest-priority contact_label_pred.
    - Priority order (lowest number wins) matches notebook defaults.
    - Final dataframe is sorted by the numeric prefix of frame_id.
    """
    if priority is None:
        priority = {
            "Portable Object": 1,
            "Portable Object Contact": 1,
            "Stationary Object Contact": 2,
            "No Contact": 3,
        }

    if df.empty and "contact_label_pred" not in df.columns:
        return pd.DataFrame(columns=["frame_id", "contact_label_pred"])
    if "contact_label_pred" not in df.columns:
        raise ValueError("Expected column 'contact_label_pred' in detector output.")
    if "frame_id" not in df.columns:
        raise ValueError("Expected column 'frame_id' in detector output.")

    df = df.copy()
    df["priority"] = df["contact_label_pred"].map(priority)
    df = df.sort_values(by=["frame_id", "priority"])
    df = df.drop(columns=["priority"])
    df = df.drop_duplicates(subset="frame_id", keep="first")

    def _frame_sort_key(val: str) -> int:
        try:
            return int(str(val).split("_")[0])
        except Exception:
            return 0

    df = df.iloc[df["frame_id"].map(_frame_sort_key).argsort()].reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    blue_refine = not args.no_blue_refine

    # Run the existing detector; returns a pandas DataFrame.
    detections = my_demo_fn.main(save_imgs=False, img_dir=str(input_path), blue_refine=blue_refine)
    full_path = output_dir / "detections_full.csv"
    detections.to_csv(full_path, index=False)

    condensed = condense_dataframe(detections)
    condensed_path = output_dir / "detections_condensed.csv"
    condensed.to_csv(condensed_path, index=False)

    print(f"Wrote {full_path}")
    print(f"Wrote {condensed_path}")


if __name__ == "__main__":
    main()
