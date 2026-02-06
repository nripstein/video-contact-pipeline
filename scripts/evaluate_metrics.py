from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import metrics as _metrics
import visualization as _viz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate condensed predictions against GT.")
    parser.add_argument("--pred", required=True, help="Path to detections_condensed.csv")
    parser.add_argument("--gt", required=True, help="Path to GT CSV (frame_number/gt_binary or frame_id/label)")
    parser.add_argument(
        "--overlaps",
        default="0.1,0.25,0.5,0.75",
        help="Comma-separated overlap thresholds for F1 (e.g., 0.1,0.25,0.5,0.75)",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    pred_df = pd.read_csv(pred_path)
    pred_df = pred_df.sort_values(by=["frame_number"], kind="mergesort").reset_index(drop=True)
    frame_numbers = pred_df["frame_number"].tolist()

    pred_bin = _viz.pred_binary_from_condensed(pred_df)
    gt_bin = _viz.load_gt_binary_from_csv(str(gt_path), frame_numbers)

    if len(pred_bin) != len(gt_bin):
        raise ValueError(f"Length mismatch: pred={len(pred_bin)} gt={len(gt_bin)}")

    mof = _metrics.frame_accuracy(pred_bin, gt_bin)
    mof_pct = mof * 100.0
    edit = _metrics.edit_score(pred_bin, gt_bin, bg_class=(0,), norm=True)

    overlap_thresholds = [float(x) for x in args.overlaps.split(",") if x.strip()]
    f1_scores = {f"F1@{int(t * 100)}": _metrics.f_score(pred_bin, gt_bin, t, bg_class=(0,)) for t in overlap_thresholds}

    tp, fp, tn, fn = _metrics.confusion_counts(pred_bin, gt_bin)

    results = {
        "MoF": round(mof, 4),
        "MoF_pct": round(mof_pct, 2),
        "Edit": round(edit, 2),
        "F1": {k: round(v, 2) for k, v in f1_scores.items()},
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_frames": int(len(pred_bin)),
    }

    print(f"MoF: {results['MoF']:.2f} ({results['MoF_pct']:.2f}%)")
    print(f"Edit Score: {results['Edit']:.2f}")
    for k, v in results["F1"].items():
        print(f"{k}: {v:.2f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
