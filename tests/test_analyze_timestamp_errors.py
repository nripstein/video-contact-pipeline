from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_timestamp_errors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_timestamp_errors", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_writes_analysis_outputs(tmp_path: Path):
    mod = _load_module()

    selected_ok = tmp_path / "selected_ok.csv"
    gt_ok = tmp_path / "gt_ok.csv"
    metrics_csv = tmp_path / "metrics.csv"
    out_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {"frame_id": 10, "predicted_label": 1},
            {"frame_id": 11, "predicted_label": 1},
            {"frame_id": 12, "predicted_label": 0},
            {"frame_id": 99, "predicted_label": 1},  # missing in GT on purpose
        ]
    ).to_csv(selected_ok, index=False)
    pd.DataFrame(
        [
            {"frame_number": 10, "gt_binary": 1},
            {"frame_number": 11, "gt_binary": 0},  # mismatch => FP
            {"frame_number": 12, "gt_binary": 0},
        ]
    ).to_csv(gt_ok, index=False)

    pd.DataFrame(
        [
            {
                "dataset_key": "ok_ds",
                "status": "success",
                "error": "",
                "n_selected_total": 4,
                "n_selected_with_gt": 3,
                "n_selected_missing_gt": 1,
                "n_gt_total": 3,
                "coverage_selected_over_gt": 1.0,
                "coverage_aligned_over_selected": 0.75,
                "accuracy": 2.0 / 3.0,
                "tp": 1,
                "fp": 1,
                "tn": 1,
                "fn": 0,
                "selected_csv": str(selected_ok),
                "gt_csv": str(gt_ok),
            },
            {
                "dataset_key": "failed_ds",
                "status": "failed",
                "error": "No aligned selected frames found in GT for evaluation.",
                "selected_csv": str(tmp_path / "missing_selected.csv"),
                "gt_csv": str(tmp_path / "missing_gt.csv"),
            },
        ]
    ).to_csv(metrics_csv, index=False)

    args = argparse.Namespace(
        metrics_csv=str(metrics_csv),
        output_dir=str(out_dir),
        top_k=5,
    )
    rc = mod.run(args)
    assert rc == 0

    diag_path = out_dir / "dataset_diagnostic_table.csv"
    mismatch_path = out_dir / "frame_level_mismatches.csv"
    missing_gt_path = out_dir / "frame_level_missing_gt.csv"
    priority_path = out_dir / "priority_datasets.csv"
    density_path = out_dir / "selection_density_report.csv"
    summary_path = out_dir / "analysis_summary.json"

    assert diag_path.exists()
    assert mismatch_path.exists()
    assert missing_gt_path.exists()
    assert priority_path.exists()
    assert density_path.exists()
    assert summary_path.exists()

    diag_df = pd.read_csv(diag_path)
    assert set(diag_df["dataset_key"]) == {"ok_ds", "failed_ds"}
    ok_row = diag_df[diag_df["dataset_key"] == "ok_ds"].iloc[0]
    assert int(ok_row["mismatch_count"]) == 1
    assert int(ok_row["selected_from_csv_pred_pos"]) == 3
    assert int(ok_row["selected_from_csv_pred_neg"]) == 1

    mismatch_df = pd.read_csv(mismatch_path)
    assert len(mismatch_df) == 1
    assert mismatch_df.iloc[0]["dataset_key"] == "ok_ds"
    assert int(mismatch_df.iloc[0]["frame_id"]) == 11
    assert mismatch_df.iloc[0]["error_type"] == "fp"

    missing_gt_df = pd.read_csv(missing_gt_path)
    assert len(missing_gt_df) == 1
    assert missing_gt_df.iloc[0]["dataset_key"] == "ok_ds"
    assert int(missing_gt_df.iloc[0]["frame_id"]) == 99

    priority_df = pd.read_csv(priority_path)
    # Failed datasets should be ranked highest by priority score.
    assert priority_df.iloc[0]["dataset_key"] == "failed_ds"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_datasets_total"] == 2
    assert summary["n_datasets_success"] == 1
    assert summary["n_datasets_failed"] == 1
    assert summary["n_frame_mismatches"] == 1
    assert summary["n_selected_missing_gt_rows"] == 1

