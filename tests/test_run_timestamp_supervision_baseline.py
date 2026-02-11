from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_timestamp_supervision_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_timestamp_supervision_baseline", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prediction_outputs(pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    condensed = pd.DataFrame(
        [
            {"frame_id": "000000.png", "frame_number": 0, "contact_label": "No Contact", "source_hand": "Left"},
            {"frame_id": "000001.png", "frame_number": 1, "contact_label": "No Contact", "source_hand": "Left"},
            {"frame_id": "000002.png", "frame_number": 2, "contact_label": "Portable Object", "source_hand": "Right"},
            {"frame_id": "000003.png", "frame_number": 3, "contact_label": "Portable Object", "source_hand": "Right"},
        ]
    )
    full = pd.DataFrame(
        [
            {"frame_id": "000000.png", "frame_number": 0, "detection_type": "hand", "blue_glove_status": "participant"},
            {"frame_id": "000001.png", "frame_number": 1, "detection_type": "hand", "blue_glove_status": "participant"},
            {"frame_id": "000002.png", "frame_number": 2, "detection_type": "hand", "blue_glove_status": "participant"},
            {"frame_id": "000003.png", "frame_number": 3, "detection_type": "hand", "blue_glove_status": "participant"},
        ]
    )
    condensed.to_csv(pred_dir / "detections_condensed.csv", index=False)
    full.to_csv(pred_dir / "detections_full.csv", index=False)


def test_main_end_to_end_success(tmp_path: Path):
    mod = _load_module()
    run_root = tmp_path / "run"
    pred_dir = run_root / "predictions" / "sv1"
    _write_prediction_outputs(pred_dir)

    gt_csv = tmp_path / "sv1_gt.csv"
    pd.DataFrame(
        [
            {"frame_number": 0, "gt_binary": 0},
            {"frame_number": 1, "gt_binary": 0},
            {"frame_number": 2, "gt_binary": 1},
            {"frame_number": 3, "gt_binary": 1},
        ]
    ).to_csv(gt_csv, index=False)

    pd.DataFrame(
        [
            {"dataset_key": "sv1", "pred_dir": str(pred_dir), "gt_csv": str(gt_csv), "status": "success"},
        ]
    ).to_csv(run_root / "run_manifest.csv", index=False)

    import sys

    original_argv = sys.argv
    sys.argv = [
        "run_timestamp_supervision_baseline.py",
        "--run-root",
        str(run_root),
        "--fps",
        "1",
        "--min-island-seconds",
        "1",
        "--random-seed",
        "0",
    ]
    try:
        rc = mod.main()
    finally:
        sys.argv = original_argv

    assert rc == 0
    out_root = run_root / "timestamp_supervision_baseline"
    prep_manifest = out_root / "prep_manifest.csv"
    eval_manifest = out_root / "evaluation_manifest.csv"
    per_dataset = out_root / "evaluation" / "selected_timestamp_metrics_per_dataset.csv"
    summary_json = out_root / "evaluation" / "selected_timestamp_metrics_summary.json"
    run_state = out_root / "baseline_run_state.json"

    assert prep_manifest.exists()
    assert eval_manifest.exists()
    assert per_dataset.exists()
    assert summary_json.exists()
    assert run_state.exists()

    per_df = pd.read_csv(per_dataset)
    assert per_df["dataset_key"].tolist() == ["sv1"]
    assert per_df["status"].tolist() == ["success"]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["n_datasets_success"] == 1
    assert summary["n_datasets_failed"] == 0
    assert summary["global_micro"] is not None
    assert summary["global_macro"] is not None


def test_main_returns_nonzero_when_any_prepare_fails(tmp_path: Path):
    mod = _load_module()
    run_root = tmp_path / "run"
    pred_dir = run_root / "predictions" / "sv1"
    _write_prediction_outputs(pred_dir)

    gt_csv = tmp_path / "sv1_gt.csv"
    pd.DataFrame(
        [
            {"frame_number": 0, "gt_binary": 0},
            {"frame_number": 1, "gt_binary": 0},
            {"frame_number": 2, "gt_binary": 1},
            {"frame_number": 3, "gt_binary": 1},
        ]
    ).to_csv(gt_csv, index=False)

    pd.DataFrame(
        [
            {"dataset_key": "sv1", "pred_dir": str(pred_dir), "gt_csv": str(gt_csv), "status": "success"},
            {
                "dataset_key": "sv2",
                "pred_dir": str(run_root / "predictions" / "sv2_missing"),
                "gt_csv": str(gt_csv),
                "status": "success",
            },
        ]
    ).to_csv(run_root / "run_manifest.csv", index=False)

    import sys

    original_argv = sys.argv
    sys.argv = [
        "run_timestamp_supervision_baseline.py",
        "--run-root",
        str(run_root),
        "--fps",
        "1",
        "--min-island-seconds",
        "1",
        "--random-seed",
        "0",
    ]
    try:
        rc = mod.main()
    finally:
        sys.argv = original_argv

    assert rc == 1
    prep_manifest = run_root / "timestamp_supervision_baseline" / "prep_manifest.csv"
    assert prep_manifest.exists()
    prep_df = pd.read_csv(prep_manifest)
    status_map = {str(row["dataset_key"]): str(row["status"]) for _, row in prep_df.iterrows()}
    assert status_map["sv1"] == "prepared"
    assert status_map["sv2"] == "failed"
