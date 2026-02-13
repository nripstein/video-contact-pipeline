from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timestamp_supervision_extraction.evaluate_selected_timestamps import (
    evaluate_manifest,
    evaluate_single_dataset,
    run,
)


def _write_selected_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_gt_binary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_gt_label_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_evaluate_single_dataset_numeric_gt(tmp_path: Path):
    selected_csv = _write_selected_csv(
        tmp_path / "tmp_selected_numeric.csv",
        [
            {"frame_id": 1, "predicted_label": 1},
            {"frame_id": 2, "predicted_label": 1},
            {"frame_id": 3, "predicted_label": 0},
            {"frame_id": 4, "predicted_label": 0},
        ],
    )
    gt_csv = _write_gt_binary_csv(
        tmp_path / "tmp_gt_numeric.csv",
        [
            {"frame_number": 1, "gt_binary": 1},
            {"frame_number": 2, "gt_binary": 0},
            {"frame_number": 3, "gt_binary": 0},
            {"frame_number": 4, "gt_binary": 0},
        ],
    )
    result = evaluate_single_dataset(
        dataset_key="demo",
        selected_csv=selected_csv,
        gt_csv=gt_csv,
    ).metrics
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 2
    assert result["fn"] == 0
    assert result["positive_precision"] == pytest.approx(0.5)
    assert result["positive_recall"] == pytest.approx(1.0)


def test_evaluate_single_dataset_label_gt(tmp_path: Path):
    selected_csv = _write_selected_csv(
        tmp_path / "tmp_selected_label.csv",
        [
            {"frame_id": 10, "predicted_label": 1},
            {"frame_id": 11, "predicted_label": 0},
        ],
    )
    gt_csv = _write_gt_label_csv(
        tmp_path / "tmp_gt_label.csv",
        [
            {"frame_id": "frame_00010.jpg", "label": "holding"},
            {"frame_id": "frame_00011.jpg", "label": "not_holding"},
        ],
    )
    result = evaluate_single_dataset(
        dataset_key="demo_label",
        selected_csv=selected_csv,
        gt_csv=gt_csv,
    ).metrics
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["tp"] == 1
    assert result["tn"] == 1


def test_evaluate_single_dataset_tracks_missing_gt(tmp_path: Path):
    selected_csv = _write_selected_csv(
        tmp_path / "tmp_selected_missing.csv",
        [
            {"frame_id": 1, "predicted_label": 1},
            {"frame_id": 2, "predicted_label": 0},
            {"frame_id": 3, "predicted_label": 1},
        ],
    )
    gt_csv = _write_gt_binary_csv(
        tmp_path / "tmp_gt_missing.csv",
        [
            {"frame_number": 1, "gt_binary": 1},
            {"frame_number": 3, "gt_binary": 0},
        ],
    )
    result = evaluate_single_dataset(
        dataset_key="demo_missing",
        selected_csv=selected_csv,
        gt_csv=gt_csv,
    ).metrics
    assert result["n_selected_total"] == 3
    assert result["n_selected_with_gt"] == 2
    assert result["n_selected_missing_gt"] == 1
    assert result["coverage_selected_over_gt"] == pytest.approx(1.0)


def test_evaluate_single_dataset_duplicate_frames_default_and_strict(tmp_path: Path):
    selected_csv = _write_selected_csv(
        tmp_path / "tmp_selected_dupe.csv",
        [
            {"frame_id": 1, "predicted_label": 1},
            {"frame_id": 1, "predicted_label": 0},
        ],
    )
    gt_csv = _write_gt_binary_csv(
        tmp_path / "tmp_gt_dupe.csv",
        [
            {"frame_number": 1, "gt_binary": 1},
        ],
    )
    result = evaluate_single_dataset(
        dataset_key="demo_dupe",
        selected_csv=selected_csv,
        gt_csv=gt_csv,
        strict_duplicates=False,
    ).metrics
    assert result["n_selected_total"] == 2
    assert result["n_selected_unique"] == 1
    assert result["n_selected_duplicates_dropped"] == 1
    with pytest.raises(ValueError):
        evaluate_single_dataset(
            dataset_key="demo_dupe",
            selected_csv=selected_csv,
            gt_csv=gt_csv,
            strict_duplicates=True,
        )


def test_evaluate_single_dataset_rejects_invalid_selected_label(tmp_path: Path):
    selected_csv = _write_selected_csv(
        tmp_path / "tmp_selected_invalid.csv",
        [
            {"frame_id": 1, "predicted_label": 2},
        ],
    )
    gt_csv = _write_gt_binary_csv(
        tmp_path / "tmp_gt_invalid.csv",
        [
            {"frame_number": 1, "gt_binary": 1},
        ],
    )
    with pytest.raises(ValueError):
        evaluate_single_dataset(
            dataset_key="demo_invalid",
            selected_csv=selected_csv,
            gt_csv=gt_csv,
        )


def test_evaluate_single_dataset_no_overlap_raises(tmp_path: Path):
    selected_csv = _write_selected_csv(
        tmp_path / "tmp_selected_no_overlap.csv",
        [
            {"frame_id": 1, "predicted_label": 1},
        ],
    )
    gt_csv = _write_gt_binary_csv(
        tmp_path / "tmp_gt_no_overlap.csv",
        [
            {"frame_number": 99, "gt_binary": 1},
        ],
    )
    with pytest.raises(ValueError):
        evaluate_single_dataset(
            dataset_key="demo_no_overlap",
            selected_csv=selected_csv,
            gt_csv=gt_csv,
        )


def test_evaluate_manifest_aggregates_and_filters(tmp_path: Path):
    selected1 = _write_selected_csv(
        tmp_path / "selected" / "sv1.csv",
        [
            {"frame_id": 1, "predicted_label": 1},
            {"frame_id": 2, "predicted_label": 0},
        ],
    )
    gt1 = _write_gt_binary_csv(
        tmp_path / "gt" / "sv1.csv",
        [
            {"frame_number": 1, "gt_binary": 1},
            {"frame_number": 2, "gt_binary": 0},
        ],
    )
    selected2 = _write_selected_csv(
        tmp_path / "selected" / "sv2.csv",
        [
            {"frame_id": 3, "predicted_label": 1},
            {"frame_id": 4, "predicted_label": 1},
        ],
    )
    gt2 = _write_gt_binary_csv(
        tmp_path / "gt" / "sv2.csv",
        [
            {"frame_number": 3, "gt_binary": 0},
            {"frame_number": 4, "gt_binary": 1},
        ],
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "dataset_key": "sv1",
                "selected_csv": str(selected1.relative_to(tmp_path)),
                "gt_csv": str(gt1.relative_to(tmp_path)),
            },
            {
                "dataset_key": "sv2",
                "selected_csv": str(selected2.relative_to(tmp_path)),
                "gt_csv": str(gt2.relative_to(tmp_path)),
            },
        ]
    ).to_csv(manifest_path, index=False)

    results = evaluate_manifest(manifest_csv=manifest_path, dataset_keys={"sv2"})
    assert results["n_datasets_input"] == 1
    assert results["n_datasets_success"] == 1
    assert results["n_datasets_failed"] == 0
    assert results["rows"][0]["dataset_key"] == "sv2"
    assert results["global_micro"] is not None
    assert results["global_macro"] is not None


def test_run_multi_mode_writes_outputs_and_returns_nonzero_on_partial_fail(tmp_path: Path):
    selected_ok = _write_selected_csv(
        tmp_path / "selected_ok.csv",
        [
            {"frame_id": 1, "predicted_label": 1},
        ],
    )
    gt_ok = _write_gt_binary_csv(
        tmp_path / "gt_ok.csv",
        [
            {"frame_number": 1, "gt_binary": 1},
        ],
    )
    manifest_path = tmp_path / "metrics_manifest.csv"
    pd.DataFrame(
        [
            {"dataset_key": "ok", "selected_csv": str(selected_ok), "gt_csv": str(gt_ok)},
            {"dataset_key": "bad", "selected_csv": "missing.csv", "gt_csv": str(gt_ok)},
        ]
    ).to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    args = type(
        "Args",
        (),
        {
            "selected_csv": None,
            "gt_csv": None,
            "dataset_key": "single_dataset",
            "manifest_csv": str(manifest_path),
            "dataset_keys": None,
            "output_dir": str(output_dir),
            "json_out": None,
            "strict_duplicates": False,
        },
    )()
    rc = run(args)
    assert rc == 1

    per_dataset_csv = output_dir / "selected_timestamp_metrics_per_dataset.csv"
    summary_json = output_dir / "selected_timestamp_metrics_summary.json"
    assert per_dataset_csv.exists()
    assert summary_json.exists()

    out_df = pd.read_csv(per_dataset_csv)
    status_by_dataset = {str(row["dataset_key"]): str(row["status"]) for _, row in out_df.iterrows()}
    assert status_by_dataset["ok"] == "success"
    assert status_by_dataset["bad"] == "failed"
