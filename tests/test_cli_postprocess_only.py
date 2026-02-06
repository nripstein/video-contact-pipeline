from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

import run_pipeline


def _base_args() -> Namespace:
    return Namespace(
        input_path=None,
        output_dir=None,
        thresh_hand=0.5,
        thresh_obj=0.5,
        no_crop=False,
        no_flip=False,
        obj_bigger_filter=False,
        obj_bigger_k=1.0,
        blue_threshold=0.5,
        verbose=False,
        preprocess_only=False,
        inference_only=False,
        no_blue_glove_filter=False,
        no_object_size_filter=False,
        object_size_max_area_ratio=0.5,
        no_visualizations=False,
        gt_csv_path=None,
        no_progress=False,
        save_annotated_frames=False,
        skip_existing=False,
        barcodes_only=False,
        annotated_frames_only=False,
        condensed_csv=None,
        full_csv=None,
        image_dir=None,
    )


def test_barcodes_only_from_condensed_csv(tmp_path: Path, monkeypatch):
    condensed_path = tmp_path / "detections_condensed.csv"
    pd.DataFrame(
        {
            "frame_id": ["000001.png", "000002.png", "000003.png"],
            "frame_number": [1, 2, 3],
            "contact_label": ["No Contact", "Portable Object", "No Contact"],
            "source_hand": ["Left", "Left", "Right"],
        }
    ).to_csv(condensed_path, index=False)

    calls = {}

    def fake_save_barcodes(condensed_df, output_dir, gt_csv_path=None):
        calls["rows"] = len(condensed_df)
        calls["output_dir"] = output_dir
        calls["gt"] = gt_csv_path
        return [str(Path(output_dir) / "visualizations" / "barcode_pred.png")]

    monkeypatch.setattr(run_pipeline, "save_barcodes", fake_save_barcodes)

    args = _base_args()
    args.barcodes_only = True
    args.condensed_csv = str(condensed_path)

    rc = run_pipeline._run_postprocess_only(args)
    assert rc == 0
    assert calls["rows"] == 3
    assert calls["output_dir"] == str(tmp_path)


def test_annotated_frames_only_from_full_csv(tmp_path: Path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    full_path = tmp_path / "detections_full.csv"
    pd.DataFrame(
        [
            {
                "frame_id": "000001.png",
                "frame_number": 1,
                "detection_type": "hand",
                "bbox_x1": 1,
                "bbox_y1": 1,
                "bbox_x2": 4,
                "bbox_y2": 4,
                "confidence": 95.0,
                "contact_state": 0,
                "contact_label": "No Contact",
                "hand_side": "Left",
                "offset_x": 0.0,
                "offset_y": 0.0,
                "offset_mag": 0.0,
                "blue_prop": None,
                "blue_glove_status": "NA",
                "is_filtered": False,
                "filtered_by": "",
                "filtered_reason": "",
            }
        ]
    ).to_csv(full_path, index=False)

    calls = {}

    def fake_save_annotated_frames(image_dir, full_df, output_dir):
        calls["image_dir"] = image_dir
        calls["rows"] = len(full_df)
        calls["output_dir"] = output_dir
        return [str(Path(output_dir) / "visualizations" / "frames_det" / "000001_det.png")]

    monkeypatch.setattr(run_pipeline, "save_annotated_frames", fake_save_annotated_frames)

    args = _base_args()
    args.annotated_frames_only = True
    args.full_csv = str(full_path)
    args.image_dir = str(frames_dir)

    rc = run_pipeline._run_postprocess_only(args)
    assert rc == 0
    assert calls["image_dir"] == str(frames_dir)
    assert calls["rows"] == 1
    assert calls["output_dir"] == str(tmp_path)


def test_postprocess_only_requires_required_paths():
    args = _base_args()
    args.barcodes_only = True

    try:
        run_pipeline._run_postprocess_only(args)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "--condensed-csv" in str(exc)
