from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_shrunk_inference_batch.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_shrunk_inference_batch", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_intersection_metrics_uses_common_frames_only(tmp_path: Path):
    mod = _load_script_module()

    condensed_path = tmp_path / "detections_condensed.csv"
    condensed_path.write_text(
        "\n".join(
            [
                "frame_id,frame_number,contact_label,source_hand",
                "1_x.jpg,1,No Contact,Left",
                "2_x.jpg,2,Portable Object,Left",
                "3_x.jpg,3,Portable Object,Right",
            ]
        ),
        encoding="utf-8",
    )

    gt_path = tmp_path / "labels.csv"
    gt_path.write_text(
        "\n".join(
            [
                "video_id,frame_id,label,fps",
                "demo,2_x.jpg,holding,60",
                "demo,3_x.jpg,not_holding,60",
                "demo,4_x.jpg,holding,60",
            ]
        ),
        encoding="utf-8",
    )

    out = mod._compute_intersection_metrics(condensed_path, gt_path)
    assert out["n_frames_common"] == 2
    assert out["MoF"] == 0.5
    assert out["confusion"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}


def test_intersection_metrics_fails_when_no_overlap(tmp_path: Path):
    mod = _load_script_module()

    condensed_path = tmp_path / "detections_condensed.csv"
    condensed_path.write_text(
        "\n".join(
            [
                "frame_id,frame_number,contact_label,source_hand",
                "1_x.jpg,1,No Contact,Left",
                "2_x.jpg,2,Portable Object,Left",
            ]
        ),
        encoding="utf-8",
    )

    gt_path = tmp_path / "labels.csv"
    gt_path.write_text(
        "\n".join(
            [
                "video_id,frame_id,label,fps",
                "demo,10_x.jpg,holding,60",
                "demo,11_x.jpg,not_holding,60",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        mod._compute_intersection_metrics(condensed_path, gt_path)
