from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "render_timeline_videos_from_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_timeline_videos_from_manifest", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_fps_from_gt_prefers_gt_value_and_fallback(tmp_path: Path):
    mod = _load_module()

    gt_with_fps = tmp_path / "gt_with_fps.csv"
    gt_with_fps.write_text(
        "\n".join(
            [
                "video_id,frame_id,label,fps",
                "demo,0_demo.jpg,holding,29.97",
                "demo,1_demo.jpg,not_holding,29.97",
            ]
        ),
        encoding="utf-8",
    )
    fps = mod._extract_fps_from_gt(gt_with_fps, fallback=30.0)
    assert abs(fps - 29.97) < 1e-9

    gt_no_fps = tmp_path / "gt_no_fps.csv"
    gt_no_fps.write_text(
        "\n".join(
            [
                "video_id,frame_id,label",
                "demo,0_demo.jpg,holding",
                "demo,1_demo.jpg,not_holding",
            ]
        ),
        encoding="utf-8",
    )
    fps2 = mod._extract_fps_from_gt(gt_no_fps, fallback=30.0)
    assert fps2 == 30.0


def test_render_single_dataset_skip_existing_and_dry_run(tmp_path: Path):
    mod = _load_module()

    pred_dir = tmp_path / "preds" / "demo"
    image_dir = pred_dir / "visualizations" / "frames_det"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Condensed has 2 frames; create at least 2 images.
    (pred_dir / "detections_condensed.csv").write_text(
        "\n".join(
            [
                "frame_id,frame_number,contact_label,source_hand",
                "0_demo.jpg,0,No Contact,Left",
                "1_demo.jpg,1,Portable Object,Right",
            ]
        ),
        encoding="utf-8",
    )
    (image_dir / "0_demo_det.png").write_bytes(b"img")
    (image_dir / "1_demo_det.png").write_bytes(b"img")

    gt_csv = tmp_path / "gt.csv"
    gt_csv.write_text(
        "\n".join(
            [
                "video_id,frame_id,label,fps",
                "demo,0_demo.jpg,not_holding,60",
                "demo,1_demo.jpg,holding,60",
            ]
        ),
        encoding="utf-8",
    )

    logs_dir = tmp_path / "logs"

    # Existing output -> skipped_existing when skip_existing=True.
    output_video = pred_dir / "visualizations" / "contact_timeline_frames_det.mp4"
    output_video.write_bytes(b"existing")
    row = mod._render_single_dataset(
        dataset_key="demo",
        pred_dir=pred_dir,
        gt_csv=gt_csv,
        image_subdir="visualizations/frames_det",
        output_name="contact_timeline_frames_det.mp4",
        title_prefix="Contact Timeline (frames_det)",
        fps_fallback=30.0,
        skip_existing=True,
        dry_run=False,
        logs_dir=logs_dir,
    )
    assert row["status"] == "success"
    assert row["action"] == "skipped_existing"

    # Dry run should not invoke rendering and reports dry_run.
    output_video.unlink()
    row2 = mod._render_single_dataset(
        dataset_key="demo",
        pred_dir=pred_dir,
        gt_csv=gt_csv,
        image_subdir="visualizations/frames_det",
        output_name="contact_timeline_frames_det.mp4",
        title_prefix="Contact Timeline (frames_det)",
        fps_fallback=30.0,
        skip_existing=True,
        dry_run=True,
        logs_dir=logs_dir,
    )
    assert row2["status"] == "success"
    assert row2["action"] == "dry_run"


def test_main_filters_success_rows_and_writes_manifest(tmp_path: Path):
    mod = _load_module()

    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    pred_success = run_root / "predictions" / "ok"
    pred_failed = run_root / "predictions" / "bad"
    pred_success.mkdir(parents=True, exist_ok=True)
    pred_failed.mkdir(parents=True, exist_ok=True)

    gt_ok = tmp_path / "gt_ok.csv"
    gt_bad = tmp_path / "gt_bad.csv"
    gt_ok.write_text("video_id,frame_id,label,fps\nok,0_ok.jpg,holding,60\n", encoding="utf-8")
    gt_bad.write_text("video_id,frame_id,label,fps\nbad,0_bad.jpg,holding,60\n", encoding="utf-8")

    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "ok",
                "status": "success",
                "pred_dir": str(pred_success),
                "gt_csv": str(gt_ok),
            },
            {
                "dataset_key": "bad",
                "status": "failed",
                "pred_dir": str(pred_failed),
                "gt_csv": str(gt_bad),
            },
        ]
    )
    manifest_path = run_root / "run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Monkeypatch renderer to avoid heavy work.
    def _fake_render(**kwargs):
        return {
            "dataset_key": kwargs["dataset_key"],
            "status": "success",
            "action": "dry_run",
            "error": "",
            "started_at_utc": "x",
            "ended_at_utc": "y",
            "duration_sec": 0.0,
            "pred_dir": str(kwargs["pred_dir"]),
            "gt_csv": str(kwargs["gt_csv"]),
            "image_dir": str(kwargs["pred_dir"] / kwargs["image_subdir"]),
            "condensed_csv": str(kwargs["pred_dir"] / "detections_condensed.csv"),
            "output_video": str(kwargs["pred_dir"] / "visualizations" / kwargs["output_name"]),
            "fps": 60.0,
            "n_frames_condensed": 1,
            "n_images": 1,
        }

    original = mod._render_single_dataset
    mod._render_single_dataset = _fake_render
    try:
        import sys

        original_argv = sys.argv
        sys.argv = [
            "render_timeline_videos_from_manifest.py",
            "--run-root",
            str(run_root),
            "--dry-run",
        ]
        try:
            rc = mod.main()
        finally:
            sys.argv = original_argv
    finally:
        mod._render_single_dataset = original

    assert rc == 0
    render_manifest = run_root / "videos" / "contact_timeline_frames_det_manifest.csv"
    assert render_manifest.exists()
    out_df = pd.read_csv(render_manifest)
    # Should include only the source manifest rows with status=success.
    assert out_df["dataset_key"].tolist() == ["ok"]
