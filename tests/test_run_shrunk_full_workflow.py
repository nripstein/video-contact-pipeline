from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_shrunk_full_workflow.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_shrunk_full_workflow", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_run_root_prefers_run_name(tmp_path: Path):
    mod = _load_module()

    import sys

    original_argv = sys.argv
    sys.argv = [
        "run_shrunk_full_workflow.py",
        "--data-root",
        str(tmp_path),
        "--run-name",
        "custom_run",
        "--dry-run",
    ]
    try:
        args = mod.parse_args()
    finally:
        sys.argv = original_argv

    resolved = mod._resolve_run_root(args)
    assert resolved == (tmp_path / "pipeline_runs" / "custom_run")


def test_main_dry_run_builds_inference_and_video_commands(tmp_path: Path):
    mod = _load_module()

    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "sv1",
                "status": "success",
                "pred_dir": str(run_root / "predictions" / "sv1"),
                "input_dir": str(tmp_path / "frames"),
                "gt_csv": str(tmp_path / "gt.csv"),
            }
        ]
    )
    manifest.to_csv(run_root / "run_manifest.csv", index=False)

    cmds = []

    def _fake_run_subprocess(cmd, *, dry_run):
        cmds.append((cmd, dry_run))
        return 0

    def _fake_generate_frames_det(**kwargs):
        return 0

    original_run = mod._run_subprocess
    original_frames = mod._generate_frames_det
    mod._run_subprocess = _fake_run_subprocess
    mod._generate_frames_det = _fake_generate_frames_det
    try:
        import sys

        original_argv = sys.argv
        sys.argv = [
            "run_shrunk_full_workflow.py",
            "--run-root",
            str(run_root),
            "--dry-run",
            "--datasets",
            "sv1",
        ]
        try:
            rc = mod.main()
        finally:
            sys.argv = original_argv
    finally:
        mod._run_subprocess = original_run
        mod._generate_frames_det = original_frames

    assert rc == 0
    assert len(cmds) == 2
    infer_cmd = " ".join(cmds[0][0])
    video_cmd = " ".join(cmds[1][0])
    assert "run_shrunk_inference_batch.py" in infer_cmd
    assert "--condense-priority-strategy no_contact_first" in infer_cmd
    assert "render_timeline_videos_from_manifest.py" in video_cmd
    assert cmds[0][1] is True
    assert cmds[1][1] is True


def test_main_forwards_strict_portable_flags_to_inference(tmp_path: Path):
    mod = _load_module()

    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        [
            {
                "dataset_key": "sv1",
                "status": "success",
                "pred_dir": str(run_root / "predictions" / "sv1"),
                "input_dir": str(tmp_path / "frames"),
                "gt_csv": str(tmp_path / "gt.csv"),
            }
        ]
    )
    manifest.to_csv(run_root / "run_manifest.csv", index=False)

    cmds = []

    def _fake_run_subprocess(cmd, *, dry_run):
        cmds.append((cmd, dry_run))
        return 0

    def _fake_generate_frames_det(**kwargs):
        return 0

    original_run = mod._run_subprocess
    original_frames = mod._generate_frames_det
    mod._run_subprocess = _fake_run_subprocess
    mod._generate_frames_det = _fake_generate_frames_det
    try:
        import sys

        original_argv = sys.argv
        sys.argv = [
            "run_shrunk_full_workflow.py",
            "--run-root",
            str(run_root),
            "--dry-run",
            "--datasets",
            "sv1",
            "--strict-portable-match",
            "--strict-portable-detected-iou-threshold",
            "0.09",
            "--condense-priority-strategy",
            "portable_first",
        ]
        try:
            rc = mod.main()
        finally:
            sys.argv = original_argv
    finally:
        mod._run_subprocess = original_run
        mod._generate_frames_det = original_frames

    assert rc == 0
    infer_cmd = cmds[0][0]
    assert "--strict-portable-match" in infer_cmd
    assert "--strict-portable-detected-iou-threshold" in infer_cmd
    assert "0.09" in infer_cmd
    assert "--condense-priority-strategy" in infer_cmd
    assert "portable_first" in infer_cmd


def test_generate_frames_det_skips_when_existing_count_sufficient(tmp_path: Path):
    mod = _load_module()

    run_root = tmp_path / "run"
    pred_dir = run_root / "predictions" / "sv1"
    frames_det_dir = pred_dir / "visualizations" / "frames_det"
    pred_dir.mkdir(parents=True, exist_ok=True)
    frames_det_dir.mkdir(parents=True, exist_ok=True)

    (pred_dir / "detections_full.csv").write_text("x", encoding="utf-8")
    (pred_dir / "detections_condensed.csv").write_text(
        "frame_id,frame_number,contact_label,source_hand\n0.jpg,0,No Contact,Left\n1.jpg,1,Portable Object,Right\n",
        encoding="utf-8",
    )
    for name in ["0_det.png", "1_det.png"]:
        (frames_det_dir / name).write_bytes(b"img")

    manifest_df = pd.DataFrame(
        [
            {
                "dataset_key": "sv1",
                "pred_dir": str(pred_dir),
                "input_dir": str(tmp_path / "input_frames"),
                "gt_csv": str(tmp_path / "gt.csv"),
            }
        ]
    )
    (tmp_path / "input_frames").mkdir(parents=True, exist_ok=True)

    rc = mod._generate_frames_det(
        run_root=run_root,
        manifest_df=manifest_df,
        skip_existing=True,
        dry_run=False,
    )
    assert rc == 0
    out_manifest = run_root / "frames_det" / "frames_det_manifest.csv"
    assert out_manifest.exists()
    out = pd.read_csv(out_manifest)
    assert out.iloc[0]["status"] == "success"
    assert out.iloc[0]["action"] == "skipped_existing"
