from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/nripstein/Documents/thesis data/thesis labels")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tqdm():
    try:
        from tqdm import tqdm

        return tqdm
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end workflow for shrunk datasets: "
            "inference+metrics -> frames_det -> timeline videos."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset root.")
    parser.add_argument(
        "--run-root",
        default=None,
        help="Explicit run root. If omitted, uses --run-name or a date/profile-based default.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name under <data-root>/pipeline_runs when --run-root is omitted.",
    )
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset keys.")

    parser.add_argument(
        "--profile",
        choices=("baseline", "default", "tracking"),
        default="baseline",
        help="Inference profile passed to run_shrunk_inference_batch.py.",
    )
    parser.add_argument(
        "--pipeline-arg",
        action="append",
        default=[],
        help=(
            "Extra run_pipeline token. Repeatable. "
            "Use equals form when token begins with '-'; "
            "e.g. --pipeline-arg=--tracking-max-missed-frames --pipeline-arg=12"
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip existing outputs where possible.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Recompute/re-render even if outputs exist.",
    )
    parser.add_argument("--recompute-all", action="store_true", help="Force inference recompute.")
    parser.add_argument("--no-progress", action="store_true", help="Disable inference progress output.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without doing work.")
    parser.add_argument(
        "--strict-portable-match",
        action="store_true",
        help="Enable strict portable gating during inference.",
    )
    parser.add_argument(
        "--strict-portable-detected-iou-threshold",
        type=float,
        default=0.05,
        help="Detected object IoU threshold used by strict portable gating.",
    )
    parser.add_argument(
        "--condense-priority-strategy",
        choices=("no_contact_first", "portable_first"),
        default="no_contact_first",
        help="Tie-break strategy used when condensing frame-level labels.",
    )

    parser.add_argument("--skip-inference", action="store_true", help="Skip inference+metrics phase.")
    parser.add_argument("--skip-frames-det", action="store_true", help="Skip annotated frames generation phase.")
    parser.add_argument("--skip-videos", action="store_true", help="Skip timeline video phase.")

    parser.add_argument(
        "--frames-det-skip-existing",
        action="store_true",
        default=True,
        help="Skip frames_det generation if enough rendered frames already exist.",
    )
    parser.add_argument(
        "--no-frames-det-skip-existing",
        action="store_false",
        dest="frames_det_skip_existing",
        help="Always regenerate frames_det outputs.",
    )

    parser.add_argument(
        "--video-image-subdir",
        default="visualizations/frames_det",
        help="Image subdir under each pred_dir for timeline rendering.",
    )
    parser.add_argument(
        "--video-output-name",
        default="contact_timeline_frames_det.mp4",
        help="Timeline video filename under pred_dir/visualizations.",
    )
    parser.add_argument(
        "--video-title-prefix",
        default="Contact Timeline (frames_det)",
        help="Timeline title prefix.",
    )
    parser.add_argument(
        "--video-fps-fallback",
        type=float,
        default=30.0,
        help="FPS fallback for timeline rendering.",
    )
    return parser.parse_args()


def _resolve_run_root(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).expanduser()
    if args.run_root:
        return Path(args.run_root).expanduser()
    if args.run_name:
        return data_root / "pipeline_runs" / args.run_name
    stamp = date.today().isoformat()
    return data_root / "pipeline_runs" / f"{stamp}_shrunk_{args.profile}"


def _run_subprocess(cmd: List[str], *, dry_run: bool) -> int:
    print("CMD:", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(result.returncode)


def _load_manifest(run_root: Path) -> pd.DataFrame:
    manifest_path = run_root / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {"dataset_key", "pred_dir", "input_dir", "gt_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")
    if "status" in df.columns:
        df = df[df["status"] == "success"]
    return df.sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)


def _generate_frames_det(
    *,
    run_root: Path,
    manifest_df: pd.DataFrame,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    frames_root = run_root / "frames_det"
    logs_dir = frames_root / "logs"
    manifest_path = frames_root / "frames_det_manifest.csv"
    frames_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    iterable = list(manifest_df.itertuples(index=False))
    tqdm_fn = _tqdm()
    if tqdm_fn is not None:
        iterable = tqdm_fn(iterable, desc="frames_det", unit="ds")

    for item in iterable:
        dataset_key = str(item.dataset_key)
        pred_dir = Path(item.pred_dir).expanduser()
        input_dir = Path(item.input_dir).expanduser()
        full_csv = pred_dir / "detections_full.csv"
        condensed_csv = pred_dir / "detections_condensed.csv"
        output_dir = pred_dir
        frame_det_dir = pred_dir / "visualizations" / "frames_det"

        row = {
            "dataset_key": dataset_key,
            "status": "pending",
            "action": "",
            "error": "",
            "started_at_utc": _utc_now(),
            "ended_at_utc": None,
            "duration_sec": None,
            "full_csv": str(full_csv),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "frames_det_dir": str(frame_det_dir),
            "n_frames_condensed": None,
            "n_frames_det": None,
        }

        started = datetime.now(timezone.utc)
        try:
            if not full_csv.exists():
                raise FileNotFoundError(f"Missing full csv: {full_csv}")
            if not condensed_csv.exists():
                raise FileNotFoundError(f"Missing condensed csv: {condensed_csv}")
            if not input_dir.exists():
                raise FileNotFoundError(f"Missing input image dir: {input_dir}")

            n_frames_condensed = int(len(pd.read_csv(condensed_csv)))
            row["n_frames_condensed"] = n_frames_condensed

            n_existing = 0
            if frame_det_dir.exists():
                n_existing = len([p for p in frame_det_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
            row["n_frames_det"] = int(n_existing)
            if skip_existing and n_existing >= n_frames_condensed and n_frames_condensed > 0:
                row["status"] = "success"
                row["action"] = "skipped_existing"
            else:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "run_pipeline.py"),
                    "--annotated-frames-only",
                    "--full-csv",
                    str(full_csv),
                    "--image-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ]
                if dry_run:
                    row["status"] = "success"
                    row["action"] = "dry_run"
                else:
                    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
                    log_path = logs_dir / f"{dataset_key}.log"
                    log_path.write_text(
                        f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
                        encoding="utf-8",
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"annotated-frames-only failed with code {result.returncode}")
                    row["status"] = "success"
                    row["action"] = "rendered"
                    if frame_det_dir.exists():
                        row["n_frames_det"] = len(
                            [p for p in frame_det_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
                        )
        except Exception as exc:
            row["status"] = "failed"
            row["action"] = "error"
            row["error"] = str(exc)
        finally:
            ended = datetime.now(timezone.utc)
            row["ended_at_utc"] = ended.isoformat()
            row["duration_sec"] = round((ended - started).total_seconds(), 3)
            rows.append(row)
            pd.DataFrame(rows).to_csv(manifest_path, index=False)

    failed = [r for r in rows if r["status"] == "failed"]
    print(f"frames_det manifest: {manifest_path}")
    print(f"frames_det datasets processed: {len(rows)}")
    print(f"frames_det failed datasets: {len(failed)}")
    if failed:
        for row in failed:
            print(f"[failed] {row['dataset_key']}: {row['error']}")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    run_root = _resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Run root: {run_root}")

    if not args.skip_inference:
        infer_cmd: List[str] = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_shrunk_inference_batch.py"),
            "--run-root",
            str(run_root),
            "--profile",
            args.profile,
            "--condense-priority-strategy",
            args.condense_priority_strategy,
        ]
        if args.datasets:
            infer_cmd += ["--datasets", args.datasets]
        if args.recompute_all:
            infer_cmd.append("--recompute-all")
        elif args.skip_existing:
            infer_cmd.append("--skip-existing")
        else:
            infer_cmd.append("--no-skip-existing")
        if args.no_progress:
            infer_cmd.append("--no-progress")
        if args.strict_portable_match:
            infer_cmd.extend(
                [
                    "--strict-portable-match",
                    "--strict-portable-detected-iou-threshold",
                    str(args.strict_portable_detected_iou_threshold),
                ]
            )
        if args.dry_run:
            infer_cmd.append("--dry-run")
        for token in args.pipeline_arg:
            infer_cmd.append(f"--pipeline-arg={token}")

        rc = _run_subprocess(infer_cmd, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
    else:
        print("Skipping inference+metrics phase (--skip-inference).")

    manifest_df = _load_manifest(run_root)
    if args.datasets:
        requested = {k.strip() for k in args.datasets.split(",") if k.strip()}
        manifest_df = manifest_df[manifest_df["dataset_key"].isin(requested)].reset_index(drop=True)
        if manifest_df.empty:
            raise ValueError("No successful datasets remain after --datasets filtering.")

    if not args.skip_frames_det:
        rc = _generate_frames_det(
            run_root=run_root,
            manifest_df=manifest_df,
            skip_existing=bool(args.frames_det_skip_existing),
            dry_run=bool(args.dry_run),
        )
        if rc != 0:
            return rc
    else:
        print("Skipping frames_det phase (--skip-frames-det).")

    if not args.skip_videos:
        video_cmd: List[str] = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "render_timeline_videos_from_manifest.py"),
            "--run-root",
            str(run_root),
            "--image-subdir",
            args.video_image_subdir,
            "--output-name",
            args.video_output_name,
            "--title-prefix",
            args.video_title_prefix,
            "--fps-fallback",
            str(args.video_fps_fallback),
        ]
        if args.datasets:
            video_cmd += ["--dataset-keys", args.datasets]
        if args.skip_existing:
            video_cmd.append("--skip-existing")
        else:
            video_cmd.append("--no-skip-existing")
        if args.dry_run:
            video_cmd.append("--dry-run")

        rc = _run_subprocess(video_cmd, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc
    else:
        print("Skipping videos phase (--skip-videos).")

    print("Workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
