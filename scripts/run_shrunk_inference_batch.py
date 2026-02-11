from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import metrics as _metrics


DEFAULT_DATA_ROOT = Path("/home/nripstein/Documents/thesis data/thesis labels")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    frames_rel: str
    gt_rel: str

    def frames_dir(self, data_root: Path) -> Path:
        return data_root / self.frames_rel

    def gt_csv(self, data_root: Path) -> Path:
        return data_root / self.gt_rel


DATASET_SPECS: List[DatasetSpec] = [
    DatasetSpec("nr_all_nr", "nr labels/all_nr/nr_shrunk", "nr labels/all_nr/all_nr_labels.csv"),
    DatasetSpec("sr1", "sr labels/sr1/sr1_shrunk", "sr labels/sr1/sr1_labels.csv"),
    DatasetSpec("sr2", "sr labels/sr2/sr2_shrunk", "sr labels/sr2/sr2_labels.csv"),
    DatasetSpec("sr3", "sr labels/sr3/sr3_shrunk", "sr labels/sr3/sr3_labels.csv"),
    DatasetSpec("sr4", "sr labels/sr4/sr4_shrunk", "sr labels/sr4/sr4_labels.csv"),
    DatasetSpec("sr_extra1", "sr labels/sr_extra1/sr_shrunk", "sr labels/sr_extra1/sr_extra1_labels.csv"),
    DatasetSpec("sr_extra2", "sr labels/sr_extra2/sr_shrunk", "sr labels/sr_extra2/sr_extra2_labels.csv"),
    DatasetSpec("sv1", "sv labels/sv1_frames/sv1_shrunk", "sv labels/sv1_frames/sv1_frames_labels.csv"),
    DatasetSpec("sv2", "sv labels/sv2_frames/sv2_shrunk", "sv labels/sv2_frames/sv2_frames_labels.csv"),
    DatasetSpec("sv3", "sv labels/sv3_frames/sv3_shrunk", "sv labels/sv3_frames/sv3_frames_labels.csv"),
    DatasetSpec("sv4", "sv labels/sv4_frames/sv4_shrunk", "sv labels/sv4_frames/sv4_frames_labels.csv"),
    DatasetSpec("sv5", "sv labels/sv5_frames/sv5_shrunk", "sv labels/sv5_frames/sv5_frames_labels.csv"),
    DatasetSpec("sv_extra", "sv labels/sv_extra_frames/sv_shrunk", "sv labels/sv_extra_frames/sv_extra_frames_labels.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run parity-style inference over canonical shrunk datasets and compute "
            "frame-intersection metrics."
        )
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root containing NR/SR/SV labels directories.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Explicit output run directory. Default: <data-root>/pipeline_runs/<date>_shrunk_baseline",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset keys to run (default: all canonical keys).",
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "default", "tracking"),
        default="baseline",
        help=(
            "Inference profile. "
            "'baseline' = no-crop/no-flip/no-object-size-filter/no-small-object-filter; "
            "'default' = run_pipeline defaults; "
            "'tracking' = baseline + --tracking-bridge."
        ),
    )
    parser.add_argument(
        "--pipeline-arg",
        action="append",
        default=[],
        help=(
            "Additional run_pipeline flag/value token. Repeatable. "
            "Use equals form for tokens that start with '-'. "
            "Example: --pipeline-arg=--tracking-max-missed-frames --pipeline-arg=12"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip inference for datasets that already have full+condensed CSVs in run root.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Recompute datasets even when outputs already exist.",
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Ignore existing outputs and recompute all selected datasets.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Do not run inference; only compute metrics from existing condensed CSVs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without execution.")
    parser.add_argument("--no-progress", action="store_true", help="Pass --no-progress to run_pipeline.")
    parser.add_argument(
        "--strict-portable-match",
        action="store_true",
        help="Enable strict portable gating in run_pipeline.py.",
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
    return parser.parse_args()


def _default_run_root(data_root: Path) -> Path:
    stamp = date.today().isoformat()
    return data_root / "pipeline_runs" / f"{stamp}_shrunk_baseline"


def _profile_pipeline_args(profile: str) -> List[str]:
    if profile == "baseline":
        return [
            "--no-crop",
            "--no-flip",
            "--no-object-size-filter",
            "--no-small-object-filter",
        ]
    if profile == "default":
        return []
    if profile == "tracking":
        return [
            "--no-crop",
            "--no-flip",
            "--no-object-size-filter",
            "--no-small-object-filter",
            "--tracking-bridge",
        ]
    raise ValueError(f"Unsupported profile: {profile}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_index() -> Dict[str, DatasetSpec]:
    return {spec.key: spec for spec in DATASET_SPECS}


def _select_specs(keys_csv: str | None) -> List[DatasetSpec]:
    if not keys_csv:
        return DATASET_SPECS
    selected = []
    index = _dataset_index()
    for key in [item.strip() for item in keys_csv.split(",") if item.strip()]:
        if key not in index:
            raise ValueError(f"Unknown dataset key '{key}'. Available: {sorted(index.keys())}")
        selected.append(index[key])
    return selected


def _label_to_binary(label: object) -> int:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return 0
    text = str(label).strip().lower()
    if text in {"holding", "portable object", "portable object contact", "1", "true", "yes"}:
        return 1
    if text in {"not_holding", "no contact", "stationary object", "stationary object contact", "0", "false", "no"}:
        return 0
    return 0


def _extract_frame_number(frame_id: object) -> int | None:
    if frame_id is None or (isinstance(frame_id, float) and np.isnan(frame_id)):
        return None
    match = re.search(r"(\d+)", str(frame_id))
    if not match:
        return None
    return int(match.group(1))


def _pred_map_from_condensed(path: Path) -> Dict[int, int]:
    df = pd.read_csv(path)
    if "frame_number" not in df.columns or "contact_label" not in df.columns:
        raise ValueError(f"{path} must contain frame_number and contact_label.")
    df = df.sort_values(by=["frame_number", "frame_id"], kind="mergesort").drop_duplicates(
        subset=["frame_number"], keep="first"
    )
    mapping: Dict[int, int] = {}
    for _, row in df.iterrows():
        frame_number = int(row["frame_number"])
        mapping[frame_number] = 1 if str(row["contact_label"]) == "Portable Object" else 0
    return mapping


def _gt_map_from_csv(path: Path) -> Dict[int, int]:
    gt_df = pd.read_csv(path, sep=None, engine="python")
    if "frame_number" in gt_df.columns and "gt_binary" in gt_df.columns:
        out: Dict[int, int] = {}
        for _, row in gt_df.iterrows():
            out[int(row["frame_number"])] = int(row["gt_binary"])
        return out
    if "frame_id" in gt_df.columns and "label" in gt_df.columns:
        out = {}
        for _, row in gt_df.iterrows():
            frame_number = _extract_frame_number(row["frame_id"])
            if frame_number is None:
                continue
            out[frame_number] = _label_to_binary(row["label"])
        return out
    raise ValueError(f"{path} must contain (frame_number,gt_binary) or (frame_id,label).")


def _compute_intersection_metrics(condensed_csv: Path, gt_csv: Path) -> Dict[str, object]:
    pred_map = _pred_map_from_condensed(condensed_csv)
    gt_map = _gt_map_from_csv(gt_csv)

    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common:
        raise ValueError(f"No overlapping frame numbers between {condensed_csv} and {gt_csv}.")

    pred_bin = np.array([pred_map[frame] for frame in common], dtype=int)
    gt_bin = np.array([gt_map[frame] for frame in common], dtype=int)

    mof = float(_metrics.frame_accuracy(pred_bin, gt_bin))
    edit = float(_metrics.edit_score(pred_bin, gt_bin, bg_class=(0,), norm=True))
    f1 = {
        "F1@10": float(_metrics.f_score(pred_bin, gt_bin, 0.1, bg_class=(0,))),
        "F1@25": float(_metrics.f_score(pred_bin, gt_bin, 0.25, bg_class=(0,))),
        "F1@50": float(_metrics.f_score(pred_bin, gt_bin, 0.5, bg_class=(0,))),
        "F1@75": float(_metrics.f_score(pred_bin, gt_bin, 0.75, bg_class=(0,))),
    }
    tp, fp, tn, fn = _metrics.confusion_counts(pred_bin, gt_bin)
    return {
        "MoF": mof,
        "MoF_pct": mof * 100.0,
        "Edit": edit,
        "F1": f1,
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n_frames_common": int(len(common)),
        "frame_number_min": int(common[0]),
        "frame_number_max": int(common[-1]),
        "n_pred_unique_frames": int(len(pred_map)),
        "n_gt_unique_frames": int(len(gt_map)),
    }


def _run_inference(
    *,
    spec: DatasetSpec,
    data_root: Path,
    pred_dir: Path,
    pipeline_args: Sequence[str],
    no_progress: bool,
    dry_run: bool,
    log_path: Path,
) -> tuple[bool, str]:
    input_dir = spec.frames_dir(data_root)
    gt_csv = spec.gt_csv(data_root)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_pipeline.py"),
        "--input",
        str(input_dir),
        "--output-dir",
        str(pred_dir),
        "--gt-csv",
        str(gt_csv),
        *pipeline_args,
    ]
    if no_progress:
        cmd.append("--no-progress")

    if dry_run:
        return True, "dry_run"

    pred_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    log_path.write_text(
        f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        return False, f"run_pipeline failed with code {result.returncode}."
    return True, "ok"


def _write_summary_markdown(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("# Intersection Metrics Summary\n\nNo successful dataset metrics.\n", encoding="utf-8")
        return
    cols = [
        "dataset_key",
        "n_frames_common",
        "MoF",
        "MoF_pct",
        "Edit",
        "F1@10",
        "F1@25",
        "F1@50",
        "F1@75",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    rows = [f"| {' | '.join(cols)} |", f"| {' | '.join(['---'] * len(cols))} |"]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in cols]
        rows.append(f"| {' | '.join(values)} |")
    text = "# Intersection Metrics Summary\n\n" + "\n".join(rows) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    run_root = Path(args.run_root).expanduser() if args.run_root else _default_run_root(data_root)
    specs = _select_specs(args.datasets)
    pipeline_args: List[str] = [*_profile_pipeline_args(args.profile), *list(args.pipeline_arg)]
    if args.strict_portable_match:
        pipeline_args.extend(
            [
                "--strict-portable-match",
                "--strict-portable-detected-iou-threshold",
                str(args.strict_portable_detected_iou_threshold),
            ]
        )
    pipeline_args.extend(
        [
            "--condense-priority-strategy",
            args.condense_priority_strategy,
        ]
    )

    if args.recompute_all:
        args.skip_existing = False

    pred_root = run_root / "predictions"
    metrics_root = run_root / "metrics"
    logs_root = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_settings = {
        "generated_at_utc": _utc_now(),
        "data_root": str(data_root),
        "run_root": str(run_root),
        "profile": args.profile,
        "pipeline_args": pipeline_args,
        "datasets": [spec.key for spec in specs],
        "skip_existing": bool(args.skip_existing),
        "recompute_all": bool(args.recompute_all),
        "metrics_only": bool(args.metrics_only),
        "dry_run": bool(args.dry_run),
        "strict_portable_match": bool(args.strict_portable_match),
        "strict_portable_detected_iou_threshold": float(args.strict_portable_detected_iou_threshold),
        "condense_priority_strategy": str(args.condense_priority_strategy),
    }
    (run_root / "run_settings.json").write_text(json.dumps(run_settings, indent=2, sort_keys=True), encoding="utf-8")

    manifest_rows: List[Dict[str, object]] = []
    metrics_rows: List[Dict[str, object]] = []

    for spec in specs:
        pred_dir = pred_root / spec.key
        condensed_csv = pred_dir / "detections_condensed.csv"
        full_csv = pred_dir / "detections_full.csv"
        gt_csv = spec.gt_csv(data_root)
        log_path = logs_root / f"{spec.key}.log"
        per_metrics_dir = metrics_root / spec.key
        per_metrics_json = per_metrics_dir / "metrics_intersection.json"

        row = {
            "dataset_key": spec.key,
            "input_dir": str(spec.frames_dir(data_root)),
            "gt_csv": str(gt_csv),
            "pred_dir": str(pred_dir),
            "status": "pending",
            "started_at_utc": _utc_now(),
            "ended_at_utc": None,
            "duration_sec": None,
            "error": "",
            "inference_action": "",
            "barcode_pred_exists": False,
            "barcode_pred_vs_gt_exists": False,
            "n_frames_common": None,
        }

        start = datetime.now(timezone.utc)
        try:
            if not gt_csv.exists():
                raise FileNotFoundError(f"GT CSV not found: {gt_csv}")
            if not spec.frames_dir(data_root).exists():
                raise FileNotFoundError(f"Input frames dir not found: {spec.frames_dir(data_root)}")

            should_skip = args.skip_existing and full_csv.exists() and condensed_csv.exists()
            if args.metrics_only:
                row["inference_action"] = "metrics_only"
            elif should_skip:
                row["inference_action"] = "skipped_existing"
            else:
                ok, note = _run_inference(
                    spec=spec,
                    data_root=data_root,
                    pred_dir=pred_dir,
                    pipeline_args=pipeline_args,
                    no_progress=args.no_progress,
                    dry_run=args.dry_run,
                    log_path=log_path,
                )
                if not ok:
                    raise RuntimeError(note)
                row["inference_action"] = "ran" if note == "ok" else note

            if not args.dry_run:
                if not condensed_csv.exists():
                    raise FileNotFoundError(f"Missing condensed output: {condensed_csv}")
                if not full_csv.exists():
                    raise FileNotFoundError(f"Missing full output: {full_csv}")

                metrics = _compute_intersection_metrics(condensed_csv, gt_csv)
                per_metrics_dir.mkdir(parents=True, exist_ok=True)
                per_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

                metric_row = {
                    "dataset_key": spec.key,
                    "n_frames_common": metrics["n_frames_common"],
                    "MoF": round(metrics["MoF"], 4),
                    "MoF_pct": round(metrics["MoF_pct"], 2),
                    "Edit": round(metrics["Edit"], 2),
                    "F1@10": round(metrics["F1"]["F1@10"], 2),
                    "F1@25": round(metrics["F1"]["F1@25"], 2),
                    "F1@50": round(metrics["F1"]["F1@50"], 2),
                    "F1@75": round(metrics["F1"]["F1@75"], 2),
                    "tp": metrics["confusion"]["tp"],
                    "fp": metrics["confusion"]["fp"],
                    "tn": metrics["confusion"]["tn"],
                    "fn": metrics["confusion"]["fn"],
                }
                metrics_rows.append(metric_row)
                row["n_frames_common"] = metrics["n_frames_common"]

            row["barcode_pred_exists"] = (pred_dir / "visualizations" / "barcode_pred.png").exists()
            row["barcode_pred_vs_gt_exists"] = (pred_dir / "visualizations" / "barcode_pred_vs_gt.png").exists()
            row["status"] = "success" if not args.dry_run else "dry_run"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        finally:
            end = datetime.now(timezone.utc)
            row["ended_at_utc"] = end.isoformat()
            row["duration_sec"] = round((end - start).total_seconds(), 3)
            manifest_rows.append(row)
            pd.DataFrame(manifest_rows).to_csv(run_root / "run_manifest.csv", index=False)

    if not args.dry_run:
        metrics_df = pd.DataFrame(metrics_rows)
        if not metrics_df.empty:
            numeric_cols = ["MoF", "MoF_pct", "Edit", "F1@10", "F1@25", "F1@50", "F1@75"]
            macro = {"dataset_key": "__macro__", "n_frames_common": int(metrics_df["n_frames_common"].sum())}
            for col in numeric_cols:
                macro[col] = round(float(metrics_df[col].mean()), 4)
            for col in ["tp", "fp", "tn", "fn"]:
                macro[col] = int(metrics_df[col].sum())

            total_frames = float(metrics_df["n_frames_common"].sum())
            weighted = {"dataset_key": "__weighted__", "n_frames_common": int(total_frames)}
            if total_frames > 0:
                weights = metrics_df["n_frames_common"] / total_frames
                for col in numeric_cols:
                    weighted[col] = round(float((metrics_df[col] * weights).sum()), 4)
            else:
                for col in numeric_cols:
                    weighted[col] = np.nan
            for col in ["tp", "fp", "tn", "fn"]:
                weighted[col] = int(metrics_df[col].sum())

            summary_df = pd.concat(
                [
                    metrics_df.sort_values(by=["dataset_key"], kind="mergesort"),
                    pd.DataFrame([macro, weighted]),
                ],
                ignore_index=True,
            )
        else:
            summary_df = pd.DataFrame(
                columns=[
                    "dataset_key",
                    "n_frames_common",
                    "MoF",
                    "MoF_pct",
                    "Edit",
                    "F1@10",
                    "F1@25",
                    "F1@50",
                    "F1@75",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                ]
            )

        summary_csv = metrics_root / "summary_intersection.csv"
        summary_md = metrics_root / "summary_intersection.md"
        summary_df.to_csv(summary_csv, index=False)
        _write_summary_markdown(summary_df, summary_md)

    failed = [row for row in manifest_rows if row["status"] == "failed"]
    print(f"Run root: {run_root}")
    print(f"Datasets processed: {len(manifest_rows)}")
    print(f"Failed datasets: {len(failed)}")
    if failed:
        for row in failed:
            print(f"[failed] {row['dataset_key']}: {row['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
