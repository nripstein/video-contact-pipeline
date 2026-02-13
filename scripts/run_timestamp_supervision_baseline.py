from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timestamp_supervision_extraction.extract_timestamps import run as run_extract_timestamps
from timestamp_supervision_extraction.evaluate_selected_timestamps import run as run_selected_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run timestamp-supervision baseline over one or many datasets from a run manifest: "
            "prepare extractor inputs, select confident frames, and evaluate against GT."
        )
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run root containing run_manifest.csv and prediction directories.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Optional explicit manifest path (default: <run-root>/run_manifest.csv).",
    )
    parser.add_argument(
        "--dataset-keys",
        default=None,
        help="Optional comma-separated dataset keys to include.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root for baseline artifacts (default: <run-root>/timestamp_supervision_baseline).",
    )
    parser.add_argument(
        "--include-failed-manifest-rows",
        action="store_true",
        help="Include manifest rows with status != success. Default keeps only successful rows when status exists.",
    )
    parser.add_argument(
        "--condensed-relpath",
        default="detections_condensed.csv",
        help=(
            "Relative path from each manifest pred_dir to condensed predictions CSV "
            "(e.g., hsmm_refinement/detections_condensed_hsmm.csv)."
        ),
    )
    parser.add_argument(
        "--full-relpath",
        default="detections_full.csv",
        help=(
            "Relative path from each manifest pred_dir to full detections CSV used for glove metadata "
            "(default: detections_full.csv)."
        ),
    )
    parser.add_argument("--fps", type=float, default=60.0, help="FPS for island length threshold conversion.")
    parser.add_argument(
        "--min-island-seconds",
        type=float,
        default=1.0,
        help="Minimum island duration in seconds for confident-frame selection.",
    )
    parser.add_argument(
        "--join-mode",
        choices=("inner", "left"),
        default="inner",
        help="Join mode used by extraction for predictions/metadata.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for center-biased representative frame sampling.",
    )
    parser.add_argument(
        "--strict-duplicates",
        action="store_true",
        help="Fail evaluation if selected CSVs contain duplicate frame_id rows.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without execution.")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dataset_keys(keys_csv: str | None) -> set[str] | None:
    if not keys_csv:
        return None
    out = {item.strip() for item in keys_csv.split(",") if item.strip()}
    return out if out else None


def _resolve_path(base_dir: Path, raw: object) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_source_manifest(
    manifest_csv: Path,
    dataset_keys: set[str] | None,
    include_failed_rows: bool,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    required = {"dataset_key", "pred_dir", "gt_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_csv} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["dataset_key"] = out["dataset_key"].astype(str)
    if "status" in out.columns and not include_failed_rows:
        out = out[out["status"] == "success"]
    if dataset_keys is not None:
        out = out[out["dataset_key"].isin(dataset_keys)]
    out = out.sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)
    if out.empty:
        raise ValueError("No datasets remain after filtering manifest.")
    return out


def _to_binary_label(label: object) -> int:
    if label is None:
        return 0
    text = str(label).strip().lower()
    if text in {"portable object", "portable object contact", "holding", "1", "true", "yes"}:
        return 1
    return 0


def _build_predictions_and_metadata(
    *,
    condensed_csv: Path,
    full_csv: Path,
    out_dir: Path,
) -> Tuple[Path, Path]:
    condensed_df = pd.read_csv(condensed_csv)
    if "frame_number" not in condensed_df.columns or "contact_label" not in condensed_df.columns:
        raise ValueError(f"{condensed_csv} must contain frame_number and contact_label")

    sort_cols = [c for c in ("frame_number", "frame_id") if c in condensed_df.columns]
    condensed_df = condensed_df.sort_values(by=sort_cols, kind="mergesort").drop_duplicates(
        subset=["frame_number"], keep="first"
    )

    predictions_df = pd.DataFrame(
        {
            "frame_id": condensed_df["frame_number"].astype(int),
            "predicted_label": condensed_df["contact_label"].map(_to_binary_label).astype(int),
        }
    )

    full_df = pd.read_csv(full_csv)
    if "frame_number" not in full_df.columns:
        raise ValueError(f"{full_csv} must contain frame_number")

    if "detection_type" in full_df.columns:
        hand_df = full_df[full_df["detection_type"] == "hand"].copy()
    else:
        hand_df = full_df.copy()

    if "blue_glove_status" in hand_df.columns:
        hand_df["blue_glove_detected"] = (
            hand_df["blue_glove_status"].fillna("").astype(str).str.strip().str.lower() == "experimenter"
        )
    else:
        hand_df["blue_glove_detected"] = False

    by_frame = (
        hand_df.groupby("frame_number", sort=False)["blue_glove_detected"].any().astype(bool).to_dict()
        if not hand_df.empty
        else {}
    )
    metadata_df = pd.DataFrame(
        {
            "frame_id": predictions_df["frame_id"].astype(int),
            "blue_glove_detected": predictions_df["frame_id"].map(by_frame).fillna(False).astype(bool),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.csv"
    metadata_path = out_dir / "metadata.csv"
    predictions_df.to_csv(predictions_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)
    return predictions_path, metadata_path


def _prepare_extractor_inputs(
    *,
    manifest_df: pd.DataFrame,
    manifest_base_dir: Path,
    extractor_input_root: Path,
    condensed_relpath: str,
    full_relpath: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in manifest_df.iterrows():
        dataset_key = str(row["dataset_key"])
        pred_dir = _resolve_path(manifest_base_dir, row["pred_dir"])
        gt_csv = _resolve_path(manifest_base_dir, row["gt_csv"])
        condensed_csv = pred_dir / str(condensed_relpath)
        full_csv = pred_dir / str(full_relpath)
        dataset_out_dir = extractor_input_root / dataset_key

        rec: Dict[str, object] = {
            "dataset_key": dataset_key,
            "pred_dir": str(pred_dir),
            "gt_csv": str(gt_csv),
            "condensed_csv": str(condensed_csv),
            "full_csv": str(full_csv),
            "extractor_dataset_dir": str(dataset_out_dir),
            "status": "pending",
            "error": "",
        }
        try:
            if not gt_csv.exists():
                raise FileNotFoundError(f"missing GT CSV: {gt_csv}")
            if not condensed_csv.exists():
                raise FileNotFoundError(f"missing condensed CSV: {condensed_csv}")
            if not full_csv.exists():
                raise FileNotFoundError(f"missing full CSV: {full_csv}")
            _build_predictions_and_metadata(
                condensed_csv=condensed_csv,
                full_csv=full_csv,
                out_dir=dataset_out_dir,
            )
            rec["status"] = "prepared"
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = str(exc)
        rows.append(rec)
    return pd.DataFrame(rows)


def _run_selection_step(
    *,
    extractor_input_root: Path,
    selected_root: Path,
    fps: float,
    min_island_seconds: float,
    join_mode: str,
    random_seed: int | None,
    dry_run: bool,
) -> int:
    if dry_run:
        return 0
    extract_args = argparse.Namespace(
        results_dir=str(extractor_input_root),
        videos_dir="videos",
        frames_dir=None,
        output_dir=str(selected_root),
        fps=float(fps),
        min_island_seconds=float(min_island_seconds),
        join_mode=str(join_mode),
        random_seed=random_seed,
        extract_frames=False,
        backend="opencv",
        image_format="jpg",
    )
    return int(run_extract_timestamps(extract_args))


def _build_eval_manifest(
    *,
    prep_df: pd.DataFrame,
    eval_manifest_csv: Path,
    selected_root: Path,
) -> pd.DataFrame:
    prepared = prep_df[prep_df["status"] == "prepared"].copy()
    rows: List[Dict[str, object]] = []
    for _, row in prepared.iterrows():
        dataset_key = str(row["dataset_key"])
        rows.append(
            {
                "dataset_key": dataset_key,
                "selected_csv": str(selected_root / "per_video" / f"{dataset_key}_selected_timestamps.csv"),
                "gt_csv": str(row["gt_csv"]),
            }
        )
    out_df = pd.DataFrame(rows).sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)
    eval_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(eval_manifest_csv, index=False)
    return out_df


def _run_evaluation_step(
    *,
    eval_manifest_csv: Path,
    eval_output_dir: Path,
    strict_duplicates: bool,
    dry_run: bool,
) -> int:
    if dry_run:
        return 0
    eval_args = argparse.Namespace(
        selected_csv=None,
        gt_csv=None,
        dataset_key="single_dataset",
        manifest_csv=str(eval_manifest_csv),
        dataset_keys=None,
        output_dir=str(eval_output_dir),
        json_out=None,
        strict_duplicates=bool(strict_duplicates),
    )
    return int(run_selected_eval(eval_args))


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser()
    manifest_csv = Path(args.manifest_csv).expanduser() if args.manifest_csv else (run_root / "run_manifest.csv")
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest CSV not found: {manifest_csv}")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else (run_root / "timestamp_supervision_baseline")
    )
    extractor_input_root = output_dir / "extractor_input"
    selected_root = output_dir / "selected_timestamps"
    evaluation_root = output_dir / "evaluation"
    run_state_json = output_dir / "baseline_run_state.json"
    prep_manifest_csv = output_dir / "prep_manifest.csv"
    eval_manifest_csv = output_dir / "evaluation_manifest.csv"

    dataset_keys = _parse_dataset_keys(args.dataset_keys)
    source_manifest = _load_source_manifest(
        manifest_csv=manifest_csv,
        dataset_keys=dataset_keys,
        include_failed_rows=bool(args.include_failed_manifest_rows),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prep_df = _prepare_extractor_inputs(
        manifest_df=source_manifest,
        manifest_base_dir=manifest_csv.parent,
        extractor_input_root=extractor_input_root,
        condensed_relpath=str(args.condensed_relpath),
        full_relpath=str(args.full_relpath),
    )
    prep_df.to_csv(prep_manifest_csv, index=False)

    prepared_count = int((prep_df["status"] == "prepared").sum())
    failed_prep = prep_df[prep_df["status"] == "failed"]
    if prepared_count == 0:
        print("No datasets were prepared successfully. See prep manifest for errors.")
        print(f"prep_manifest: {prep_manifest_csv}")
        return 1

    selection_rc = _run_selection_step(
        extractor_input_root=extractor_input_root,
        selected_root=selected_root,
        fps=float(args.fps),
        min_island_seconds=float(args.min_island_seconds),
        join_mode=str(args.join_mode),
        random_seed=args.random_seed,
        dry_run=bool(args.dry_run),
    )
    if selection_rc != 0:
        print("Selection step failed.")
        print(f"prep_manifest: {prep_manifest_csv}")
        return selection_rc

    eval_manifest_df = _build_eval_manifest(
        prep_df=prep_df,
        eval_manifest_csv=eval_manifest_csv,
        selected_root=selected_root,
    )
    if eval_manifest_df.empty:
        print("Evaluation manifest is empty after preparation.")
        print(f"prep_manifest: {prep_manifest_csv}")
        return 1

    eval_rc = _run_evaluation_step(
        eval_manifest_csv=eval_manifest_csv,
        eval_output_dir=evaluation_root,
        strict_duplicates=bool(args.strict_duplicates),
        dry_run=bool(args.dry_run),
    )

    run_state = {
        "generated_at_utc": _utc_now(),
        "run_root": str(run_root),
        "manifest_csv": str(manifest_csv),
        "output_dir": str(output_dir),
        "prepared_count": prepared_count,
        "failed_prepare_count": int(len(failed_prep)),
        "selection_return_code": int(selection_rc),
        "evaluation_return_code": int(eval_rc),
        "dry_run": bool(args.dry_run),
        "dataset_keys": sorted(dataset_keys) if dataset_keys else None,
        "fps": float(args.fps),
        "min_island_seconds": float(args.min_island_seconds),
        "join_mode": str(args.join_mode),
        "random_seed": args.random_seed,
        "strict_duplicates": bool(args.strict_duplicates),
        "condensed_relpath": str(args.condensed_relpath),
        "full_relpath": str(args.full_relpath),
    }
    run_state_json.write_text(json.dumps(run_state, indent=2, sort_keys=True), encoding="utf-8")

    print(f"prepared_datasets: {prepared_count}")
    print(f"failed_prepare_datasets: {len(failed_prep)}")
    print(f"prep_manifest: {prep_manifest_csv}")
    print(f"evaluation_manifest: {eval_manifest_csv}")
    print(f"evaluation_dir: {evaluation_root}")
    print(f"run_state: {run_state_json}")
    if len(failed_prep) > 0:
        for _, row in failed_prep.iterrows():
            print(f"[prepare_failed] {row['dataset_key']}: {row['error']}")
    return 0 if int(eval_rc) == 0 and len(failed_prep) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
