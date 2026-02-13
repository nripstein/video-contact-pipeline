from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frame-number coverage between predictions and GT across datasets listed in a run manifest."
        )
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Run root containing run_manifest.csv (e.g., results/all_preds).",
    )
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Explicit manifest path. If omitted, uses <run-root>/run_manifest.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path (default: <run-root>/metrics/coverage_audit.csv when --run-root is set).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional output summary JSON path (default: <run-root>/metrics/coverage_audit_summary.json when --run-root is set).",
    )
    parser.add_argument(
        "--min-gt-coverage-over-pred",
        type=float,
        default=None,
        help=(
            "Fail (exit code 1) if any dataset has overlap/pred_unique_frames below this threshold "
            "(e.g., 0.98 for >=98%%)."
        ),
    )
    parser.add_argument(
        "--require-missing-gt-tail-only",
        action="store_true",
        help="Fail if any dataset has missing GT frames for predictions that are not trailing-tail only.",
    )
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="Print segment summaries for missing frame ranges.",
    )
    return parser.parse_args()


def _extract_frame_number(value: object) -> int | None:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def _unique_frame_numbers(df: pd.DataFrame) -> List[int]:
    if "frame_number" in df.columns:
        series = pd.to_numeric(df["frame_number"], errors="coerce").dropna().astype(int)
        return sorted(set(series.tolist()))
    if "frame_id" in df.columns:
        extracted = df["frame_id"].map(_extract_frame_number)
        series = pd.Series(extracted).dropna().astype(int)
        return sorted(set(series.tolist()))
    raise ValueError("Expected frame_number or frame_id column.")


def _segments(values: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not values:
        return []
    out: List[Tuple[int, int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev, prev - start + 1))
        start = value
        prev = value
    out.append((start, prev, prev - start + 1))
    return out


def _default_paths(
    run_root: Path | None,
    manifest_csv: Path | None,
    output_csv: Path | None,
    summary_json: Path | None,
) -> tuple[Path, Path | None, Path | None]:
    if manifest_csv is None:
        if run_root is None:
            raise ValueError("Provide --manifest-csv or --run-root.")
        manifest_csv = run_root / "run_manifest.csv"

    if output_csv is None and run_root is not None:
        output_csv = run_root / "metrics" / "coverage_audit.csv"
    if summary_json is None and run_root is not None:
        summary_json = run_root / "metrics" / "coverage_audit_summary.json"
    return manifest_csv, output_csv, summary_json


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser() if args.run_root else None
    manifest_csv = Path(args.manifest_csv).expanduser() if args.manifest_csv else None
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else None
    summary_json = Path(args.summary_json).expanduser() if args.summary_json else None

    manifest_csv, output_csv, summary_json = _default_paths(run_root, manifest_csv, output_csv, summary_json)

    manifest = pd.read_csv(manifest_csv)
    required_cols = {"dataset_key", "pred_dir", "gt_csv"}
    missing_cols = required_cols - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"{manifest_csv} is missing required columns: {sorted(missing_cols)}")

    rows: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        dataset_key = str(row["dataset_key"])
        pred_dir = Path(str(row["pred_dir"])).expanduser()
        gt_csv = Path(str(row["gt_csv"])).expanduser()
        pred_csv = pred_dir / "detections_condensed.csv"

        pred_df = pd.read_csv(pred_csv)
        gt_df = pd.read_csv(gt_csv, sep=None, engine="python")
        pred_nums = _unique_frame_numbers(pred_df)
        gt_nums = _unique_frame_numbers(gt_df)

        pred_set = set(pred_nums)
        gt_set = set(gt_nums)
        overlap_set = pred_set & gt_set
        missing_gt_for_pred = sorted(pred_set - gt_set)
        missing_pred_for_gt = sorted(gt_set - pred_set)

        pred_n = len(pred_set)
        gt_n = len(gt_set)
        overlap_n = len(overlap_set)
        pred_min = min(pred_set) if pred_set else None
        pred_max = max(pred_set) if pred_set else None
        gt_min = min(gt_set) if gt_set else None
        gt_max = max(gt_set) if gt_set else None

        gt_coverage_over_pred = float(overlap_n / pred_n) if pred_n > 0 else None
        pred_coverage_over_gt = float(overlap_n / gt_n) if gt_n > 0 else None
        missing_gt_tail_only = bool(missing_gt_for_pred) and gt_max is not None and min(missing_gt_for_pred) > gt_max
        missing_gt_segments = _segments(missing_gt_for_pred)
        missing_pred_segments = _segments(missing_pred_for_gt)

        rows.append(
            {
                "dataset_key": dataset_key,
                "pred_n_unique_frames": pred_n,
                "gt_n_unique_frames": gt_n,
                "overlap_n_frames": overlap_n,
                "gt_coverage_over_pred": round(gt_coverage_over_pred, 6) if gt_coverage_over_pred is not None else None,
                "pred_coverage_over_gt": round(pred_coverage_over_gt, 6) if pred_coverage_over_gt is not None else None,
                "missing_gt_for_pred_count": len(missing_gt_for_pred),
                "missing_pred_for_gt_count": len(missing_pred_for_gt),
                "pred_min_frame": pred_min,
                "pred_max_frame": pred_max,
                "gt_min_frame": gt_min,
                "gt_max_frame": gt_max,
                "missing_gt_tail_only": bool(missing_gt_tail_only),
                "missing_gt_segments": json.dumps(missing_gt_segments),
                "missing_pred_segments": json.dumps(missing_pred_segments),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)
    display_cols = [
        "dataset_key",
        "pred_n_unique_frames",
        "gt_n_unique_frames",
        "overlap_n_frames",
        "gt_coverage_over_pred",
        "missing_gt_for_pred_count",
        "missing_pred_for_gt_count",
        "missing_gt_tail_only",
    ]
    print(out_df[display_cols].to_string(index=False))

    if args.show_segments:
        for _, row in out_df.iterrows():
            print(
                f"[{row['dataset_key']}] missing_gt_segments={row['missing_gt_segments']} "
                f"missing_pred_segments={row['missing_pred_segments']}"
            )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")

    summary = {
        "manifest_csv": str(manifest_csv),
        "n_datasets": int(len(out_df)),
        "n_with_missing_gt_for_pred": int((out_df["missing_gt_for_pred_count"] > 0).sum()),
        "n_with_missing_pred_for_gt": int((out_df["missing_pred_for_gt_count"] > 0).sum()),
        "n_tail_only_missing_gt": int((out_df["missing_gt_tail_only"] == True).sum()),
        "total_missing_gt_for_pred": int(out_df["missing_gt_for_pred_count"].sum()),
        "total_missing_pred_for_gt": int(out_df["missing_pred_for_gt_count"].sum()),
        "macro_gt_coverage_over_pred": float(out_df["gt_coverage_over_pred"].mean()),
        "frame_weighted_gt_coverage_over_pred": float(
            (out_df["overlap_n_frames"].sum() / out_df["pred_n_unique_frames"].sum())
            if out_df["pred_n_unique_frames"].sum() > 0
            else 0.0
        ),
        "datasets_non_tail_missing_gt": out_df[
            (out_df["missing_gt_for_pred_count"] > 0) & (out_df["missing_gt_tail_only"] == False)
        ]["dataset_key"].tolist(),
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {summary_json}")

    violations: List[str] = []
    if args.min_gt_coverage_over_pred is not None:
        failing = out_df[out_df["gt_coverage_over_pred"] < float(args.min_gt_coverage_over_pred)]
        if not failing.empty:
            violations.append(
                "coverage_violation: "
                + ", ".join(
                    f"{row.dataset_key}({row.gt_coverage_over_pred:.4f})" for row in failing.itertuples(index=False)
                )
            )
    if args.require_missing_gt_tail_only:
        failing = out_df[(out_df["missing_gt_for_pred_count"] > 0) & (out_df["missing_gt_tail_only"] == False)]
        if not failing.empty:
            violations.append("non_tail_missing_gt: " + ", ".join(failing["dataset_key"].tolist()))

    if violations:
        for message in violations:
            print(f"FAIL: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
