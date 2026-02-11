from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timestamp_supervision_extraction.evaluate_selected_timestamps import (  # noqa: E402
    load_gt_binary_map,
    load_selected_csv,
)


REQUIRED_METRICS_COLUMNS = ("dataset_key", "status", "selected_csv", "gt_csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze timestamp-supervision evaluation artifacts: per-dataset diagnostics, "
            "frame-level mismatches, and prioritized datasets for follow-up."
        )
    )
    parser.add_argument(
        "--metrics-csv",
        required=True,
        help="Path to selected_timestamp_metrics_per_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write analysis outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-priority datasets to print to stdout.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_columns(df: pd.DataFrame, required: Tuple[str, ...], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_metrics_csv(path: Path | str) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    df = pd.read_csv(csv_path)
    _require_columns(df, REQUIRED_METRICS_COLUMNS, csv_path)
    out = df.copy()
    out["dataset_key"] = out["dataset_key"].astype(str)
    out["status"] = out["status"].astype(str)
    return out


def _selected_class_balance(selected_csv: Path | str) -> Dict[str, Any]:
    csv_path = Path(selected_csv).expanduser()
    if not csv_path.exists():
        return {
            "selected_csv_exists": False,
            "selected_from_csv_total": 0,
            "selected_from_csv_pred_pos": 0,
            "selected_from_csv_pred_neg": 0,
            "selected_csv_error": f"missing selected CSV: {csv_path}",
        }
    try:
        df, _ = load_selected_csv(csv_path, strict_duplicates=False)
    except Exception as exc:
        return {
            "selected_csv_exists": True,
            "selected_from_csv_total": 0,
            "selected_from_csv_pred_pos": 0,
            "selected_from_csv_pred_neg": 0,
            "selected_csv_error": str(exc),
        }

    pred_pos = int((df["predicted_label"] == 1).sum()) if not df.empty else 0
    pred_neg = int((df["predicted_label"] == 0).sum()) if not df.empty else 0
    return {
        "selected_csv_exists": True,
        "selected_from_csv_total": int(len(df)),
        "selected_from_csv_pred_pos": pred_pos,
        "selected_from_csv_pred_neg": pred_neg,
        "selected_csv_error": "",
    }


def build_dataset_diagnostic_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        dataset_key = str(row["dataset_key"])
        status = str(row["status"])
        error = str(row.get("error", "") if row.get("error", "") == row.get("error", "") else "")
        selected_csv = str(row.get("selected_csv", ""))
        gt_csv = str(row.get("gt_csv", ""))

        n_selected_total = _safe_int(row.get("n_selected_total"))
        n_selected_with_gt = _safe_int(row.get("n_selected_with_gt"))
        n_selected_missing_gt = _safe_int(row.get("n_selected_missing_gt"))
        n_gt_total = _safe_int(row.get("n_gt_total"))

        tp = _safe_int(row.get("tp"))
        fp = _safe_int(row.get("fp"))
        tn = _safe_int(row.get("tn"))
        fn = _safe_int(row.get("fn"))

        accuracy = _safe_float(row.get("accuracy"))
        coverage_selected_over_gt = _safe_float(row.get("coverage_selected_over_gt"))
        coverage_aligned_over_selected = _safe_float(row.get("coverage_aligned_over_selected"))

        selected_balance = _selected_class_balance(selected_csv)
        selected_pred_pos = int(selected_balance["selected_from_csv_pred_pos"])
        selected_pred_neg = int(selected_balance["selected_from_csv_pred_neg"])

        aligned = n_selected_with_gt
        mismatch_count = fp + fn
        pred_pos_aligned = tp + fp
        pred_neg_aligned = tn + fn
        gt_pos_aligned = tp + fn
        gt_neg_aligned = tn + fp

        fp_rate_among_pred_pos = (
            float(fp / pred_pos_aligned) if pred_pos_aligned > 0 else None
        )
        fn_rate_among_gt_pos = (
            float(fn / gt_pos_aligned) if gt_pos_aligned > 0 else None
        )
        selected_density = (
            float(n_selected_with_gt / n_gt_total) if n_gt_total > 0 else None
        )
        aligned_fraction_of_selected = (
            float(n_selected_with_gt / n_selected_total) if n_selected_total > 0 else None
        )
        pred_pos_rate_selected = (
            float(selected_pred_pos / (selected_pred_pos + selected_pred_neg))
            if (selected_pred_pos + selected_pred_neg) > 0
            else None
        )
        gt_pos_rate_aligned = (
            float(gt_pos_aligned / aligned) if aligned > 0 else None
        )

        rows.append(
            {
                "dataset_key": dataset_key,
                "status": status,
                "error": error,
                "selected_csv": selected_csv,
                "gt_csv": gt_csv,
                "n_selected_total": n_selected_total,
                "n_selected_with_gt": n_selected_with_gt,
                "n_selected_missing_gt": n_selected_missing_gt,
                "n_gt_total": n_gt_total,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "mismatch_count": mismatch_count,
                "accuracy": accuracy,
                "coverage_selected_over_gt": coverage_selected_over_gt,
                "coverage_aligned_over_selected": coverage_aligned_over_selected,
                "selected_density": selected_density,
                "aligned_fraction_of_selected": aligned_fraction_of_selected,
                "selected_csv_exists": bool(selected_balance["selected_csv_exists"]),
                "selected_csv_error": str(selected_balance["selected_csv_error"]),
                "selected_from_csv_total": int(selected_balance["selected_from_csv_total"]),
                "selected_from_csv_pred_pos": selected_pred_pos,
                "selected_from_csv_pred_neg": selected_pred_neg,
                "pred_pos_rate_selected": pred_pos_rate_selected,
                "pred_pos_aligned": pred_pos_aligned,
                "pred_neg_aligned": pred_neg_aligned,
                "gt_pos_aligned": gt_pos_aligned,
                "gt_neg_aligned": gt_neg_aligned,
                "gt_pos_rate_aligned": gt_pos_rate_aligned,
                "fp_rate_among_pred_pos": fp_rate_among_pred_pos,
                "fn_rate_among_gt_pos": fn_rate_among_gt_pos,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["status", "dataset_key"], kind="mergesort").reset_index(drop=True)
    return out


def build_frame_error_tables(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mismatches: List[Dict[str, Any]] = []
    missing_gt_rows: List[Dict[str, Any]] = []

    for _, row in metrics_df.iterrows():
        dataset_key = str(row["dataset_key"])
        status = str(row["status"])
        if status != "success":
            continue

        selected_csv = Path(str(row["selected_csv"])).expanduser()
        gt_csv = Path(str(row["gt_csv"])).expanduser()

        try:
            selected_df, _ = load_selected_csv(selected_csv, strict_duplicates=False)
            gt_map = load_gt_binary_map(gt_csv)
        except Exception as exc:
            missing_gt_rows.append(
                {
                    "dataset_key": dataset_key,
                    "frame_id": None,
                    "predicted_label": None,
                    "reason": f"failed_to_load_selected_or_gt: {exc}",
                    "selected_csv": str(selected_csv),
                    "gt_csv": str(gt_csv),
                }
            )
            continue

        for _, sel_row in selected_df.iterrows():
            frame_id = int(sel_row["frame_id"])
            pred = int(sel_row["predicted_label"])
            gt = gt_map.get(frame_id)
            if gt is None:
                missing_gt_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "frame_id": frame_id,
                        "predicted_label": pred,
                        "reason": "missing_gt_for_selected_frame",
                        "selected_csv": str(selected_csv),
                        "gt_csv": str(gt_csv),
                    }
                )
                continue

            if pred != gt:
                error_type = "fp" if pred == 1 else "fn"
                mismatches.append(
                    {
                        "dataset_key": dataset_key,
                        "frame_id": frame_id,
                        "predicted_label": pred,
                        "gt_label": int(gt),
                        "error_type": error_type,
                        "selected_csv": str(selected_csv),
                        "gt_csv": str(gt_csv),
                    }
                )

    mismatch_df = (
        pd.DataFrame(mismatches)
        if mismatches
        else pd.DataFrame(
            columns=[
                "dataset_key",
                "frame_id",
                "predicted_label",
                "gt_label",
                "error_type",
                "selected_csv",
                "gt_csv",
            ]
        )
    )
    missing_gt_df = (
        pd.DataFrame(missing_gt_rows)
        if missing_gt_rows
        else pd.DataFrame(
            columns=[
                "dataset_key",
                "frame_id",
                "predicted_label",
                "reason",
                "selected_csv",
                "gt_csv",
            ]
        )
    )

    if not mismatch_df.empty:
        mismatch_df = mismatch_df.sort_values(
            by=["dataset_key", "error_type", "frame_id"], kind="mergesort"
        ).reset_index(drop=True)
    if not missing_gt_df.empty:
        missing_gt_df = missing_gt_df.sort_values(
            by=["dataset_key", "frame_id"], kind="mergesort"
        ).reset_index(drop=True)
    return mismatch_df, missing_gt_df


def build_priority_datasets(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in diagnostic_df.iterrows():
        status = str(row["status"])
        accuracy = _safe_float(row.get("accuracy"))
        mismatch_count = _safe_int(row.get("mismatch_count"))
        n_selected_with_gt = _safe_int(row.get("n_selected_with_gt"))
        coverage = _safe_float(row.get("coverage_aligned_over_selected"))
        fp = _safe_int(row.get("fp"))
        fn = _safe_int(row.get("fn"))

        failure_penalty = 1000.0 if status != "success" else 0.0
        accuracy_penalty = (1.0 - accuracy) * 50.0 if accuracy is not None else 50.0
        mismatch_penalty = float((fp + fn) * 10)
        low_sample_penalty = 20.0 if n_selected_with_gt < 5 else 0.0
        coverage_penalty = (1.0 - coverage) * 20.0 if coverage is not None else 0.0
        priority_score = (
            failure_penalty
            + accuracy_penalty
            + mismatch_penalty
            + low_sample_penalty
            + coverage_penalty
        )

        rows.append(
            {
                "dataset_key": str(row["dataset_key"]),
                "status": status,
                "priority_score": float(priority_score),
                "accuracy": accuracy,
                "mismatch_count": mismatch_count,
                "fp": fp,
                "fn": fn,
                "n_selected_with_gt": n_selected_with_gt,
                "coverage_aligned_over_selected": coverage,
                "error": str(row.get("error", "")),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["priority_score", "dataset_key"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    out["priority_rank"] = out.index + 1
    cols = [
        "priority_rank",
        "dataset_key",
        "status",
        "priority_score",
        "accuracy",
        "mismatch_count",
        "fp",
        "fn",
        "n_selected_with_gt",
        "coverage_aligned_over_selected",
        "error",
    ]
    return out[cols]


def build_selection_density_report(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_key",
        "status",
        "n_selected_total",
        "n_selected_with_gt",
        "n_selected_missing_gt",
        "n_gt_total",
        "selected_density",
        "coverage_selected_over_gt",
        "coverage_aligned_over_selected",
        "selected_from_csv_pred_pos",
        "selected_from_csv_pred_neg",
        "pred_pos_rate_selected",
        "gt_pos_rate_aligned",
    ]
    return diagnostic_df[cols].copy()


def _summary_payload(
    metrics_csv: Path,
    diagnostic_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    missing_gt_df: pd.DataFrame,
) -> Dict[str, Any]:
    success_df = diagnostic_df[diagnostic_df["status"] == "success"]
    failed_df = diagnostic_df[diagnostic_df["status"] != "success"]
    return {
        "generated_at_utc": _utc_now(),
        "metrics_csv": str(metrics_csv),
        "n_datasets_total": int(len(diagnostic_df)),
        "n_datasets_success": int(len(success_df)),
        "n_datasets_failed": int(len(failed_df)),
        "n_frame_mismatches": int(len(mismatch_df)),
        "n_selected_missing_gt_rows": int(len(missing_gt_df)),
        "fp_total": int(success_df["fp"].fillna(0).astype(int).sum()) if not success_df.empty else 0,
        "fn_total": int(success_df["fn"].fillna(0).astype(int).sum()) if not success_df.empty else 0,
        "mean_accuracy_success": (
            float(success_df["accuracy"].astype(float).mean()) if not success_df.empty else None
        ),
    }


def run(args: argparse.Namespace) -> int:
    metrics_csv = Path(args.metrics_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics_csv(metrics_csv)
    diagnostic_df = build_dataset_diagnostic_table(metrics_df)
    mismatch_df, missing_gt_df = build_frame_error_tables(metrics_df)
    priority_df = build_priority_datasets(diagnostic_df)
    density_df = build_selection_density_report(diagnostic_df)

    diagnostic_path = output_dir / "dataset_diagnostic_table.csv"
    mismatch_path = output_dir / "frame_level_mismatches.csv"
    missing_gt_path = output_dir / "frame_level_missing_gt.csv"
    priority_path = output_dir / "priority_datasets.csv"
    density_path = output_dir / "selection_density_report.csv"
    summary_path = output_dir / "analysis_summary.json"

    diagnostic_df.to_csv(diagnostic_path, index=False)
    mismatch_df.to_csv(mismatch_path, index=False)
    missing_gt_df.to_csv(missing_gt_path, index=False)
    priority_df.to_csv(priority_path, index=False)
    density_df.to_csv(density_path, index=False)
    summary_path.write_text(
        json.dumps(
            _summary_payload(
                metrics_csv=metrics_csv,
                diagnostic_df=diagnostic_df,
                mismatch_df=mismatch_df,
                missing_gt_df=missing_gt_df,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    top_k = max(1, int(args.top_k))
    top = priority_df.head(top_k)
    print(f"analysis_output_dir: {output_dir}")
    print(f"dataset_diagnostic_table: {diagnostic_path}")
    print(f"frame_level_mismatches: {mismatch_path}")
    print(f"frame_level_missing_gt: {missing_gt_path}")
    print(f"priority_datasets: {priority_path}")
    print(f"selection_density_report: {density_path}")
    print(f"analysis_summary_json: {summary_path}")
    print("top_priority_datasets:")
    for _, row in top.iterrows():
        print(
            f"  rank={int(row['priority_rank'])} dataset={row['dataset_key']} "
            f"status={row['status']} score={float(row['priority_score']):.2f}"
        )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"timestamp error analysis failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
