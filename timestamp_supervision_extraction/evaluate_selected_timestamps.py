from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.metrics import confusion_counts, frame_accuracy


REQUIRED_SELECTED_COLUMNS = ("frame_id", "predicted_label")
REQUIRED_MANIFEST_COLUMNS = ("dataset_key", "selected_csv", "gt_csv")


@dataclass(frozen=True)
class DatasetEvaluation:
    dataset_key: str
    metrics: Dict[str, Any]
    pred_array: np.ndarray
    gt_array: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected timestamp predictions against GT labels in single- or "
            "multi-dataset mode."
        )
    )
    parser.add_argument(
        "--selected-csv",
        default=None,
        help="Single-mode path to selected_timestamps.csv (requires --gt-csv).",
    )
    parser.add_argument(
        "--gt-csv",
        default=None,
        help="Single-mode GT CSV path (frame_number/gt_binary or frame_id/label).",
    )
    parser.add_argument(
        "--dataset-key",
        default="single_dataset",
        help="Single-mode dataset key used in output.",
    )
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help=(
            "Multi-mode manifest CSV with required columns: dataset_key, selected_csv, gt_csv."
        ),
    )
    parser.add_argument(
        "--dataset-keys",
        default=None,
        help="Optional comma-separated dataset keys to evaluate in multi mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Multi-mode output directory for CSV/JSON outputs. "
            "Defaults to <manifest_parent>/timestamp_metrics."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional single-mode JSON output path.",
    )
    parser.add_argument(
        "--strict-duplicates",
        action="store_true",
        help="Treat duplicate frame_id rows in selected CSV as errors.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dataset_keys(keys_csv: str | None) -> set[str] | None:
    if not keys_csv:
        return None
    out = {item.strip() for item in keys_csv.split(",") if item.strip()}
    return out if out else None


def _validate_mode_args(args: argparse.Namespace) -> None:
    in_single_mode = bool(args.selected_csv or args.gt_csv)
    in_multi_mode = bool(args.manifest_csv)

    if in_single_mode and in_multi_mode:
        raise ValueError("Use either single mode (--selected-csv/--gt-csv) or multi mode (--manifest-csv), not both.")
    if not in_single_mode and not in_multi_mode:
        raise ValueError("Provide either --manifest-csv or both --selected-csv and --gt-csv.")

    if in_single_mode and (not args.selected_csv or not args.gt_csv):
        raise ValueError("Single mode requires both --selected-csv and --gt-csv.")
    if in_single_mode and args.output_dir:
        raise ValueError("--output-dir is only supported in multi mode.")
    if in_single_mode and args.dataset_keys:
        raise ValueError("--dataset-keys is only supported in multi mode.")
    if in_multi_mode and args.json_out:
        raise ValueError("--json-out is only supported in single mode.")
    if in_multi_mode and (args.selected_csv or args.gt_csv):
        raise ValueError("Multi mode does not accept --selected-csv or --gt-csv.")


def _require_columns(df: pd.DataFrame, required: Sequence[str], csv_path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def _coerce_int_series(series: pd.Series, csv_path: Path, column_name: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    invalid_mask = coerced.isna()
    if invalid_mask.any():
        bad = series[invalid_mask].head(5).tolist()
        raise ValueError(
            f"{csv_path} contains non-integer values in '{column_name}'. Examples: {bad}"
        )
    return coerced.astype(np.int64)


def _extract_frame_number(frame_id: object) -> int | None:
    if frame_id is None or (isinstance(frame_id, float) and np.isnan(frame_id)):
        return None
    match = re.search(r"(\d+)", str(frame_id))
    if not match:
        return None
    return int(match.group(1))


def _label_to_binary(label: object) -> int:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return 0
    text = str(label).strip().lower()
    if text in {"holding", "portable object", "portable object contact", "1", "true", "yes"}:
        return 1
    if text in {"not_holding", "no contact", "stationary object", "stationary object contact", "0", "false", "no"}:
        return 0
    return 0


def load_selected_csv(path: Path | str, strict_duplicates: bool = False) -> tuple[pd.DataFrame, Dict[str, int]]:
    csv_path = Path(path).expanduser()
    df = pd.read_csv(csv_path)
    _require_columns(df, REQUIRED_SELECTED_COLUMNS, csv_path)

    selected = df[list(REQUIRED_SELECTED_COLUMNS)].copy()
    selected["frame_id"] = _coerce_int_series(selected["frame_id"], csv_path, "frame_id")
    selected["predicted_label"] = _coerce_int_series(selected["predicted_label"], csv_path, "predicted_label")

    bad_mask = ~selected["predicted_label"].isin((0, 1))
    if bad_mask.any():
        bad = selected.loc[bad_mask, "predicted_label"].head(5).tolist()
        raise ValueError(
            f"{csv_path} contains invalid predicted_label values. Expected {{0,1}}; examples: {bad}"
        )

    n_selected_total = int(len(selected))
    duplicated_mask = selected.duplicated(subset=["frame_id"], keep="first")
    n_duplicates = int(duplicated_mask.sum())
    if strict_duplicates and n_duplicates > 0:
        duplicate_frames = selected.loc[duplicated_mask, "frame_id"].head(10).tolist()
        raise ValueError(
            f"{csv_path} has duplicate frame_id values in selected rows. "
            f"Examples: {duplicate_frames}"
        )
    if n_duplicates > 0:
        selected = selected.loc[~duplicated_mask].reset_index(drop=True)

    stats = {
        "n_selected_total": n_selected_total,
        "n_selected_unique": int(len(selected)),
        "n_selected_duplicates_dropped": n_duplicates,
    }
    return selected, stats


def load_gt_binary_map(path: Path | str) -> Dict[int, int]:
    csv_path = Path(path).expanduser()
    gt_df = pd.read_csv(csv_path, sep=None, engine="python")

    if "frame_number" in gt_df.columns and "gt_binary" in gt_df.columns:
        out: Dict[int, int] = {}
        for _, row in gt_df.iterrows():
            frame_number = int(row["frame_number"])
            value = int(row["gt_binary"])
            if value not in (0, 1):
                raise ValueError(f"{csv_path} has non-binary gt_binary at frame_number={frame_number}: {value}")
            out[frame_number] = value
        return out

    if "frame_id" in gt_df.columns and "label" in gt_df.columns:
        out = {}
        for _, row in gt_df.iterrows():
            frame_number = _extract_frame_number(row["frame_id"])
            if frame_number is None:
                continue
            out[frame_number] = _label_to_binary(row["label"])
        if not out:
            raise ValueError(
                f"{csv_path} did not produce any GT rows after parsing frame_id values."
            )
        return out

    raise ValueError(
        f"{csv_path} must contain either (frame_number,gt_binary) or (frame_id,label)."
    )


def align_selected_with_gt(
    selected_df: pd.DataFrame,
    gt_map: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    pred_values: List[int] = []
    gt_values: List[int] = []
    missing = 0

    for _, row in selected_df.iterrows():
        frame_id = int(row["frame_id"])
        gt_value = gt_map.get(frame_id)
        if gt_value is None:
            missing += 1
            continue
        pred_values.append(int(row["predicted_label"]))
        gt_values.append(int(gt_value))

    pred_array = np.asarray(pred_values, dtype=int)
    gt_array = np.asarray(gt_values, dtype=int)
    align_stats = {
        "n_selected_with_gt": int(len(pred_array)),
        "n_selected_missing_gt": int(missing),
        "n_gt_total": int(len(gt_map)),
    }
    return pred_array, gt_array, align_stats


def _binary_prf_from_confusion(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_selected_metrics(
    *,
    dataset_key: str,
    pred_array: np.ndarray,
    gt_array: np.ndarray,
    n_selected_total: int,
    n_selected_unique: int,
    n_selected_duplicates_dropped: int,
    n_selected_with_gt: int,
    n_selected_missing_gt: int,
    n_gt_total: int,
) -> Dict[str, Any]:
    if pred_array.shape != gt_array.shape:
        raise ValueError("pred and gt arrays must have identical shape.")
    if pred_array.size == 0:
        raise ValueError("No aligned selected frames found in GT for evaluation.")

    accuracy = float(frame_accuracy(pred_array, gt_array))
    tp, fp, tn, fn = confusion_counts(pred_array, gt_array)

    pos_scores = _binary_prf_from_confusion(tp=tp, fp=fp, fn=fn)
    neg_scores = _binary_prf_from_confusion(tp=tn, fp=fn, fn=fp)
    macro_f1 = float((pos_scores["f1"] + neg_scores["f1"]) / 2.0)

    support_pos = int((gt_array == 1).sum())
    support_neg = int((gt_array == 0).sum())

    coverage_selected_over_gt = (
        float(n_selected_with_gt / n_gt_total) if n_gt_total > 0 else 0.0
    )
    coverage_aligned_over_selected = (
        float(n_selected_with_gt / n_selected_total) if n_selected_total > 0 else 0.0
    )

    return {
        "dataset_key": dataset_key,
        "status": "success",
        "error": "",
        "n_selected_total": int(n_selected_total),
        "n_selected_unique": int(n_selected_unique),
        "n_selected_duplicates_dropped": int(n_selected_duplicates_dropped),
        "n_selected_with_gt": int(n_selected_with_gt),
        "n_selected_missing_gt": int(n_selected_missing_gt),
        "n_gt_total": int(n_gt_total),
        "coverage_selected_over_gt": coverage_selected_over_gt,
        "coverage_aligned_over_selected": coverage_aligned_over_selected,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "positive_precision": pos_scores["precision"],
        "positive_recall": pos_scores["recall"],
        "positive_f1": pos_scores["f1"],
        "positive_support": support_pos,
        "negative_precision": neg_scores["precision"],
        "negative_recall": neg_scores["recall"],
        "negative_f1": neg_scores["f1"],
        "negative_support": support_neg,
        "macro_f1": macro_f1,
    }


def evaluate_single_dataset(
    *,
    dataset_key: str,
    selected_csv: Path | str,
    gt_csv: Path | str,
    strict_duplicates: bool = False,
) -> DatasetEvaluation:
    selected_df, selected_stats = load_selected_csv(selected_csv, strict_duplicates=strict_duplicates)
    gt_map = load_gt_binary_map(gt_csv)
    pred_array, gt_array, align_stats = align_selected_with_gt(selected_df, gt_map)
    metrics = compute_selected_metrics(
        dataset_key=dataset_key,
        pred_array=pred_array,
        gt_array=gt_array,
        n_selected_total=selected_stats["n_selected_total"],
        n_selected_unique=selected_stats["n_selected_unique"],
        n_selected_duplicates_dropped=selected_stats["n_selected_duplicates_dropped"],
        n_selected_with_gt=align_stats["n_selected_with_gt"],
        n_selected_missing_gt=align_stats["n_selected_missing_gt"],
        n_gt_total=align_stats["n_gt_total"],
    )
    return DatasetEvaluation(
        dataset_key=dataset_key,
        metrics=metrics,
        pred_array=pred_array,
        gt_array=gt_array,
    )


def _resolve_manifest_path(base_dir: Path, raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_manifest_csv(path: Path | str, dataset_keys: set[str] | None = None) -> pd.DataFrame:
    manifest_path = Path(path).expanduser()
    df = pd.read_csv(manifest_path)
    _require_columns(df, REQUIRED_MANIFEST_COLUMNS, manifest_path)

    manifest = df[list(REQUIRED_MANIFEST_COLUMNS)].copy()
    manifest["dataset_key"] = manifest["dataset_key"].astype(str)
    if dataset_keys:
        manifest = manifest[manifest["dataset_key"].isin(dataset_keys)].reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No datasets remain after manifest loading/filtering.")
    return manifest


def _global_micro_metrics(
    evaluations: Sequence[DatasetEvaluation],
) -> Dict[str, Any]:
    pred_all = np.concatenate([item.pred_array for item in evaluations])
    gt_all = np.concatenate([item.gt_array for item in evaluations])

    total_selected = int(sum(item.metrics["n_selected_total"] for item in evaluations))
    total_unique = int(sum(item.metrics["n_selected_unique"] for item in evaluations))
    total_dropped = int(sum(item.metrics["n_selected_duplicates_dropped"] for item in evaluations))
    total_with_gt = int(sum(item.metrics["n_selected_with_gt"] for item in evaluations))
    total_missing = int(sum(item.metrics["n_selected_missing_gt"] for item in evaluations))
    total_gt = int(sum(item.metrics["n_gt_total"] for item in evaluations))

    return compute_selected_metrics(
        dataset_key="__global_micro__",
        pred_array=pred_all,
        gt_array=gt_all,
        n_selected_total=total_selected,
        n_selected_unique=total_unique,
        n_selected_duplicates_dropped=total_dropped,
        n_selected_with_gt=total_with_gt,
        n_selected_missing_gt=total_missing,
        n_gt_total=total_gt,
    )


def _global_macro_metrics(
    per_dataset_metrics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    scalar_fields = [
        "coverage_selected_over_gt",
        "coverage_aligned_over_selected",
        "accuracy",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "negative_precision",
        "negative_recall",
        "negative_f1",
        "macro_f1",
    ]
    out: Dict[str, Any] = {
        "dataset_key": "__global_macro__",
        "status": "success",
        "error": "",
    }
    for key in scalar_fields:
        out[key] = float(np.mean([float(item[key]) for item in per_dataset_metrics]))

    count_fields = [
        "n_selected_total",
        "n_selected_unique",
        "n_selected_duplicates_dropped",
        "n_selected_with_gt",
        "n_selected_missing_gt",
        "n_gt_total",
        "tp",
        "fp",
        "tn",
        "fn",
        "positive_support",
        "negative_support",
    ]
    for key in count_fields:
        out[key] = int(sum(int(item[key]) for item in per_dataset_metrics))
    return out


def evaluate_manifest(
    *,
    manifest_csv: Path | str,
    dataset_keys: set[str] | None = None,
    strict_duplicates: bool = False,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_csv).expanduser()
    manifest_df = load_manifest_csv(manifest_path, dataset_keys=dataset_keys)

    base_dir = manifest_path.parent
    rows: List[Dict[str, Any]] = []
    successes: List[DatasetEvaluation] = []

    for _, row in manifest_df.iterrows():
        dataset_key = str(row["dataset_key"])
        selected_csv = _resolve_manifest_path(base_dir, row["selected_csv"])
        gt_csv = _resolve_manifest_path(base_dir, row["gt_csv"])
        try:
            evaluation = evaluate_single_dataset(
                dataset_key=dataset_key,
                selected_csv=selected_csv,
                gt_csv=gt_csv,
                strict_duplicates=strict_duplicates,
            )
            success_row = dict(evaluation.metrics)
            success_row["selected_csv"] = str(selected_csv)
            success_row["gt_csv"] = str(gt_csv)
            rows.append(success_row)
            successes.append(evaluation)
        except Exception as exc:
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "selected_csv": str(selected_csv),
                    "gt_csv": str(gt_csv),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    success_metrics = [row for row in rows if row.get("status") == "success"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]

    if successes:
        global_micro = _global_micro_metrics(successes)
        global_macro = _global_macro_metrics(success_metrics)
    else:
        global_micro = None
        global_macro = None

    return {
        "generated_at_utc": _utc_now(),
        "manifest_csv": str(manifest_path),
        "n_datasets_input": int(len(manifest_df)),
        "n_datasets_success": int(len(success_metrics)),
        "n_datasets_failed": int(len(failed_rows)),
        "rows": rows,
        "global_micro": global_micro,
        "global_macro": global_macro,
    }


def _single_mode_result_payload(evaluation: DatasetEvaluation) -> Dict[str, Any]:
    return {
        "generated_at_utc": _utc_now(),
        "dataset_key": evaluation.dataset_key,
        "metrics": evaluation.metrics,
    }


def _print_single_summary(metrics: Mapping[str, Any]) -> None:
    print(f"dataset_key: {metrics['dataset_key']}")
    print(
        "selected_total={0} selected_with_gt={1} missing_gt={2} gt_total={3}".format(
            metrics["n_selected_total"],
            metrics["n_selected_with_gt"],
            metrics["n_selected_missing_gt"],
            metrics["n_gt_total"],
        )
    )
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(
        "positive: precision={0:.4f} recall={1:.4f} f1={2:.4f}".format(
            metrics["positive_precision"],
            metrics["positive_recall"],
            metrics["positive_f1"],
        )
    )
    print(
        "negative: precision={0:.4f} recall={1:.4f} f1={2:.4f}".format(
            metrics["negative_precision"],
            metrics["negative_recall"],
            metrics["negative_f1"],
        )
    )
    print(
        "confusion: tp={0} fp={1} tn={2} fn={3}".format(
            metrics["tp"],
            metrics["fp"],
            metrics["tn"],
            metrics["fn"],
        )
    )


def _print_multi_summary(results: Mapping[str, Any]) -> None:
    print(f"datasets_input: {results['n_datasets_input']}")
    print(f"datasets_success: {results['n_datasets_success']}")
    print(f"datasets_failed: {results['n_datasets_failed']}")
    if results["global_micro"] is not None:
        print(f"global_micro_accuracy: {results['global_micro']['accuracy']:.4f}")
    if results["global_macro"] is not None:
        print(f"global_macro_accuracy: {results['global_macro']['accuracy']:.4f}")
    if results["n_datasets_failed"] > 0:
        for row in results["rows"]:
            if row.get("status") == "failed":
                print(f"[failed] {row['dataset_key']}: {row['error']}")


def run(args: argparse.Namespace) -> int:
    _validate_mode_args(args)

    if args.manifest_csv:
        keys = _parse_dataset_keys(args.dataset_keys)
        results = evaluate_manifest(
            manifest_csv=args.manifest_csv,
            dataset_keys=keys,
            strict_duplicates=bool(args.strict_duplicates),
        )
        manifest_path = Path(args.manifest_csv).expanduser()
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else (manifest_path.parent / "timestamp_metrics")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        per_dataset_csv = output_dir / "selected_timestamp_metrics_per_dataset.csv"
        summary_json = output_dir / "selected_timestamp_metrics_summary.json"

        rows_df = pd.DataFrame(results["rows"])
        rows_df = rows_df.sort_values(by=["dataset_key"], kind="mergesort").reset_index(drop=True)
        rows_df.to_csv(per_dataset_csv, index=False)
        summary_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

        _print_multi_summary(results)
        print(f"per_dataset_csv: {per_dataset_csv}")
        print(f"summary_json: {summary_json}")
        return 1 if results["n_datasets_failed"] > 0 else 0

    evaluation = evaluate_single_dataset(
        dataset_key=str(args.dataset_key),
        selected_csv=str(args.selected_csv),
        gt_csv=str(args.gt_csv),
        strict_duplicates=bool(args.strict_duplicates),
    )
    _print_single_summary(evaluation.metrics)
    payload = _single_mode_result_payload(evaluation)
    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"json_out: {json_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"timestamp selected metrics failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
