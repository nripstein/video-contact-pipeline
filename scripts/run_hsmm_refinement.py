from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_refinement import HSMMKSegmentsConfig, HSMMKSegmentsRefiner
from binary_refinement.evaluator import (
    save_confidence_refined_gt_barcode,
    save_original_refined_gt_barcode,
)
from pipeline.metrics import confusion_counts, edit_score, f_score, frame_accuracy
from pipeline.visualization import load_gt_binary_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run k-constrained left-to-right HSMM refinement on detections_condensed.csv "
            "without any GT fitting."
        )
    )

    parser.add_argument("--condensed-csv", default=None, help="Path to detections_condensed.csv")
    parser.add_argument("--run-root", default=None, help="Run root containing predictions/<dataset_key>/")
    parser.add_argument("--dataset-key", default=None, help="Dataset key under predictions/ (e.g., sr1)")
    parser.add_argument(
        "--gt-csv",
        default=None,
        help=(
            "Optional GT CSV path. If omitted and --run-root/--dataset-key is used, GT is resolved from "
            "run_manifest.csv."
        ),
    )

    parser.add_argument("--frame-column", default="frame_number", help="Frame index column name")
    parser.add_argument("--label-column", default="contact_label", help="Label column name")
    parser.add_argument(
        "--positive-labels",
        default="portable object,portable object contact,holding,1,true,yes",
        help="Comma-separated label values mapped to binary 1",
    )

    parser.add_argument("--k-segments", type=int, required=True, help="Exact number of HSMM segments")
    parser.add_argument("--alpha-during-trial", type=float, required=True, help="Gamma alpha for state=1")
    parser.add_argument("--lambda-during-trial", type=float, required=True, help="Gamma lambda for state=1")
    parser.add_argument("--alpha-between-trials", type=float, required=True, help="Gamma alpha for state=0")
    parser.add_argument("--lambda-between-trials", type=float, required=True, help="Gamma lambda for state=0")
    parser.add_argument("--fpr", type=float, default=0.1, help="False positive rate P(y=1|x=0)")
    parser.add_argument("--fnr", type=float, default=0.1, help="False negative rate P(y=0|x=1)")
    parser.add_argument("--start-state", type=int, default=0, choices=[0, 1], help="Initial hidden state")
    parser.add_argument("--duration-weight", type=float, default=1.0, help="Weight for duration log-likelihood")
    parser.add_argument("--emission-weight", type=float, default=1.0, help="Weight for emission log-likelihood")
    parser.add_argument(
        "--max-segment-frames",
        type=int,
        default=540,
        help="Hard maximum duration per segment in frames (default: 540).",
    )
    parser.add_argument(
        "--no-max-segment-cap",
        action="store_true",
        help="Disable hard max segment-length cap.",
    )
    parser.add_argument(
        "--numba-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Decoder backend mode: auto uses numba if available; on requires numba; off uses Python loops.",
    )
    parser.add_argument(
        "--return-posteriors",
        action="store_true",
        help="Compute frame-wise posterior P(contact|y) and export hsmm_posteriors.csv.",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", help="Show tqdm progress for HSMM DP decoding (default).")
    progress_group.add_argument("--no-progress", dest="progress", action="store_false", help="Disable tqdm progress output.")
    parser.set_defaults(progress=True)

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <pred_dir>/hsmm_refinement)",
    )
    parser.add_argument(
        "--refined-positive-label",
        default="Portable Object",
        help="Label to write for refined binary 1 in refined condensed CSV",
    )
    parser.add_argument(
        "--refined-negative-label",
        default="Stationary Object",
        help="Label to write for refined binary 0 in refined condensed CSV",
    )
    return parser.parse_args()


def _resolve_condensed_csv(args: argparse.Namespace) -> Path:
    has_direct = bool(args.condensed_csv)
    has_dataset = bool(args.run_root and args.dataset_key)
    if has_direct == has_dataset:
        raise ValueError("Provide exactly one of: --condensed-csv OR (--run-root and --dataset-key)")

    if has_direct:
        path = Path(args.condensed_csv).expanduser()
    else:
        path = Path(args.run_root).expanduser() / "predictions" / str(args.dataset_key) / "detections_condensed.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing condensed CSV: {path}")
    return path


def _resolve_gt_csv(args: argparse.Namespace) -> Path:
    if args.gt_csv:
        gt_path = Path(args.gt_csv).expanduser()
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT CSV: {gt_path}")
        return gt_path

    if args.run_root and args.dataset_key:
        manifest_path = Path(args.run_root).expanduser() / "run_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run manifest for GT lookup: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        rows = manifest.loc[manifest["dataset_key"].astype(str) == str(args.dataset_key)]
        if rows.empty:
            raise ValueError(f"Dataset key '{args.dataset_key}' not found in {manifest_path}")
        gt_csv = rows.iloc[0].get("gt_csv")
        if gt_csv is None or (isinstance(gt_csv, float) and np.isnan(gt_csv)):
            raise ValueError(f"No gt_csv value for dataset '{args.dataset_key}' in {manifest_path}")
        gt_path = Path(str(gt_csv)).expanduser()
        if not gt_path.exists():
            raise FileNotFoundError(f"Resolved GT CSV does not exist: {gt_path}")
        return gt_path

    raise ValueError(
        "GT is required for requested barcode format. Provide --gt-csv or use --run-root with --dataset-key."
    )


def _positive_set(raw: str) -> Set[str]:
    out = {x.strip().lower() for x in str(raw).split(",") if x.strip()}
    if not out:
        raise ValueError("--positive-labels must contain at least one non-empty value")
    return out


def _label_to_binary(label: object, positives: Set[str]) -> int:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return 0
    if isinstance(label, (int, np.integer)):
        return 1 if int(label) == 1 else 0
    if isinstance(label, float):
        return 1 if int(label) == 1 else 0
    return 1 if str(label).strip().lower() in positives else 0


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mof = float(frame_accuracy(pred, gt))
    edit = float(edit_score(pred, gt, bg_class=(0,), norm=True))
    f1 = {
        "F1@10": float(f_score(pred, gt, 0.1, bg_class=(0,))),
        "F1@25": float(f_score(pred, gt, 0.25, bg_class=(0,))),
        "F1@50": float(f_score(pred, gt, 0.5, bg_class=(0,))),
        "F1@75": float(f_score(pred, gt, 0.75, bg_class=(0,))),
    }
    tp, fp, tn, fn = confusion_counts(pred, gt)
    return {
        "MoF": mof,
        "MoF_pct": mof * 100.0,
        "Edit": edit,
        "F1": f1,
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n_frames": int(len(gt)),
    }


def main() -> int:
    args = parse_args()
    condensed_csv = _resolve_condensed_csv(args)
    gt_csv = _resolve_gt_csv(args)
    pred_dir = condensed_csv.parent
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (pred_dir / "hsmm_refinement")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(condensed_csv)
    for required_col in (args.frame_column, args.label_column):
        if required_col not in df.columns:
            raise ValueError(f"{condensed_csv} missing required column: {required_col}")

    sort_cols = [c for c in (args.frame_column, "frame_id") if c in df.columns]
    ordered = df.sort_values(by=sort_cols, kind="mergesort").drop_duplicates(
        subset=[args.frame_column], keep="first"
    )
    frame_numbers = ordered[args.frame_column].astype(int).to_numpy()

    positives = _positive_set(args.positive_labels)
    original_binary = ordered[args.label_column].map(lambda x: _label_to_binary(x, positives)).astype(int).to_numpy()
    max_segment_length_frames = None if bool(args.no_max_segment_cap) else int(args.max_segment_frames)

    cfg = HSMMKSegmentsConfig(
        k_segments=int(args.k_segments),
        alpha_non_contact=float(args.alpha_between_trials),
        lambda_non_contact=float(args.lambda_between_trials),
        alpha_contact=float(args.alpha_during_trial),
        lambda_contact=float(args.lambda_during_trial),
        fpr=float(args.fpr),
        fnr=float(args.fnr),
        start_state=int(args.start_state),
        duration_weight=float(args.duration_weight),
        emission_weight=float(args.emission_weight),
        max_segment_length_frames=max_segment_length_frames,
        numba_mode=str(args.numba_mode),
    )
    progress_desc = f"hsmm:{args.dataset_key}" if args.dataset_key else "hsmm"
    result = HSMMKSegmentsRefiner(cfg).predict(
        original_binary,
        progress=args.progress,
        progress_desc=progress_desc,
        return_posteriors=bool(args.return_posteriors),
    )
    gt_binary = load_gt_binary_from_csv(str(gt_csv), frame_numbers.tolist()).astype(int)
    original_metrics = _compute_metrics(original_binary, gt_binary)
    refined_metrics = _compute_metrics(result.sequence.astype(int), gt_binary)

    refined_binary = result.sequence.astype(int)
    refined_labels = np.where(
        refined_binary == 1,
        str(args.refined_positive_label),
        str(args.refined_negative_label),
    )

    refined_binary_csv = out_dir / "hsmm_refined_binary.csv"
    pd.DataFrame(
        {
            args.frame_column: frame_numbers,
            "pred_binary_original": original_binary,
            "pred_binary_refined": refined_binary,
        }
    ).to_csv(refined_binary_csv, index=False)

    posteriors_csv = None
    confidence_barcode_path = None
    if bool(args.return_posteriors):
        if result.posteriors is None:
            raise RuntimeError("return_posteriors was enabled, but the refiner returned no posterior vector.")
        posteriors = np.asarray(result.posteriors, dtype=float).reshape(-1)
        if posteriors.shape[0] != frame_numbers.shape[0]:
            raise RuntimeError(
                "Posterior length mismatch: "
                f"frames={frame_numbers.shape[0]} posteriors={posteriors.shape[0]}"
            )
        posteriors_csv = out_dir / "hsmm_posteriors.csv"
        pd.DataFrame(
            {
                args.frame_column: frame_numbers,
                "posterior_contact": np.clip(posteriors, 0.0, 1.0),
            }
        ).to_csv(posteriors_csv, index=False)
        confidence_barcode_path = out_dir / "barcode_confidence_refined_gt.png"
        save_confidence_refined_gt_barcode(
            confidence=posteriors,
            refined_signal=refined_binary,
            ground_truth=gt_binary,
            save_path=str(confidence_barcode_path),
            confidence_name="P(Holding)",
            refined_name="HSMM Refined",
            ground_truth_name="Ground Truth",
        )

    refined_condensed = ordered.copy()
    refined_condensed[args.label_column] = refined_labels
    refined_condensed_csv = out_dir / "detections_condensed_hsmm.csv"
    refined_condensed.to_csv(refined_condensed_csv, index=False)

    barcode_path = out_dir / "barcode_original_refined_gt.png"
    save_original_refined_gt_barcode(
        original_signal=original_binary,
        refined_signal=refined_binary,
        ground_truth=gt_binary,
        save_path=str(barcode_path),
        original_name="Original",
        refined_name="HSMM Refined",
        ground_truth_name="Ground Truth",
    )

    config_json = out_dir / "hsmm_config_used.json"
    config_json.write_text(
        json.dumps(
            {
                "condensed_csv": str(condensed_csv),
                "gt_csv": str(gt_csv),
                "k_segments": int(args.k_segments),
                "alpha_during_trial": float(args.alpha_during_trial),
                "lambda_during_trial": float(args.lambda_during_trial),
                "alpha_between_trials": float(args.alpha_between_trials),
                "lambda_between_trials": float(args.lambda_between_trials),
                "fpr": float(args.fpr),
                "fnr": float(args.fnr),
                "start_state": int(args.start_state),
                "duration_weight": float(args.duration_weight),
                "emission_weight": float(args.emission_weight),
                "max_segment_length_frames": (
                    int(max_segment_length_frames) if max_segment_length_frames is not None else None
                ),
                "numba_mode": str(args.numba_mode),
                "return_posteriors": bool(args.return_posteriors),
                "positive_labels": sorted(positives),
                "objective": float(result.objective) if result.objective is not None else None,
                "num_transitions_refined": int(result.num_transitions),
                "decoder_backend": result.metadata.get("decoder_backend"),
                "confidence_barcode_path": str(confidence_barcode_path) if confidence_barcode_path is not None else None,
            },
            indent=2,
        )
    )
    metrics_json = out_dir / "hsmm_metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "original": original_metrics,
                "refined": refined_metrics,
                "delta": {
                    "MoF": float(refined_metrics["MoF"] - original_metrics["MoF"]),
                    "Edit": float(refined_metrics["Edit"] - original_metrics["Edit"]),
                    "F1@10": float(refined_metrics["F1"]["F1@10"] - original_metrics["F1"]["F1@10"]),
                    "F1@25": float(refined_metrics["F1"]["F1@25"] - original_metrics["F1"]["F1@25"]),
                    "F1@50": float(refined_metrics["F1"]["F1@50"] - original_metrics["F1"]["F1@50"]),
                    "F1@75": float(refined_metrics["F1"]["F1@75"] - original_metrics["F1"]["F1@75"]),
                },
            },
            indent=2,
        )
    )

    print(f"Input: {condensed_csv}")
    print(f"Frames: {len(original_binary)}")
    print(f"Transitions (original): {int(np.count_nonzero(original_binary[1:] != original_binary[:-1])) if len(original_binary) > 1 else 0}")
    print(f"Transitions (refined): {int(result.num_transitions)}")
    print(
        "Original metrics: "
        f"MoF={original_metrics['MoF']:.4f} "
        f"Edit={original_metrics['Edit']:.2f} "
        f"F1@10={original_metrics['F1']['F1@10']:.2f} "
        f"F1@25={original_metrics['F1']['F1@25']:.2f} "
        f"F1@50={original_metrics['F1']['F1@50']:.2f} "
        f"F1@75={original_metrics['F1']['F1@75']:.2f}"
    )
    print(
        "Refined metrics: "
        f"MoF={refined_metrics['MoF']:.4f} "
        f"Edit={refined_metrics['Edit']:.2f} "
        f"F1@10={refined_metrics['F1']['F1@10']:.2f} "
        f"F1@25={refined_metrics['F1']['F1@25']:.2f} "
        f"F1@50={refined_metrics['F1']['F1@50']:.2f} "
        f"F1@75={refined_metrics['F1']['F1@75']:.2f}"
    )
    print(f"Output binary CSV: {refined_binary_csv}")
    if posteriors_csv is not None:
        print(f"Posterior CSV: {posteriors_csv}")
    if confidence_barcode_path is not None:
        print(f"Confidence Barcode: {confidence_barcode_path}")
    print(f"Output condensed CSV: {refined_condensed_csv}")
    print(f"Barcode: {barcode_path}")
    print(f"Config JSON: {config_json}")
    print(f"Metrics JSON: {metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
