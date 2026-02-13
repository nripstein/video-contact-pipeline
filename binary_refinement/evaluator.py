from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from binary_refinement.base import BinaryRefinementStrategy
from binary_refinement.types import EvaluationResult, RefinementResult
from pipeline.metrics import confusion_counts, edit_score, f_score, frame_accuracy
from pipeline.visualization import plot_confidence_iterative_barcodes, plot_iterative_barcodes


def _as_binary(arr: np.ndarray, name: str, allow_probabilities: bool) -> np.ndarray:
    out = np.asarray(arr).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(~np.isfinite(out)):
        raise ValueError(f"{name} must be finite")

    if np.issubdtype(out.dtype, np.bool_):
        return out.astype(int)

    float_arr = out.astype(float)
    unique = np.unique(float_arr)
    if np.all(np.isin(unique, [0.0, 1.0])):
        return float_arr.astype(int)
    if allow_probabilities and np.all((float_arr >= 0.0) & (float_arr <= 1.0)):
        return (float_arr >= 0.5).astype(int)
    raise ValueError(f"{name} must be binary values in {{0,1}}")


def evaluate_binary_predictions(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    taus: Iterable[float] = (0.1, 0.25, 0.5, 0.75),
) -> EvaluationResult:
    gt = _as_binary(ground_truth, "ground_truth", allow_probabilities=False)
    pred = _as_binary(predictions, "predictions", allow_probabilities=True)
    if gt.shape[0] != pred.shape[0]:
        raise ValueError(f"Length mismatch: ground_truth={gt.shape[0]} predictions={pred.shape[0]}")

    mof = float(frame_accuracy(pred, gt))
    edit = float(edit_score(pred, gt, bg_class=(0,), norm=True))
    f1 = {}
    for tau in taus:
        key = f"F1@{int(round(float(tau) * 100))}"
        f1[key] = float(f_score(pred, gt, float(tau), bg_class=(0,)))

    tp, fp, tn, fn = confusion_counts(pred, gt)
    confusion = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    return EvaluationResult(
        mof=mof,
        edit=edit,
        f1=f1,
        confusion=confusion,
        n_frames=int(gt.shape[0]),
    )


def save_comparison_barcode(
    ground_truth: np.ndarray,
    original_signal: np.ndarray,
    refined_signal: np.ndarray,
    save_path: str,
) -> str:
    gt = _as_binary(ground_truth, "ground_truth", allow_probabilities=False)
    orig = _as_binary(original_signal, "original_signal", allow_probabilities=True)
    ref = _as_binary(refined_signal, "refined_signal", allow_probabilities=True)

    if not (gt.shape[0] == orig.shape[0] == ref.shape[0]):
        raise ValueError(
            "Length mismatch for barcode plot: "
            f"ground_truth={gt.shape[0]} original_signal={orig.shape[0]} refined_signal={ref.shape[0]}"
        )

    return _save_binary_rows_barcode(
        signals=[gt, orig, ref],
        row_names=["Ground Truth", "Original", "Refined"],
        save_path=save_path,
    )


def save_original_vs_refined_barcode(
    original_signal: np.ndarray,
    refined_signal: np.ndarray,
    save_path: str,
    original_name: str = "Original",
    refined_name: str = "Refined",
) -> str:
    orig = _as_binary(original_signal, "original_signal", allow_probabilities=True)
    ref = _as_binary(refined_signal, "refined_signal", allow_probabilities=True)
    if orig.shape[0] != ref.shape[0]:
        raise ValueError(
            "Length mismatch for barcode plot: "
            f"original_signal={orig.shape[0]} refined_signal={ref.shape[0]}"
        )
    return _save_binary_rows_barcode(
        signals=[orig, ref],
        row_names=[str(original_name), str(refined_name)],
        save_path=save_path,
    )


def save_original_refined_gt_barcode(
    original_signal: np.ndarray,
    refined_signal: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str,
    original_name: str = "Original",
    refined_name: str = "Refined",
    ground_truth_name: str = "Ground Truth",
) -> str:
    orig = _as_binary(original_signal, "original_signal", allow_probabilities=True)
    ref = _as_binary(refined_signal, "refined_signal", allow_probabilities=True)
    gt = _as_binary(ground_truth, "ground_truth", allow_probabilities=False)
    if not (orig.shape[0] == ref.shape[0] == gt.shape[0]):
        raise ValueError(
            "Length mismatch for barcode plot: "
            f"original_signal={orig.shape[0]} refined_signal={ref.shape[0]} ground_truth={gt.shape[0]}"
        )
    return _save_binary_rows_barcode(
        signals=[orig, ref, gt],
        row_names=[str(original_name), str(refined_name), str(ground_truth_name)],
        save_path=save_path,
    )


def save_confidence_refined_gt_barcode(
    confidence: np.ndarray,
    refined_signal: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str,
    confidence_name: str = "P(Holding)",
    refined_name: str = "Prediction",
    ground_truth_name: str = "Ground Truth",
) -> str:
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    if conf.size == 0:
        raise ValueError("confidence must be non-empty")
    if np.any(~np.isfinite(conf)):
        raise ValueError("confidence must be finite")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidence must be in [0, 1]")

    ref = _as_binary(refined_signal, "refined_signal", allow_probabilities=True)
    gt = _as_binary(ground_truth, "ground_truth", allow_probabilities=False)
    if not (conf.shape[0] == ref.shape[0] == gt.shape[0]):
        raise ValueError(
            "Length mismatch for confidence barcode plot: "
            f"confidence={conf.shape[0]} refined_signal={ref.shape[0]} ground_truth={gt.shape[0]}"
        )

    out = Path(save_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_confidence_iterative_barcodes(
        confidence=conf,
        data_list=[ref, gt],
        confidence_title=str(confidence_name),
        titles=[str(refined_name), str(ground_truth_name)],
        show=False,
        save_file=str(out),
    )
    return str(out)


def _save_binary_rows_barcode(
    signals: Iterable[np.ndarray],
    row_names: Iterable[str],
    save_path: str,
) -> str:
    signal_list = [np.asarray(sig).reshape(-1).astype(int) for sig in signals]
    name_list = [str(name) for name in row_names]
    if len(signal_list) != len(name_list):
        raise ValueError("signals and row_names must have same length")
    if len(signal_list) == 0:
        raise ValueError("signals must be non-empty")

    n = int(signal_list[0].shape[0])
    for idx, sig in enumerate(signal_list):
        if sig.shape[0] != n:
            raise ValueError(
                "Length mismatch across barcode rows: "
                f"row0={n} row{idx}={sig.shape[0]}"
            )

    out = Path(save_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_iterative_barcodes(
        data_list=signal_list,
        titles=name_list,
        show=False,
        save_file=str(out),
    )
    return str(out)


def evaluate_strategy(
    strategy: BinaryRefinementStrategy,
    observations: np.ndarray,
    ground_truth: np.ndarray,
    taus: Iterable[float] = (0.1, 0.25, 0.5, 0.75),
    save_barcode_path: str | None = None,
    original_signal: np.ndarray | None = None,
    **predict_kwargs,
) -> Tuple[RefinementResult, EvaluationResult]:
    refinement = strategy.predict(observations, **predict_kwargs)
    evaluation = evaluate_binary_predictions(ground_truth, refinement.sequence, taus=taus)
    if save_barcode_path:
        source_signal = observations if original_signal is None else original_signal
        saved = save_comparison_barcode(
            ground_truth=ground_truth,
            original_signal=source_signal,
            refined_signal=refinement.sequence,
            save_path=save_barcode_path,
        )
        evaluation.artifacts["barcode_comparison"] = saved
    return refinement, evaluation
