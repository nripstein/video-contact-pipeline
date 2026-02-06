from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_RED = np.append(np.array([200, 28, 52]) / 255, np.array([1]))
_GREEN = np.append(np.array([38, 140, 47]) / 255, np.array([1]))
_COLOR_MAP = LinearSegmentedColormap.from_list("Custom", [_RED, _GREEN], N=2)


def plot_barcode(gt: Optional[np.ndarray] = None,
                 pred: Optional[np.ndarray] = None,
                 show: bool = False,
                 save_file: Optional[str] = None) -> None:
    rows: List[np.ndarray] = []
    labels: List[str] = []
    if gt is not None:
        rows.append(gt)
        labels.append("GT")
    if pred is not None:
        rows.append(pred)
        labels.append("Pred")
    if not rows:
        return

    fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10, max(2, len(rows) * 1.2)))
    if len(rows) == 1:
        axes = [axes]
    for ax, row, label in zip(axes, rows, labels):
        data = row.astype(int).reshape(1, -1)
        ax.imshow(data, aspect="auto", cmap=_COLOR_MAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        ax.set_xticks([])
        ax.set_ylabel("")
    axes[-1].set_xlabel("Frame")

    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_iterative_barcodes(data_list: Iterable[np.ndarray],
                            titles: Optional[List[str]] = None,
                            show: bool = False,
                            save_file: Optional[str] = None) -> None:
    data_list = list(data_list)
    if not data_list:
        return
    titles = titles or [f"series_{i}" for i in range(len(data_list))]

    fig, axes = plt.subplots(nrows=len(data_list), ncols=1, figsize=(10, max(2, len(data_list) * 1.2)))
    if len(data_list) == 1:
        axes = [axes]
    for ax, row, title in zip(axes, data_list, titles):
        data = row.astype(int).reshape(1, -1)
        ax.imshow(data, aspect="auto", cmap=_COLOR_MAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([title])
        ax.set_xticks([])
        ax.set_ylabel("")
    axes[-1].set_xlabel("Frame")

    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def pred_binary_from_condensed(condensed_df: pd.DataFrame) -> np.ndarray:
    ordered = condensed_df.sort_values(by=["frame_number"], kind="mergesort")
    labels = ordered["contact_label"].fillna("")
    binary = (labels == "Portable Object").astype(int).to_numpy()
    return binary


def _label_to_binary(label: object) -> int:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return 0
    if isinstance(label, (int, np.integer)):
        return 1 if int(label) == 1 else 0
    if isinstance(label, float):
        return 1 if int(label) == 1 else 0
    text = str(label).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "false", "f", "no", "n"}:
        return 0
    if text == "holding":
        return 1
    if text == "not_holding":
        return 0
    return 0


def load_gt_binary_from_csv(gt_csv_path: str, frame_numbers: Iterable[int]) -> np.ndarray:
    gt_df = pd.read_csv(gt_csv_path, sep=None, engine="python")
    if "frame_number" in gt_df.columns and "gt_binary" in gt_df.columns:
        gt_vals = gt_df["gt_binary"].astype(int)
        if not gt_vals.isin([0, 1]).all():
            raise ValueError("gt_binary values must be 0 or 1.")
        mapping = dict(zip(gt_df["frame_number"], gt_vals))
    elif "frame_id" in gt_df.columns and "label" in gt_df.columns:
        extracted = gt_df["frame_id"].astype(str).str.extract(r"(\d+)")[0]
        valid = extracted.notna()
        frame_nums = extracted[valid].astype(int)
        labels = gt_df.loc[valid, "label"].map(_label_to_binary).astype(int)
        mapping = dict(zip(frame_nums, labels))
    else:
        raise ValueError(
            "GT CSV must contain either ('frame_number', 'gt_binary') or ('frame_id', 'label')."
        )

    aligned = [int(mapping.get(int(fn), 0)) for fn in frame_numbers]
    return np.array(aligned, dtype=int)


def save_barcodes(condensed_df: pd.DataFrame,
                  output_dir: str,
                  gt_csv_path: Optional[str] = None) -> List[str]:
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    ordered = condensed_df.sort_values(by=["frame_number"], kind="mergesort")
    frame_numbers = ordered["frame_number"].tolist()
    pred = pred_binary_from_condensed(ordered)

    created: List[str] = []
    pred_path = viz_dir / "barcode_pred.png"
    plot_barcode(pred=pred, show=False, save_file=str(pred_path))
    created.append(str(pred_path))

    if gt_csv_path:
        gt = load_gt_binary_from_csv(gt_csv_path, frame_numbers)
        pred_vs_gt_path = viz_dir / "barcode_pred_vs_gt.png"
        plot_iterative_barcodes([gt, pred], titles=["GT", "Pred"], show=False, save_file=str(pred_vs_gt_path))
        created.append(str(pred_vs_gt_path))

    return created
