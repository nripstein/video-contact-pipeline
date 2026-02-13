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
_CONFIDENCE_COLOR_MAP = LinearSegmentedColormap.from_list("CustomConfidence", [_RED, _GREEN], N=256)


def _apply_barcode_layout(fig: plt.Figure) -> None:
    fig.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.12, hspace=0.35)


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

    _apply_barcode_layout(fig)
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

    _apply_barcode_layout(fig)
    if save_file:
        fig.savefig(save_file, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_confidence_iterative_barcodes(
    confidence: np.ndarray,
    data_list: Iterable[np.ndarray],
    confidence_title: str = "P(Holding)",
    titles: Optional[List[str]] = None,
    show: bool = False,
    save_file: Optional[str] = None,
) -> None:
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    data_list = list(data_list)
    if conf.size == 0 or not data_list:
        return

    titles = titles or [f"series_{i}" for i in range(len(data_list))]
    if len(titles) != len(data_list):
        raise ValueError("titles must match data_list length")

    for idx, row in enumerate(data_list):
        arr = np.asarray(row).reshape(-1)
        if arr.shape[0] != conf.shape[0]:
            raise ValueError(
                "Length mismatch for confidence barcode rows: "
                f"confidence={conf.shape[0]} row{idx}={arr.shape[0]}"
            )

    nrows = len(data_list) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, max(2, nrows * 1.2)))
    if nrows == 1:
        axes = [axes]

    conf_ax = axes[0]
    conf_ax.imshow(
        conf.reshape(1, -1),
        aspect="auto",
        cmap=_CONFIDENCE_COLOR_MAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    conf_ax.set_yticks([0])
    conf_ax.set_yticklabels([confidence_title])
    conf_ax.set_xticks([])
    conf_ax.set_ylabel("")

    conf_line_ax = conf_ax.twinx()
    conf_line_ax.plot(np.arange(conf.shape[0]), conf, color="black", linewidth=1.0)
    conf_line_ax.set_ylim(0.0, 1.0)
    conf_line_ax.set_xlim(-0.5, conf.shape[0] - 0.5)
    conf_line_ax.set_yticks([])
    conf_line_ax.set_xticks([])
    for spine in conf_line_ax.spines.values():
        spine.set_visible(False)

    for ax, row, title in zip(axes[1:], data_list, titles):
        data = np.asarray(row).astype(int).reshape(1, -1)
        ax.imshow(data, aspect="auto", cmap=_COLOR_MAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([title])
        ax.set_xticks([])
        ax.set_ylabel("")
    axes[-1].set_xlabel("Frame")

    _apply_barcode_layout(fig)
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


def _hand_side_to_lr(hand_side: object) -> int:
    if hand_side == "Right":
        return 1
    return 0


def save_annotated_frames(image_dir: str, full_df: pd.DataFrame, output_dir: str) -> List[str]:
    import cv2
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    archive_dir = repo_root / "archive"
    if str(archive_dir) not in sys.path:
        sys.path.insert(0, str(archive_dir))

    from nr_utils.bbox_draw import draw_presentation_bboxes

    image_dir = Path(image_dir)
    output_dir = Path(output_dir) / "visualizations" / "frames_det"
    output_dir.mkdir(parents=True, exist_ok=True)

    created: List[str] = []
    for frame_id, group in full_df.groupby("frame_id"):
        image_path = image_dir / frame_id
        if not image_path.exists():
            continue
        im = cv2.imread(str(image_path))
        if im is None:
            continue

        hand_rows = group[group["detection_type"] == "hand"]
        obj_rows = group[group["detection_type"] == "object"]

        hand_dets = None
        if not hand_rows.empty:
            hand_list = []
            for _, row in hand_rows.iterrows():
                hand_list.append([
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                    float(row["confidence"]) / 100.0,
                    float(row["contact_state"]),
                    float(row["offset_x"]),
                    float(row["offset_y"]),
                    float(row["offset_mag"]),
                    float(_hand_side_to_lr(row["hand_side"])),
                    1.0 if row.get("blue_glove_status", "NA") == "experimenter" else 0.0,
                ])
            hand_dets = np.array(hand_list, dtype=np.float32)

        obj_dets = None
        if not obj_rows.empty:
            obj_list = []
            for _, row in obj_rows.iterrows():
                obj_list.append([
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                    float(row["confidence"]) / 100.0,
                ])
            obj_dets = np.array(obj_list, dtype=np.float32)

        im = draw_presentation_bboxes(im, obj_dets, hand_dets)

        out_name = f"{Path(frame_id).stem}_det.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), im)
        created.append(str(out_path))

    return created
