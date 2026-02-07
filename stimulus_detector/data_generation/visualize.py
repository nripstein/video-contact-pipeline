from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
import numpy as np

from stimulus_detector.data_generation.types import PseudoLabel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sample_evenly(labels: List[PseudoLabel], sample_size: int) -> List[PseudoLabel]:
    if sample_size <= 0 or len(labels) <= sample_size:
        return labels
    idx = np.linspace(0, len(labels) - 1, num=sample_size, dtype=int)
    return [labels[i] for i in idx]


def save_selection_overlays(
    labels: List[PseudoLabel],
    output_dir: str,
    sample_size: int,
) -> List[str]:
    out_dir = Path(output_dir).expanduser() / "reports" / "selection_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _sample_evenly(
        sorted(labels, key=lambda x: (x.frame.video_id, x.frame.frame_idx)),
        sample_size=sample_size,
    )

    created: List[str] = []
    for label in selected:
        img = cv2.imread(label.frame.frame_path)
        if img is None:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in label.bbox_xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 220, 0), thickness=2)
        text = f"stimulus {label.confidence:.2f}"
        cv2.putText(img, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        out_name = (
            f"{label.frame.participant_id}__{label.frame.video_id}"
            f"__{int(label.frame.frame_idx):06d}.png"
        )
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), img)
        created.append(str(out_path))

    return created


def save_pose_diversity_heatmap(
    labels: List[PseudoLabel],
    output_dir: str,
    bins: int,
) -> Optional[str]:
    if not labels:
        return None

    out_path = Path(output_dir).expanduser() / "reports" / "pose_diversity_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = []
    ys = []
    for label in labels:
        cx, cy = label.center()
        w = max(1.0, float(label.frame.width))
        h = max(1.0, float(label.frame.height))
        xs.append(cx / w)
        ys.append(cy / h)

    xs_arr = np.clip(np.array(xs, dtype=float), 0.0, 1.0)
    ys_arr = np.clip(np.array(ys, dtype=float), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    hist = ax.hist2d(xs_arr, ys_arr, bins=bins, range=[[0, 1], [0, 1]], cmap="magma")
    fig.colorbar(hist[3], ax=ax, label="Count")
    ax.set_title("Selected Stimulus Center Distribution")
    ax.set_xlabel("Normalized X Center")
    ax.set_ylabel("Normalized Y Center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return str(out_path)
