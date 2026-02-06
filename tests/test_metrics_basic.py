from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.metrics import edit_score, f_score, frame_accuracy


def test_metrics_basic_binary():
    pred = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    gt = np.array([0, 1, 1, 1, 0, 0], dtype=int)

    acc = frame_accuracy(pred, gt)
    assert acc == 4 / 6

    edit = edit_score(pred, gt, bg_class=(0,), norm=True)
    assert 0 <= edit <= 100

    f1 = f_score(pred, gt, overlap=0.5, bg_class=(0,))
    assert 0 <= f1 <= 100
