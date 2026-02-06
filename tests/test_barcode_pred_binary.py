from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.visualization import pred_binary_from_condensed


def test_pred_binary_from_condensed():
    condensed = pd.DataFrame(
        {
            "frame_number": [1, 2, 3, 4, 5],
            "contact_label": [
                "No Contact",
                "Portable Object",
                "Stationary Object",
                "Portable Object",
                "No Contact",
            ],
        }
    )
    binary = pred_binary_from_condensed(condensed)
    assert binary.tolist() == [0, 1, 0, 1, 0]
