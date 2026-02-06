from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.visualization import save_barcodes


def test_save_barcodes_creates_files(tmp_path: Path):
    condensed = pd.DataFrame(
        {
            "frame_number": [1, 2, 3],
            "contact_label": ["No Contact", "Portable Object", "No Contact"],
        }
    )

    created = save_barcodes(condensed, str(tmp_path))

    pred_path = tmp_path / "visualizations" / "barcode_pred.png"
    assert pred_path.exists()
    assert str(pred_path) in created
