from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.postprocessing import condense_dataframe


def _full_df_with_duplicate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_id": "1_demo.jpg",
                "frame_number": 1,
                "detection_type": "hand",
                "contact_label": "Portable Object",
                "hand_side": "Left",
                "is_filtered": False,
            },
            {
                "frame_id": "1_demo.jpg",
                "frame_number": 1,
                "detection_type": "hand",
                "contact_label": "No Contact",
                "hand_side": "Left",
                "is_filtered": False,
            },
        ]
    )


def test_condense_priority_strategy_defaults_to_no_contact_first():
    out = condense_dataframe(_full_df_with_duplicate_frame())
    assert out.iloc[0]["contact_label"] == "No Contact"


def test_condense_priority_strategy_portable_first():
    out = condense_dataframe(_full_df_with_duplicate_frame(), priority_strategy="portable_first")
    assert out.iloc[0]["contact_label"] == "Portable Object"


def test_condense_priority_strategy_rejects_unknown_value():
    with pytest.raises(ValueError):
        condense_dataframe(_full_df_with_duplicate_frame(), priority_strategy="unknown")

