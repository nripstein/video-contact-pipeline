from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stimulus_detector.data_generation.splitter import build_participant_split


def test_participant_split_is_deterministic():
    participants = ["sv1", "sv2", "sv3", "sv4", "sv5"]
    split_a = build_participant_split(participants, val_fraction=0.2, seed=42)
    split_b = build_participant_split(participants, val_fraction=0.2, seed=42)
    assert split_a == split_b


def test_participant_split_has_no_leakage():
    participants = ["sv1", "sv2", "sv3", "sv4"]
    split_map = build_participant_split(participants, val_fraction=0.25, seed=7)

    train = {k for k, v in split_map.items() if v == "train"}
    val = {k for k, v in split_map.items() if v == "val"}

    assert train
    assert val
    assert train.isdisjoint(val)
