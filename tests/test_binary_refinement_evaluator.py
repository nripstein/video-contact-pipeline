from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_refinement.evaluator import evaluate_binary_predictions, evaluate_strategy
from binary_refinement.identity import IdentityRefiner


def test_evaluate_binary_predictions_outputs_expected_fields():
    gt = np.array([0, 1, 1, 0, 0, 1], dtype=int)
    pred = np.array([0, 1, 0, 0, 1, 1], dtype=int)
    result = evaluate_binary_predictions(gt, pred, taus=(0.1, 0.5))

    assert 0.0 <= result.mof <= 1.0
    assert 0.0 <= result.edit <= 100.0
    assert set(result.f1.keys()) == {"F1@10", "F1@50"}
    assert result.n_frames == 6
    assert set(result.confusion.keys()) == {"tp", "fp", "tn", "fn"}


def test_evaluate_strategy_with_identity_refiner():
    gt = np.array([0, 0, 1, 1, 0], dtype=int)
    obs = np.array([0.1, 0.2, 0.9, 0.8, 0.3], dtype=float)
    refinement, metrics = evaluate_strategy(IdentityRefiner(), observations=obs, ground_truth=gt)

    assert refinement.sequence.tolist() == [0, 0, 1, 1, 0]
    assert metrics.mof == 1.0


def test_evaluate_strategy_saves_comparison_barcode(tmp_path: Path):
    gt = np.array([0, 0, 1, 1, 0], dtype=int)
    obs = np.array([0.2, 0.4, 0.8, 0.7, 0.3], dtype=float)
    out_path = tmp_path / "barcode_compare.png"

    _, metrics = evaluate_strategy(
        IdentityRefiner(),
        observations=obs,
        ground_truth=gt,
        save_barcode_path=str(out_path),
    )

    assert out_path.exists()
    assert metrics.artifacts["barcode_comparison"] == str(out_path)
