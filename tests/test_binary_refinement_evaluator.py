from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_refinement.evaluator import (
    evaluate_binary_predictions,
    evaluate_strategy,
    save_confidence_refined_gt_barcode,
    save_original_refined_gt_barcode,
    save_original_vs_refined_barcode,
)
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


def test_save_original_vs_refined_barcode(tmp_path: Path):
    orig = np.array([0, 0, 1, 1, 0], dtype=int)
    ref = np.array([0, 1, 1, 0, 0], dtype=int)
    out_path = tmp_path / "barcode_refinement.png"

    saved = save_original_vs_refined_barcode(orig, ref, str(out_path))

    assert out_path.exists()
    assert saved == str(out_path)
    with Image.open(out_path) as image:
        assert image.size == (2000, 480)


def test_save_original_refined_gt_barcode(tmp_path: Path):
    orig = np.array([0, 0, 1, 1, 0], dtype=int)
    ref = np.array([0, 1, 1, 0, 0], dtype=int)
    gt = np.array([0, 0, 1, 0, 0], dtype=int)
    out_path = tmp_path / "barcode_refinement_gt.png"

    saved = save_original_refined_gt_barcode(orig, ref, gt, str(out_path))

    assert out_path.exists()
    assert saved == str(out_path)
    with Image.open(out_path) as image:
        assert image.size[0] == 2000
        assert image.size[1] in {719, 720}


def test_save_confidence_refined_gt_barcode(tmp_path: Path):
    conf = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    pred = np.array([0, 0, 1, 1, 1], dtype=int)
    gt = np.array([0, 1, 1, 0, 1], dtype=int)
    out_path = tmp_path / "barcode_confidence_refined_gt.png"

    saved = save_confidence_refined_gt_barcode(conf, pred, gt, str(out_path))

    assert out_path.exists()
    assert saved == str(out_path)
    with Image.open(out_path) as image:
        assert image.size[0] == 2000
        assert image.size[1] in {719, 720}


def test_save_confidence_refined_gt_barcode_rejects_length_mismatch(tmp_path: Path):
    conf = np.array([0.1, 0.2, 0.3], dtype=float)
    pred = np.array([0, 1], dtype=int)
    gt = np.array([0, 1, 1], dtype=int)
    out_path = tmp_path / "bad.png"

    with np.testing.assert_raises_regex(ValueError, "Length mismatch"):
        save_confidence_refined_gt_barcode(conf, pred, gt, str(out_path))


def test_save_confidence_refined_gt_barcode_rejects_confidence_out_of_range(tmp_path: Path):
    conf = np.array([0.1, 1.2, 0.3], dtype=float)
    pred = np.array([0, 1, 0], dtype=int)
    gt = np.array([0, 1, 0], dtype=int)
    out_path = tmp_path / "bad_range.png"

    with np.testing.assert_raises_regex(ValueError, "confidence must be in \\[0, 1\\]"):
        save_confidence_refined_gt_barcode(conf, pred, gt, str(out_path))


def test_save_confidence_refined_gt_barcode_rejects_non_finite(tmp_path: Path):
    conf = np.array([0.1, np.nan, 0.3], dtype=float)
    pred = np.array([0, 1, 0], dtype=int)
    gt = np.array([0, 1, 0], dtype=int)
    out_path = tmp_path / "bad_nan.png"

    with np.testing.assert_raises_regex(ValueError, "confidence must be finite"):
        save_confidence_refined_gt_barcode(conf, pred, gt, str(out_path))
