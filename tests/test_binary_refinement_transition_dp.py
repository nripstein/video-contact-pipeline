from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_refinement.config import DurationPriorConfig, TransitionDPConfig
from binary_refinement.transition_dp import TransitionConstrainedDPRefiner


def _make_probs_from_gt(gt: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = np.where(gt == 1, 0.88, 0.12) + rng.normal(0.0, 0.06, size=gt.shape[0])
    return np.clip(probs, 0.01, 0.99)


def test_transition_dp_refiner_enforces_exact_k():
    gt = np.array([0] * 8 + [1] * 12 + [0] * 10 + [1] * 14 + [0] * 9, dtype=int)
    probs = _make_probs_from_gt(gt, seed=13)

    prior = DurationPriorConfig(
        contact_mean_sec=0.20,
        non_contact_mean_sec=0.15,
        contact_sigma_sec=0.05,
        non_contact_sigma_sec=0.05,
        edge_contact_sigma_sec=0.12,
        edge_non_contact_sigma_sec=0.12,
        fps=60.0,
    )
    cfg = TransitionDPConfig(
        k_transitions=4,
        duration_prior=prior,
        observation_weight=1.0,
        duration_weight=0.5,
        start_state=0,
    )
    result = TransitionConstrainedDPRefiner(cfg).predict(probs)

    assert result.sequence.shape == probs.shape
    assert set(np.unique(result.sequence)).issubset({0, 1})
    assert result.num_transitions == 4
    assert result.objective is not None
    assert len(result.metadata["segments"]) == 5


def test_transition_dp_rejects_invalid_k():
    probs = np.array([0.1, 0.8, 0.2], dtype=float)
    prior = DurationPriorConfig(
        contact_mean_sec=0.1,
        non_contact_mean_sec=0.1,
        contact_sigma_sec=0.1,
        non_contact_sigma_sec=0.1,
        edge_contact_sigma_sec=0.2,
        edge_non_contact_sigma_sec=0.2,
        fps=60.0,
    )
    cfg = TransitionDPConfig(k_transitions=3, duration_prior=prior)
    refiner = TransitionConstrainedDPRefiner(cfg)
    with pytest.raises(ValueError):
        refiner.predict(probs)
