from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_refinement.config import HSMMKSegmentsConfig
from binary_refinement.hsmm_k_segments import HSMMKSegmentsRefiner


def _valid_cfg(k_segments: int) -> HSMMKSegmentsConfig:
    return HSMMKSegmentsConfig(
        k_segments=k_segments,
        alpha_non_contact=32.0,
        lambda_non_contact=3.2,
        alpha_contact=30.0,
        lambda_contact=2.5,
        fpr=0.08,
        fnr=0.08,
        start_state=0,
        duration_weight=1.0,
        emission_weight=1.0,
    )


def test_hsmm_refiner_enforces_exact_k_segments():
    obs = np.array([0] * 7 + [1] * 9 + [0] * 8 + [1] * 10 + [0] * 6, dtype=int)
    obs[[3, 12, 21, 30]] = 1 - obs[[3, 12, 21, 30]]

    result = HSMMKSegmentsRefiner(_valid_cfg(k_segments=5)).predict(obs)

    assert result.sequence.shape == obs.shape
    assert set(np.unique(result.sequence)).issubset({0, 1})
    assert result.num_transitions == 4
    assert result.objective is not None
    assert len(result.metadata["segments"]) == 5


def test_hsmm_refiner_rejects_non_binary_input():
    obs = np.array([0.0, 1.0, 0.2, 1.0], dtype=float)
    with pytest.raises(ValueError):
        HSMMKSegmentsRefiner(_valid_cfg(k_segments=2)).predict(obs)


def test_hsmm_refiner_rejects_k_segments_larger_than_sequence():
    obs = np.array([0, 1, 0, 1], dtype=int)
    with pytest.raises(ValueError):
        HSMMKSegmentsRefiner(_valid_cfg(k_segments=5)).predict(obs)


def test_hsmm_segment_partition_is_complete_and_contiguous():
    obs = np.array([0] * 9 + [1] * 12 + [0] * 10 + [1] * 8, dtype=int)
    result = HSMMKSegmentsRefiner(_valid_cfg(k_segments=4)).predict(obs)
    segments = result.metadata["segments"]

    prev_end = 0
    total_len = 0
    for seg in segments:
        start_idx = int(seg["start_idx"])
        end_idx = int(seg["end_idx_exclusive"])
        seg_len = int(seg["length_frames"])
        assert start_idx == prev_end
        assert end_idx > start_idx
        assert seg_len == end_idx - start_idx
        prev_end = end_idx
        total_len += seg_len

    assert prev_end == obs.shape[0]
    assert total_len == obs.shape[0]


def test_hsmm_prefers_correct_emissions_in_easy_case():
    gt = np.array([0] * 10 + [1] * 12 + [0] * 9, dtype=int)
    obs = gt.copy()
    obs[[5, 17, 24]] = 1 - obs[[5, 17, 24]]

    cfg = HSMMKSegmentsConfig(
        k_segments=3,
        alpha_non_contact=50.0,
        lambda_non_contact=5.0,
        alpha_contact=72.0,
        lambda_contact=6.0,
        fpr=0.02,
        fnr=0.02,
        start_state=0,
        duration_weight=0.2,
        emission_weight=4.0,
    )

    result = HSMMKSegmentsRefiner(cfg).predict(obs)
    assert np.array_equal(result.sequence, gt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_segments": 0},
        {"alpha_non_contact": 0.0},
        {"lambda_non_contact": 0.0},
        {"alpha_contact": 0.0},
        {"lambda_contact": 0.0},
        {"fpr": 0.0},
        {"fpr": 1.0},
        {"fnr": 0.0},
        {"fnr": 1.0},
        {"start_state": 2},
        {"duration_weight": -1.0},
        {"emission_weight": -1.0},
        {"eps": 0.0},
        {"eps": 0.5},
    ],
)
def test_hsmm_config_rejects_invalid_values(kwargs):
    base = dict(
        k_segments=3,
        alpha_non_contact=10.0,
        lambda_non_contact=1.0,
        alpha_contact=10.0,
        lambda_contact=1.0,
        fpr=0.1,
        fnr=0.1,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        HSMMKSegmentsConfig(**base)
