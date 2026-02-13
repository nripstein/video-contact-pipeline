from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Optional

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import binary_refinement._hsmm_kernels as hsmm_kernels
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


def _iter_durations(n: int, k: int, max_segment_length: Optional[int]):
    if k == 1:
        if n >= 1 and (max_segment_length is None or n <= max_segment_length):
            yield (n,)
        return

    min_first = 1
    max_first = n - (k - 1)
    if max_segment_length is not None:
        min_first = max(min_first, n - (k - 1) * max_segment_length)
        max_first = min(max_first, max_segment_length)
    if min_first > max_first:
        return

    for d in range(min_first, max_first + 1):
        for rest in _iter_durations(n=n - d, k=k - 1, max_segment_length=max_segment_length):
            yield (d,) + rest


def _gamma_log_duration_reference(d: int, state: int, cfg: HSMMKSegmentsConfig) -> float:
    alpha = float(cfg.alpha_non_contact) if state == 0 else float(cfg.alpha_contact)
    rate = float(cfg.lambda_non_contact) if state == 0 else float(cfg.lambda_contact)
    dur = float(d)
    return alpha * math.log(rate) - math.lgamma(alpha) + (alpha - 1.0) * math.log(dur) - rate * dur


def _enumerated_contact_posteriors(obs: np.ndarray, cfg: HSMMKSegmentsConfig) -> np.ndarray:
    n = int(obs.shape[0])
    eps = float(cfg.eps)
    p_y1_x0 = float(np.clip(cfg.fpr, eps, 1.0 - eps))
    p_y0_x0 = float(np.clip(1.0 - cfg.fpr, eps, 1.0 - eps))
    p_y0_x1 = float(np.clip(cfg.fnr, eps, 1.0 - eps))
    p_y1_x1 = float(np.clip(1.0 - cfg.fnr, eps, 1.0 - eps))

    ll_state0 = np.where(obs == 1, math.log(p_y1_x0), math.log(p_y0_x0))
    ll_state1 = np.where(obs == 0, math.log(p_y0_x1), math.log(p_y1_x1))
    ll0_prefix = np.concatenate(([0.0], np.cumsum(ll_state0)))
    ll1_prefix = np.concatenate(([0.0], np.cumsum(ll_state1)))

    max_len = cfg.max_segment_length_frames
    contact_masks = []
    log_scores = []
    for durations in _iter_durations(n=n, k=int(cfg.k_segments), max_segment_length=max_len):
        start = 0
        total = 0.0
        mask = np.zeros(n, dtype=float)
        for seg_idx, d in enumerate(durations):
            state = int(cfg.start_state) if seg_idx % 2 == 0 else (1 - int(cfg.start_state))
            end = start + d
            emit_ll = float(ll0_prefix[end] - ll0_prefix[start]) if state == 0 else float(ll1_prefix[end] - ll1_prefix[start])
            dur_ll = _gamma_log_duration_reference(d=d, state=state, cfg=cfg)
            total += float(cfg.emission_weight) * emit_ll
            total += float(cfg.duration_weight) * dur_ll
            if state == 1:
                mask[start:end] = 1.0
            start = end
        if start != n:
            continue
        contact_masks.append(mask)
        log_scores.append(total)

    if not log_scores:
        raise RuntimeError("Enumeration produced no feasible HSMM segmentation.")

    log_scores_arr = np.asarray(log_scores, dtype=float)
    max_log = float(np.max(log_scores_arr))
    weights = np.exp(log_scores_arr - max_log)
    weights = weights / float(np.sum(weights))
    mask_matrix = np.vstack(contact_masks)
    return np.sum(weights[:, None] * mask_matrix, axis=0)


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


def test_hsmm_enforces_max_segment_length_on_edge_segments():
    obs = np.array([0, 0, 0, 1, 1, 1, 0], dtype=int)
    cfg = _valid_cfg(k_segments=1)
    cfg.max_segment_length_frames = 6

    with pytest.raises(ValueError, match="No feasible segmentation"):
        HSMMKSegmentsRefiner(cfg).predict(obs)


def test_hsmm_with_cap_none_matches_legacy_feasibility():
    obs = np.array([0, 0, 0, 1, 1, 1, 0], dtype=int)
    cfg = _valid_cfg(k_segments=1)
    cfg.max_segment_length_frames = None

    result = HSMMKSegmentsRefiner(cfg).predict(obs)
    assert result.sequence.shape == obs.shape
    assert len(result.metadata["segments"]) == 1


def test_hsmm_segment_lengths_respect_cap_when_feasible():
    obs = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=int)
    cfg = _valid_cfg(k_segments=2)
    cfg.max_segment_length_frames = 6

    result = HSMMKSegmentsRefiner(cfg).predict(obs)
    lengths = [int(seg["length_frames"]) for seg in result.metadata["segments"]]
    assert all(length <= 6 for length in lengths)


def test_hsmm_metadata_reports_backend_and_cap(monkeypatch):
    monkeypatch.setattr(hsmm_kernels, "_NUMBA_AVAILABLE", False)
    obs = np.array([0, 0, 1, 1, 0, 0], dtype=int)
    cfg = _valid_cfg(k_segments=2)
    cfg.max_segment_length_frames = 5
    cfg.numba_mode = "auto"

    result = HSMMKSegmentsRefiner(cfg).predict(obs)
    assert result.metadata["max_segment_length_frames"] == 5
    assert result.metadata["numba_mode"] == "auto"
    assert result.metadata["decoder_backend"] == "python"


def test_hsmm_default_does_not_return_posteriors():
    obs = np.array([0, 0, 1, 1, 0, 0], dtype=int)
    cfg = _valid_cfg(k_segments=2)

    result = HSMMKSegmentsRefiner(cfg).predict(obs)
    assert result.posteriors is None
    assert result.metadata["return_posteriors"] is False


def test_hsmm_return_posteriors_shape_and_range():
    obs = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=int)
    cfg = _valid_cfg(k_segments=4)
    cfg.max_segment_length_frames = 5

    result = HSMMKSegmentsRefiner(cfg).predict(obs, return_posteriors=True)

    assert result.posteriors is not None
    assert result.posteriors.shape == obs.shape
    assert np.all(result.posteriors >= 0.0)
    assert np.all(result.posteriors <= 1.0)
    assert result.metadata["return_posteriors"] is True


def test_hsmm_posteriors_match_unique_segmentation_when_k_equals_n():
    obs = np.array([0, 1, 0, 1, 1], dtype=int)
    cfg = _valid_cfg(k_segments=obs.shape[0])
    cfg.start_state = 1
    cfg.max_segment_length_frames = 1

    result = HSMMKSegmentsRefiner(cfg).predict(obs, return_posteriors=True)

    expected_states = np.array([1, 0, 1, 0, 1], dtype=int)
    assert np.array_equal(result.sequence, expected_states)
    assert result.posteriors is not None
    assert np.allclose(result.posteriors, expected_states.astype(float), atol=1e-12)


def test_hsmm_posteriors_match_bruteforce_enumeration_small_case():
    obs = np.array([0, 1, 1, 0, 1], dtype=int)
    cfg = HSMMKSegmentsConfig(
        k_segments=3,
        alpha_non_contact=5.0,
        lambda_non_contact=1.5,
        alpha_contact=4.0,
        lambda_contact=1.2,
        fpr=0.12,
        fnr=0.09,
        start_state=0,
        duration_weight=0.7,
        emission_weight=1.3,
        max_segment_length_frames=3,
        numba_mode="off",
    )

    result = HSMMKSegmentsRefiner(cfg).predict(obs, return_posteriors=True)
    expected = _enumerated_contact_posteriors(obs=obs, cfg=cfg)

    assert result.posteriors is not None
    assert np.allclose(result.posteriors, expected, atol=1e-8)


def test_hsmm_numba_on_requires_numba(monkeypatch):
    monkeypatch.setattr(hsmm_kernels, "_NUMBA_AVAILABLE", False)
    obs = np.array([0, 0, 1, 1, 0, 0], dtype=int)
    cfg = _valid_cfg(k_segments=2)
    cfg.numba_mode = "on"

    with pytest.raises(RuntimeError, match="requires numba"):
        HSMMKSegmentsRefiner(cfg).predict(obs)


def test_hsmm_python_vs_numba_exact_parity():
    if not hsmm_kernels.numba_is_available():
        pytest.skip("numba not installed")

    obs = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0], dtype=int)
    cfg_py = _valid_cfg(k_segments=4)
    cfg_py.max_segment_length_frames = 6
    cfg_py.numba_mode = "off"
    result_py = HSMMKSegmentsRefiner(cfg_py).predict(obs, return_posteriors=True)

    cfg_nb = _valid_cfg(k_segments=4)
    cfg_nb.max_segment_length_frames = 6
    cfg_nb.numba_mode = "on"
    result_nb = HSMMKSegmentsRefiner(cfg_nb).predict(obs, return_posteriors=True)

    assert np.array_equal(result_py.sequence, result_nb.sequence)
    assert result_py.objective == pytest.approx(result_nb.objective, abs=1e-10)
    assert result_py.posteriors is not None
    assert result_nb.posteriors is not None
    assert np.allclose(result_py.posteriors, result_nb.posteriors, atol=1e-10)
    assert result_nb.metadata["decoder_backend"] == "numba"


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
        {"max_segment_length_frames": 0},
        {"numba_mode": "invalid"},
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
