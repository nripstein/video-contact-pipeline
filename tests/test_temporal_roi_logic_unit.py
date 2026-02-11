from __future__ import annotations

import numpy as np

from pipeline.temporal_roi import TemporalROIPropagator, select_stimulus_bbox


def _det(obj_rows=None):
    if obj_rows is None:
        obj_dets = None
    else:
        obj_dets = np.asarray(obj_rows, dtype=np.float32)
    return {"hand_dets": None, "obj_dets": obj_dets}


def test_select_stimulus_bbox_prefers_continuity_over_confidence():
    prev = (50.0, 50.0, 80.0, 80.0)
    obj_dets = np.asarray(
        [
            [49.0, 49.0, 79.0, 79.0, 0.40],
            [0.0, 0.0, 20.0, 20.0, 0.99],
        ],
        dtype=np.float32,
    )
    picked = select_stimulus_bbox(obj_dets, prev)
    assert picked == (49.0, 49.0, 79.0, 79.0)


def test_temporal_propagation_runs_second_pass_with_injected_roi():
    calls = []
    responses = [
        _det([[10.0, 10.0, 20.0, 20.0, 0.95]]),
        _det(None),
        _det([[11.0, 11.0, 21.0, 21.0, 0.70]]),
    ]

    def detect_once(_im, extra_rois):
        calls.append(None if extra_rois is None else np.asarray(extra_rois, dtype=np.float32).copy())
        return responses[len(calls) - 1]

    propagator = TemporalROIPropagator(
        detect_once_fn=detect_once,
        blue_guard_fn=lambda _im, _bbox: False,
        max_missed_frames=8,
    )

    im = np.zeros((32, 32, 3), dtype=np.uint8)
    propagator.detect(im)
    propagator.detect(im)

    assert len(calls) == 3
    assert calls[0] is None
    assert calls[1] is None
    assert calls[2] is not None
    np.testing.assert_allclose(calls[2], np.asarray([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32))
    assert propagator.state.last_known_stimulus_bbox == (11.0, 11.0, 21.0, 21.0)
    assert propagator.state.missed_count == 0


def test_temporal_propagation_blocked_by_blue_guard():
    calls = []
    responses = [
        _det([[10.0, 10.0, 20.0, 20.0, 0.95]]),
        _det(None),
    ]

    def detect_once(_im, extra_rois):
        calls.append(None if extra_rois is None else np.asarray(extra_rois, dtype=np.float32).copy())
        return responses[len(calls) - 1]

    guard_calls = {"n": 0}

    def blue_guard(_im, _bbox):
        guard_calls["n"] += 1
        return guard_calls["n"] > 1

    propagator = TemporalROIPropagator(
        detect_once_fn=detect_once,
        blue_guard_fn=blue_guard,
        max_missed_frames=8,
    )

    im = np.zeros((32, 32, 3), dtype=np.uint8)
    propagator.detect(im)
    propagator.detect(im)

    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is None
    assert propagator.state.missed_count == 1


def test_temporal_propagation_respects_miss_cap_and_clears_state():
    calls = []
    responses = [
        _det([[10.0, 10.0, 20.0, 20.0, 0.95]]),
        _det(None),
        _det(None),
        _det(None),
        _det(None),
        _det(None),
    ]

    def detect_once(_im, extra_rois):
        calls.append(None if extra_rois is None else np.asarray(extra_rois, dtype=np.float32).copy())
        return responses[len(calls) - 1]

    propagator = TemporalROIPropagator(
        detect_once_fn=detect_once,
        blue_guard_fn=lambda _im, _bbox: False,
        max_missed_frames=2,
    )

    im = np.zeros((32, 32, 3), dtype=np.uint8)
    propagator.detect(im)  # frame 1: seed
    propagator.detect(im)  # frame 2: miss + fallback miss
    assert propagator.state.last_known_stimulus_bbox is not None
    assert propagator.state.missed_count == 1

    propagator.detect(im)  # frame 3: miss + fallback miss -> clear state
    assert propagator.state.last_known_stimulus_bbox is None
    assert propagator.state.missed_count == 0

    propagator.detect(im)  # frame 4: no extra pass possible

    assert len(calls) == 6
    assert calls[2] is not None
    assert calls[4] is not None
    assert calls[5] is None
