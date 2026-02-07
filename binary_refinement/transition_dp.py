from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

import numpy as np

from binary_refinement.base import BinaryRefinementStrategy
from binary_refinement.config import DurationPriorConfig, TransitionDPConfig
from binary_refinement.types import RefinementResult


def _count_transitions(seq: np.ndarray) -> int:
    if seq.size <= 1:
        return 0
    return int(np.count_nonzero(seq[1:] != seq[:-1]))


def _as_probabilities(observations: np.ndarray, eps: float) -> np.ndarray:
    probs = np.asarray(observations, dtype=float).reshape(-1)
    if probs.size == 0:
        raise ValueError("observations must be non-empty")
    if np.any(~np.isfinite(probs)):
        raise ValueError("observations must be finite")

    if np.all((probs >= 0.0) & (probs <= 1.0)):
        return np.clip(probs, eps, 1.0 - eps)

    raise ValueError("observations must contain probabilities in [0, 1] or binary values")


def _duration_penalty(
    seg_len_frames: int,
    state: int,
    is_edge: bool,
    prior: DurationPriorConfig,
) -> float:
    mean_sec = prior.contact_mean_sec if state == 1 else prior.non_contact_mean_sec
    if is_edge:
        sigma_sec = prior.edge_contact_sigma_sec if state == 1 else prior.edge_non_contact_sigma_sec
    else:
        sigma_sec = prior.contact_sigma_sec if state == 1 else prior.non_contact_sigma_sec

    mean_frames = mean_sec * prior.fps
    sigma_frames = sigma_sec * prior.fps
    z = (float(seg_len_frames) - float(mean_frames)) / float(sigma_frames)
    return 0.5 * float(z * z)


class TransitionConstrainedDPRefiner(BinaryRefinementStrategy):
    def __init__(self, config: TransitionDPConfig):
        self.config = config

    def predict(self, observations: np.ndarray, **kwargs) -> RefinementResult:
        cfg = replace(
            self.config,
            k_transitions=int(kwargs.get("k_transitions", self.config.k_transitions)),
            start_state=kwargs.get("start_state", self.config.start_state),
        )
        cfg.__post_init__()

        probs = _as_probabilities(observations, cfg.eps)
        n = int(probs.shape[0])
        if cfg.k_transitions >= n:
            raise ValueError(
                f"k_transitions must be less than sequence length ({n}); got {cfg.k_transitions}"
            )

        if cfg.start_state is None:
            candidates = [self._solve_for_start_state(probs, cfg, start_state=0), self._solve_for_start_state(probs, cfg, start_state=1)]
            return min(candidates, key=lambda x: float("inf") if x.objective is None else x.objective)

        if cfg.start_state not in (0, 1):
            raise ValueError("start_state must be None, 0, or 1")
        return self._solve_for_start_state(probs, cfg, start_state=int(cfg.start_state))

    def _solve_for_start_state(
        self,
        probs: np.ndarray,
        cfg: TransitionDPConfig,
        start_state: int,
    ) -> RefinementResult:
        n = int(probs.shape[0])
        num_segments = cfg.k_transitions + 1
        if num_segments > n:
            raise ValueError(
                f"Need at least one frame per segment, but got {num_segments} segments for length {n}."
            )

        pos_cost = -np.log(probs)
        neg_cost = -np.log(1.0 - probs)
        pos_prefix = np.concatenate(([0.0], np.cumsum(pos_cost)))
        neg_prefix = np.concatenate(([0.0], np.cumsum(neg_cost)))

        def seg_state(seg_idx: int) -> int:
            return start_state if seg_idx % 2 == 0 else 1 - start_state

        def obs_segment_cost(i: int, j: int, state: int) -> float:
            if state == 1:
                return float(pos_prefix[j] - pos_prefix[i])
            return float(neg_prefix[j] - neg_prefix[i])

        inf = float("inf")
        dp = np.full((num_segments + 1, n + 1), inf, dtype=float)
        back = np.full((num_segments + 1, n + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for m in range(1, num_segments + 1):
            seg_idx = m - 1
            state = seg_state(seg_idx)
            is_first = seg_idx == 0
            is_last = seg_idx == (num_segments - 1)

            min_j = m
            max_j = n - (num_segments - m)
            for j in range(min_j, max_j + 1):
                i_min = m - 1
                i_max = j - 1
                best_cost = inf
                best_i = -1
                for i in range(i_min, i_max + 1):
                    prev = float(dp[m - 1, i])
                    if not np.isfinite(prev):
                        continue
                    seg_len = j - i
                    fit_cost = cfg.observation_weight * obs_segment_cost(i, j, state)
                    dur_cost = cfg.duration_weight * _duration_penalty(
                        seg_len_frames=seg_len,
                        state=state,
                        is_edge=(is_first or is_last),
                        prior=cfg.duration_prior,
                    )
                    total = prev + fit_cost + dur_cost
                    if total < best_cost:
                        best_cost = total
                        best_i = i
                dp[m, j] = best_cost
                back[m, j] = best_i

        objective = float(dp[num_segments, n])
        if not np.isfinite(objective):
            raise RuntimeError("No feasible refinement found.")

        segments: List[Tuple[int, int, int]] = []
        j = n
        for m in range(num_segments, 0, -1):
            i = int(back[m, j])
            if i < 0:
                raise RuntimeError("Failed to backtrack DP segmentation.")
            state = seg_state(m - 1)
            segments.append((i, j, state))
            j = i
        segments.reverse()

        refined = np.zeros(n, dtype=int)
        for i, j, state in segments:
            refined[i:j] = int(state)

        metadata_segments = [
            {
                "start_idx": int(i),
                "end_idx_exclusive": int(j),
                "length_frames": int(j - i),
                "state": int(state),
            }
            for i, j, state in segments
        ]

        return RefinementResult(
            sequence=refined,
            num_transitions=_count_transitions(refined),
            objective=objective,
            metadata={
                "method": "transition_constrained_dp",
                "k_transitions": int(cfg.k_transitions),
                "start_state": int(start_state),
                "num_segments": int(num_segments),
                "segments": metadata_segments,
            },
        )
