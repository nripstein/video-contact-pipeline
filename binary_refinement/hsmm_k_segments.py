from __future__ import annotations

from dataclasses import replace
import math
from typing import List, Tuple

import numpy as np

from binary_refinement.base import BinaryRefinementStrategy
from binary_refinement.config import HSMMKSegmentsConfig
from binary_refinement.types import RefinementResult


def _count_transitions(seq: np.ndarray) -> int:
    if seq.size <= 1:
        return 0
    return int(np.count_nonzero(seq[1:] != seq[:-1]))


def _as_binary_observations(observations: np.ndarray) -> np.ndarray:
    obs = np.asarray(observations).reshape(-1)
    if obs.size == 0:
        raise ValueError("observations must be non-empty")
    if np.any(~np.isfinite(obs)):
        raise ValueError("observations must be finite")
    if np.issubdtype(obs.dtype, np.bool_):
        return obs.astype(int)

    obs_float = obs.astype(float)
    unique = np.unique(obs_float)
    if np.all(np.isin(unique, [0.0, 1.0])):
        return obs_float.astype(int)
    raise ValueError("observations must contain only binary values in {0, 1}")


def _state_for_segment(seg_idx: int, start_state: int) -> int:
    return start_state if (seg_idx % 2 == 0) else (1 - start_state)


def _gamma_log_duration(d: int, state: int, cfg: HSMMKSegmentsConfig) -> float:
    if d <= 0:
        raise ValueError(f"segment duration must be >= 1; got {d}")
    if state == 0:
        alpha = float(cfg.alpha_non_contact)
        rate = float(cfg.lambda_non_contact)
    else:
        alpha = float(cfg.alpha_contact)
        rate = float(cfg.lambda_contact)
    dur = float(d)
    return (
        alpha * math.log(rate)
        - math.lgamma(alpha)
        + (alpha - 1.0) * math.log(dur)
        - rate * dur
    )


class HSMMKSegmentsRefiner(BinaryRefinementStrategy):
    def __init__(self, config: HSMMKSegmentsConfig):
        self.config = config

    def predict(self, observations: np.ndarray, **kwargs) -> RefinementResult:
        cfg = replace(
            self.config,
            k_segments=int(kwargs.get("k_segments", self.config.k_segments)),
            start_state=int(kwargs.get("start_state", self.config.start_state)),
        )
        cfg.__post_init__()

        obs = _as_binary_observations(observations)
        n = int(obs.shape[0])
        if cfg.k_segments > n:
            raise ValueError(
                f"k_segments must be <= sequence length ({n}); got {cfg.k_segments}"
            )

        p_y1_x0 = float(np.clip(cfg.fpr, cfg.eps, 1.0 - cfg.eps))
        p_y0_x0 = float(np.clip(1.0 - cfg.fpr, cfg.eps, 1.0 - cfg.eps))
        p_y0_x1 = float(np.clip(cfg.fnr, cfg.eps, 1.0 - cfg.eps))
        p_y1_x1 = float(np.clip(1.0 - cfg.fnr, cfg.eps, 1.0 - cfg.eps))

        ll_state0 = np.where(obs == 1, math.log(p_y1_x0), math.log(p_y0_x0))
        ll_state1 = np.where(obs == 0, math.log(p_y0_x1), math.log(p_y1_x1))
        ll0_prefix = np.concatenate(([0.0], np.cumsum(ll_state0)))
        ll1_prefix = np.concatenate(([0.0], np.cumsum(ll_state1)))
        dur_ll_state0 = np.zeros(n + 1, dtype=float)
        dur_ll_state1 = np.zeros(n + 1, dtype=float)
        for d in range(1, n + 1):
            dur_ll_state0[d] = _gamma_log_duration(d, state=0, cfg=cfg)
            dur_ll_state1[d] = _gamma_log_duration(d, state=1, cfg=cfg)

        def segment_emission_loglik(i: int, j: int, state: int) -> float:
            if state == 0:
                return float(ll0_prefix[j] - ll0_prefix[i])
            return float(ll1_prefix[j] - ll1_prefix[i])

        k = int(cfg.k_segments)
        neg_inf = -float("inf")
        dp = np.full((k + 1, n + 1), neg_inf, dtype=float)
        back = np.full((k + 1, n + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for m in range(1, k + 1):
            seg_idx = m - 1
            state = _state_for_segment(seg_idx, int(cfg.start_state))

            min_t = m
            max_t = n - (k - m)
            for t in range(min_t, max_t + 1):
                u_min = m - 1
                u_max = t - 1
                best_score = neg_inf
                best_u = -1
                for u in range(u_min, u_max + 1):
                    prev = float(dp[m - 1, u])
                    if not np.isfinite(prev):
                        continue
                    d = t - u
                    emit = cfg.emission_weight * segment_emission_loglik(u, t, state)
                    dur_ll = float(dur_ll_state0[d] if state == 0 else dur_ll_state1[d])
                    dur = cfg.duration_weight * dur_ll
                    score = prev + emit + dur
                    if score > best_score:
                        best_score = score
                        best_u = u
                dp[m, t] = best_score
                back[m, t] = best_u

        objective = float(dp[k, n])
        if not np.isfinite(objective):
            raise RuntimeError("No feasible HSMM segmentation found.")

        segments: List[Tuple[int, int, int]] = []
        t = n
        for m in range(k, 0, -1):
            u = int(back[m, t])
            if u < 0:
                raise RuntimeError("Failed to backtrack HSMM segmentation.")
            state = _state_for_segment(m - 1, int(cfg.start_state))
            segments.append((u, t, state))
            t = u
        segments.reverse()

        refined = np.zeros(n, dtype=int)
        for start_idx, end_idx, state in segments:
            refined[start_idx:end_idx] = int(state)

        metadata_segments = [
            {
                "start_idx": int(start_idx),
                "end_idx_exclusive": int(end_idx),
                "length_frames": int(end_idx - start_idx),
                "state": int(state),
            }
            for start_idx, end_idx, state in segments
        ]

        return RefinementResult(
            sequence=refined,
            num_transitions=_count_transitions(refined),
            objective=objective,
            metadata={
                "method": "hsmm_k_segments_segmental_viterbi",
                "k_segments": int(cfg.k_segments),
                "start_state": int(cfg.start_state),
                "fpr": float(cfg.fpr),
                "fnr": float(cfg.fnr),
                "alpha_non_contact": float(cfg.alpha_non_contact),
                "lambda_non_contact": float(cfg.lambda_non_contact),
                "alpha_contact": float(cfg.alpha_contact),
                "lambda_contact": float(cfg.lambda_contact),
                "segments": metadata_segments,
            },
        )
