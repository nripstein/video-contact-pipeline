from __future__ import annotations

from dataclasses import replace
import math
from typing import List, Optional, Tuple

import numpy as np

from . import _hsmm_kernels as hsmm_kernels
from binary_refinement.base import BinaryRefinementStrategy
from binary_refinement.config import HSMMKSegmentsConfig
from binary_refinement.types import RefinementResult


def _tqdm():
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm


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


def _resolve_backend(numba_mode: str) -> str:
    mode = str(numba_mode).strip().lower()
    has_numba = hsmm_kernels.numba_is_available()
    if mode == "off":
        return "python"
    if mode == "on":
        if not has_numba:
            raise RuntimeError("numba_mode='on' requires numba to be installed.")
        return "numba"
    if mode == "auto":
        return "numba" if has_numba else "python"
    raise ValueError(f"Unsupported numba_mode: {numba_mode}")


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
        use_progress = bool(kwargs.get("progress", False))
        progress_desc = str(kwargs.get("progress_desc", "hsmm_dp"))
        max_segment_length_frames: Optional[int] = kwargs.get(
            "max_segment_length_frames", self.config.max_segment_length_frames
        )
        if max_segment_length_frames is not None:
            max_segment_length_frames = int(max_segment_length_frames)
        numba_mode = str(kwargs.get("numba_mode", self.config.numba_mode))

        cfg = replace(
            self.config,
            k_segments=int(kwargs.get("k_segments", self.config.k_segments)),
            start_state=int(kwargs.get("start_state", self.config.start_state)),
            max_segment_length_frames=max_segment_length_frames,
            numba_mode=numba_mode,
        )
        cfg.__post_init__()

        obs = _as_binary_observations(observations)
        n = int(obs.shape[0])
        if cfg.k_segments > n:
            raise ValueError(
                f"k_segments must be <= sequence length ({n}); got {cfg.k_segments}"
            )
        if cfg.max_segment_length_frames is not None:
            max_total_frames = int(cfg.k_segments) * int(cfg.max_segment_length_frames)
            if n > max_total_frames:
                raise ValueError(
                    f"No feasible segmentation: sequence length ({n}) exceeds "
                    f"k_segments * max_segment_length_frames ({max_total_frames})."
                )

        p_y1_x0 = float(np.clip(cfg.fpr, cfg.eps, 1.0 - cfg.eps))
        p_y0_x0 = float(np.clip(1.0 - cfg.fpr, cfg.eps, 1.0 - cfg.eps))
        p_y0_x1 = float(np.clip(cfg.fnr, cfg.eps, 1.0 - cfg.eps))
        p_y1_x1 = float(np.clip(1.0 - cfg.fnr, cfg.eps, 1.0 - cfg.eps))

        ll_state0 = np.where(obs == 1, math.log(p_y1_x0), math.log(p_y0_x0))
        ll_state1 = np.where(obs == 0, math.log(p_y0_x1), math.log(p_y1_x1))
        ll0_prefix = np.concatenate(([0.0], np.cumsum(ll_state0)))
        ll1_prefix = np.concatenate(([0.0], np.cumsum(ll_state1)))
        max_dur = int(cfg.max_segment_length_frames) if cfg.max_segment_length_frames is not None else n
        max_dur = min(max_dur, n)
        dur_ll_state0 = np.zeros(max_dur + 1, dtype=float)
        dur_ll_state1 = np.zeros(max_dur + 1, dtype=float)
        for d in range(1, max_dur + 1):
            dur_ll_state0[d] = _gamma_log_duration(d, state=0, cfg=cfg)
            dur_ll_state1[d] = _gamma_log_duration(d, state=1, cfg=cfg)

        k = int(cfg.k_segments)
        max_segment_length = int(cfg.max_segment_length_frames) if cfg.max_segment_length_frames is not None else -1
        back = np.full((k + 1, n + 1), -1, dtype=np.int32)
        dp_prev = np.full(n + 1, -np.inf, dtype=float)
        dp_curr = np.full(n + 1, -np.inf, dtype=float)
        dp_prev[0] = 0.0

        backend = _resolve_backend(cfg.numba_mode)
        fill_segment = hsmm_kernels.fill_segment_python
        if backend == "numba":
            fill_segment = hsmm_kernels.fill_segment_numba

        iterable = range(1, k + 1)
        if use_progress:
            tqdm_fn = _tqdm()
            if tqdm_fn is not None:
                iterable = tqdm_fn(iterable, desc=progress_desc, unit="segment")

        for m in iterable:
            fill_segment(
                m=m,
                k=k,
                n=n,
                start_state=int(cfg.start_state),
                emission_weight=float(cfg.emission_weight),
                duration_weight=float(cfg.duration_weight),
                max_segment_length=max_segment_length,
                dp_prev=dp_prev,
                dp_curr=dp_curr,
                back_row=back[m],
                ll0_prefix=ll0_prefix,
                ll1_prefix=ll1_prefix,
                dur_ll_state0=dur_ll_state0,
                dur_ll_state1=dur_ll_state1,
            )
            dp_prev, dp_curr = dp_curr, dp_prev

        objective = float(dp_prev[n])
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
                "max_segment_length_frames": (
                    int(cfg.max_segment_length_frames) if cfg.max_segment_length_frames is not None else None
                ),
                "numba_mode": str(cfg.numba_mode),
                "decoder_backend": backend,
                "segments": metadata_segments,
            },
        )
