from __future__ import annotations

import numpy as np

from binary_refinement.base import BinaryRefinementStrategy
from binary_refinement.types import RefinementResult


def _count_transitions(seq: np.ndarray) -> int:
    if seq.size <= 1:
        return 0
    return int(np.count_nonzero(seq[1:] != seq[:-1]))


class IdentityRefiner(BinaryRefinementStrategy):
    def predict(self, observations: np.ndarray, threshold: float = 0.5, **kwargs) -> RefinementResult:
        del kwargs
        obs = np.asarray(observations).reshape(-1)
        if obs.size == 0:
            raise ValueError("observations must be non-empty")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        if np.issubdtype(obs.dtype, np.bool_):
            seq = obs.astype(int)
        else:
            obs_float = obs.astype(float)
            seq = (obs_float >= threshold).astype(int)

        return RefinementResult(
            sequence=seq,
            num_transitions=_count_transitions(seq),
            objective=None,
            metadata={"threshold": float(threshold)},
        )
