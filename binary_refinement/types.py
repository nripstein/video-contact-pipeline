from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RefinementResult:
    sequence: np.ndarray
    num_transitions: int
    objective: Optional[float]
    metadata: Dict[str, Any]
    posteriors: Optional[np.ndarray] = None


@dataclass
class EvaluationResult:
    mof: float
    edit: float
    f1: Dict[str, float]
    confusion: Dict[str, int]
    n_frames: int
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class SegmentEvaluationResult:
    maed_contact_sec: float
    maed_non_contact_sec: float
    msde_contact_sec: float
    msde_non_contact_sec: float
    error_counts_contact: Dict[str, int]
    error_counts_non_contact: Dict[str, int]
    n_contact_segments: int
    n_non_contact_segments: int
    fps: float
    artifacts: Dict[str, str] = field(default_factory=dict)
