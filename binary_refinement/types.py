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


@dataclass
class EvaluationResult:
    mof: float
    edit: float
    f1: Dict[str, float]
    confusion: Dict[str, int]
    n_frames: int
    artifacts: Dict[str, str] = field(default_factory=dict)
