from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_refinement.identity import IdentityRefiner


def test_identity_refiner_threshold_and_transitions():
    obs = np.array([0.1, 0.2, 0.7, 0.9, 0.4], dtype=float)
    result = IdentityRefiner().predict(obs, threshold=0.5)

    assert result.sequence.tolist() == [0, 0, 1, 1, 0]
    assert result.num_transitions == 2
    assert result.objective is None
