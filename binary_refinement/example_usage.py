from __future__ import annotations

import numpy as np

from binary_refinement.config import DurationPriorConfig, TransitionDPConfig
from binary_refinement.evaluator import evaluate_strategy
from binary_refinement.identity import IdentityRefiner
from binary_refinement.transition_dp import TransitionConstrainedDPRefiner


def _make_demo_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    gt = np.array([0] * 8 + [1] * 12 + [0] * 9 + [1] * 11 + [0] * 10, dtype=int)
    probs = np.where(gt == 1, 0.85, 0.15) + rng.normal(loc=0.0, scale=0.1, size=gt.shape[0])
    probs = np.clip(probs, 0.01, 0.99)
    return gt, probs


def main() -> None:
    gt, obs = _make_demo_data(seed=7)

    duration_prior = DurationPriorConfig(
        contact_mean_sec=0.18,
        non_contact_mean_sec=0.15,
        contact_sigma_sec=0.04,
        non_contact_sigma_sec=0.04,
        edge_contact_sigma_sec=0.12,
        edge_non_contact_sigma_sec=0.12,
        fps=60.0,
    )
    dp_cfg = TransitionDPConfig(
        k_transitions=4,
        duration_prior=duration_prior,
        observation_weight=1.0,
        duration_weight=1.0,
        start_state=None,
    )

    methods = [
        IdentityRefiner(),
        TransitionConstrainedDPRefiner(dp_cfg),
    ]

    for method in methods:
        _, metrics = evaluate_strategy(method, observations=obs, ground_truth=gt)
        print(
            f"{method.name}: MoF={metrics.mof:.3f}, "
            f"Edit={metrics.edit:.2f}, "
            + ", ".join(f"{k}={v:.2f}" for k, v in metrics.f1.items())
        )


if __name__ == "__main__":
    main()
