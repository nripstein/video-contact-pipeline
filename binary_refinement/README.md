# Binary Refinement

Tools for refining noisy binary frame-wise contact signals and evaluating refinement quality.

## What Is In This Module

- `IdentityRefiner`
  - Baseline strategy that thresholds/returns the input signal.
- `TransitionConstrainedDPRefiner`
  - Exact dynamic-programming strategy that enforces exactly `k` transitions and uses duration priors.
- `HSMMKSegmentsRefiner`
  - Exact segmental Viterbi for a left-to-right HSMM with exactly `k` segments, binary emissions (`FPR/FNR`), and Gamma duration priors.
- Standard evaluation utilities:
  - `MoF`, `Edit Score`, `F1@tau`, confusion counts.
  - Optional barcode artifact: `Ground Truth`, `Original`, `Refined`.

## Files

- `base.py`: abstract strategy interface (`BinaryRefinementStrategy`)
- `config.py`: config dataclasses (`DurationPriorConfig`, `TransitionDPConfig`, `HSMMKSegmentsConfig`)
- `identity.py`: baseline refiner
- `transition_dp.py`: exact transition-constrained DP refiner
- `hsmm_k_segments.py`: exact `k`-segment HSMM refiner
- `evaluator.py`: metrics + barcode artifact helpers
- `types.py`: `RefinementResult` and `EvaluationResult`
- `example_usage.py`: runnable demo

## Quickstart

From repo root:

```bash
python -m binary_refinement.example_usage
```

## Minimal Python Usage

```python
import numpy as np
from binary_refinement import (
    IdentityRefiner,
    HSMMKSegmentsConfig,
    HSMMKSegmentsRefiner,
    TransitionConstrainedDPRefiner,
    DurationPriorConfig,
    TransitionDPConfig,
    evaluate_strategy,
)

gt = np.array([0,0,0,1,1,1,0,0], dtype=int)
observations = np.array([0.1,0.2,0.3,0.9,0.8,0.7,0.2,0.1], dtype=float)

prior = DurationPriorConfig(
    contact_mean_sec=1.2,
    non_contact_mean_sec=1.0,
    contact_sigma_sec=0.35,
    non_contact_sigma_sec=0.35,
    edge_contact_sigma_sec=1.0,
    edge_non_contact_sigma_sec=1.0,
    fps=60.0,
)

dp_cfg = TransitionDPConfig(
    k_transitions=2,
    duration_prior=prior,
    observation_weight=1.0,
    duration_weight=1.0,
    start_state=None,  # tries both starts and picks lower objective
)

identity = IdentityRefiner()
dp = TransitionConstrainedDPRefiner(dp_cfg)
hsmm_cfg = HSMMKSegmentsConfig(
    k_segments=3,
    alpha_non_contact=40.0,
    lambda_non_contact=4.0,
    alpha_contact=48.0,
    lambda_contact=4.0,
    fpr=0.10,
    fnr=0.10,
    start_state=0,
)
hsmm = HSMMKSegmentsRefiner(hsmm_cfg)

_, id_metrics = evaluate_strategy(identity, observations=observations, ground_truth=gt)
_, dp_metrics = evaluate_strategy(dp, observations=observations, ground_truth=gt)
_, hsmm_metrics = evaluate_strategy(hsmm, observations=(observations >= 0.5).astype(int), ground_truth=gt)
```

## Evaluate + Save Barcode Comparison

```python
_, metrics = evaluate_strategy(
    dp,
    observations=observations,
    ground_truth=gt,
    save_barcode_path="results/binary_refinement/compare.png",
    # optional override for the "Original" row in the barcode:
    # original_signal=raw_unrefined_binary_or_probs
)

print(metrics.artifacts["barcode_comparison"])
```

The saved plot has three rows:

1. Ground Truth
2. Original
3. Refined

## Input Conventions

- Ground truth:
  - Binary `0/1` (`not_holding` -> `0`, `holding` -> `1` before calling evaluator).
- Observations:
  - Either binary signal or probabilities in `[0, 1]`.
  - The evaluator thresholds probabilities at `0.5` for metric computation.
  - `HSMMKSegmentsRefiner` requires binary observations (`{0,1}`).

## Complexity

- `IdentityRefiner`
  - Time: `O(n)`
  - Memory: `O(n)`
- `TransitionConstrainedDPRefiner` (exact DP)
  - Time: `O((k+1) * n^2)`
  - Memory: `O((k+1) * n)`
  - If `start_state=None`, runtime is roughly doubled.
- `HSMMKSegmentsRefiner` (exact segmental Viterbi)
  - Time: `O(k * n^2)`
  - Memory: `O(k * n)`

`n` is sequence length. For `TransitionConstrainedDPRefiner`, `k` denotes transitions; for `HSMMKSegmentsRefiner`, `k` denotes segments.

## Notes

- The exact DP method can be slow for very long sequences and large `k`.
- For long videos, start with smaller windows or clips when tuning priors.
