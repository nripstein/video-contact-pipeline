from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _NUMBA_AVAILABLE = False


def numba_is_available() -> bool:
    return bool(_NUMBA_AVAILABLE)


def _state_for_segment(seg_idx: int, start_state: int) -> int:
    return start_state if (seg_idx % 2 == 0) else (1 - start_state)


def _bounded_t_range(m: int, k: int, n: int, max_segment_length: int) -> Tuple[int, int]:
    min_t = m
    max_t = n - (k - m)
    if max_segment_length > 0:
        min_t = max(min_t, n - (k - m) * max_segment_length)
        max_t = min(max_t, m * max_segment_length)
    return int(min_t), int(max_t)


def fill_segment_python(
    *,
    m: int,
    k: int,
    n: int,
    start_state: int,
    emission_weight: float,
    duration_weight: float,
    max_segment_length: int,
    dp_prev: np.ndarray,
    dp_curr: np.ndarray,
    back_row: np.ndarray,
    ll0_prefix: np.ndarray,
    ll1_prefix: np.ndarray,
    dur_ll_state0: np.ndarray,
    dur_ll_state1: np.ndarray,
) -> None:
    neg_inf = -float("inf")
    dp_curr.fill(neg_inf)
    back_row.fill(-1)

    state = _state_for_segment(m - 1, start_state)
    min_t, max_t = _bounded_t_range(m, k, n, max_segment_length)
    if min_t > max_t:
        return

    for t in range(min_t, max_t + 1):
        u_min = m - 1
        u_max = t - 1
        if max_segment_length > 0:
            u_min = max(u_min, t - max_segment_length)
            u_max = min(u_max, (m - 1) * max_segment_length)
        if u_min > u_max:
            continue

        best_score = neg_inf
        best_u = -1
        for u in range(u_min, u_max + 1):
            prev = float(dp_prev[u])
            if not np.isfinite(prev):
                continue
            d = t - u
            emit_ll = float(ll0_prefix[t] - ll0_prefix[u]) if state == 0 else float(ll1_prefix[t] - ll1_prefix[u])
            dur_ll = float(dur_ll_state0[d]) if state == 0 else float(dur_ll_state1[d])
            score = prev + (emission_weight * emit_ll) + (duration_weight * dur_ll)
            if score > best_score:
                best_score = score
                best_u = u
        dp_curr[t] = best_score
        back_row[t] = int(best_u)


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _fill_segment_numba_impl(
        m: int,
        k: int,
        n: int,
        start_state: int,
        emission_weight: float,
        duration_weight: float,
        max_segment_length: int,
        dp_prev: np.ndarray,
        dp_curr: np.ndarray,
        back_row: np.ndarray,
        ll0_prefix: np.ndarray,
        ll1_prefix: np.ndarray,
        dur_ll_state0: np.ndarray,
        dur_ll_state1: np.ndarray,
    ) -> None:
        neg_inf = -np.inf
        dp_curr.fill(neg_inf)
        back_row.fill(-1)

        state = start_state if (((m - 1) % 2) == 0) else (1 - start_state)
        min_t = m
        max_t = n - (k - m)
        if max_segment_length > 0:
            lhs = n - (k - m) * max_segment_length
            rhs = m * max_segment_length
            if lhs > min_t:
                min_t = lhs
            if rhs < max_t:
                max_t = rhs
        if min_t > max_t:
            return

        for t in range(min_t, max_t + 1):
            u_min = m - 1
            u_max = t - 1
            if max_segment_length > 0:
                lhs = t - max_segment_length
                rhs = (m - 1) * max_segment_length
                if lhs > u_min:
                    u_min = lhs
                if rhs < u_max:
                    u_max = rhs
            if u_min > u_max:
                continue

            best_score = neg_inf
            best_u = -1
            for u in range(u_min, u_max + 1):
                prev = dp_prev[u]
                if prev == neg_inf:
                    continue
                d = t - u
                if state == 0:
                    emit_ll = ll0_prefix[t] - ll0_prefix[u]
                    dur_ll = dur_ll_state0[d]
                else:
                    emit_ll = ll1_prefix[t] - ll1_prefix[u]
                    dur_ll = dur_ll_state1[d]
                score = prev + (emission_weight * emit_ll) + (duration_weight * dur_ll)
                if score > best_score:
                    best_score = score
                    best_u = u
            dp_curr[t] = best_score
            back_row[t] = best_u

    def fill_segment_numba(
        *,
        m: int,
        k: int,
        n: int,
        start_state: int,
        emission_weight: float,
        duration_weight: float,
        max_segment_length: int,
        dp_prev: np.ndarray,
        dp_curr: np.ndarray,
        back_row: np.ndarray,
        ll0_prefix: np.ndarray,
        ll1_prefix: np.ndarray,
        dur_ll_state0: np.ndarray,
        dur_ll_state1: np.ndarray,
    ) -> None:
        _fill_segment_numba_impl(
            m,
            k,
            n,
            start_state,
            emission_weight,
            duration_weight,
            max_segment_length,
            dp_prev,
            dp_curr,
            back_row,
            ll0_prefix,
            ll1_prefix,
            dur_ll_state0,
            dur_ll_state1,
        )

else:

    def fill_segment_numba(
        m: int,
        k: int,
        n: int,
        start_state: int,
        emission_weight: float,
        duration_weight: float,
        max_segment_length: int,
        dp_prev: np.ndarray,
        dp_curr: np.ndarray,
        back_row: np.ndarray,
        ll0_prefix: np.ndarray,
        ll1_prefix: np.ndarray,
        dur_ll_state0: np.ndarray,
        dur_ll_state1: np.ndarray,
    ) -> None:
        raise RuntimeError("Numba is not available; cannot use numba backend.")
