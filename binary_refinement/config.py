from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class DurationPriorConfig:
    contact_mean_sec: float
    non_contact_mean_sec: float
    contact_sigma_sec: float
    non_contact_sigma_sec: float
    edge_contact_sigma_sec: float
    edge_non_contact_sigma_sec: float
    fps: float

    def __post_init__(self) -> None:
        for key, value in (
            ("contact_mean_sec", self.contact_mean_sec),
            ("non_contact_mean_sec", self.non_contact_mean_sec),
            ("contact_sigma_sec", self.contact_sigma_sec),
            ("non_contact_sigma_sec", self.non_contact_sigma_sec),
            ("edge_contact_sigma_sec", self.edge_contact_sigma_sec),
            ("edge_non_contact_sigma_sec", self.edge_non_contact_sigma_sec),
            ("fps", self.fps),
        ):
            if value <= 0:
                raise ValueError(f"{key} must be > 0; got {value}")


@dataclass
class TransitionDPConfig:
    k_transitions: int
    duration_prior: DurationPriorConfig
    observation_weight: float = 1.0
    duration_weight: float = 1.0
    eps: float = 1e-6
    start_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.k_transitions < 0:
            raise ValueError("k_transitions must be >= 0")
        if self.observation_weight < 0:
            raise ValueError("observation_weight must be >= 0")
        if self.duration_weight < 0:
            raise ValueError("duration_weight must be >= 0")
        if not (0 < self.eps < 0.5):
            raise ValueError("eps must satisfy 0 < eps < 0.5")
        if self.start_state is not None and self.start_state not in (0, 1):
            raise ValueError("start_state must be None, 0, or 1")


@dataclass
class HSMMKSegmentsConfig:
    alpha_non_contact: float
    alpha_contact: float
    fpr: float
    fnr: float
    num_trials: Optional[int] = None
    start_state: int = 0
    end_state: Optional[int] = None
    fps: Optional[float] = None
    lambda_non_contact_per_sec: Optional[float] = None
    lambda_contact_per_sec: Optional[float] = None
    k_segments: Optional[int] = None
    lambda_non_contact: Optional[float] = None
    lambda_contact: Optional[float] = None
    duration_weight: float = 1.0
    emission_weight: float = 1.0
    max_segment_length_frames: Optional[int] = 540
    numba_mode: str = "auto"
    eps: float = 1e-12

    @staticmethod
    def derive_k_segments(num_trials: int, start_state: int, end_state: int) -> int:
        if start_state == 1 and end_state == 1:
            return (2 * int(num_trials)) - 1
        if start_state == 0 and end_state == 0:
            return (2 * int(num_trials)) + 1
        return 2 * int(num_trials)

    def __post_init__(self) -> None:
        if self.start_state not in (0, 1):
            raise ValueError("start_state must be 0 or 1")
        if self.end_state is not None and self.end_state not in (0, 1):
            raise ValueError("end_state must be None, 0, or 1")
        if self.fps is not None and float(self.fps) <= 0:
            raise ValueError("fps must be > 0 when provided")
        if self.num_trials is not None and int(self.num_trials) < 1:
            raise ValueError("num_trials must be >= 1")
        if self.num_trials is None and self.end_state is not None:
            raise ValueError("end_state requires num_trials")
        if self.num_trials is not None and self.end_state is None:
            raise ValueError("end_state must be provided with num_trials")

        k_from_triplet: Optional[int] = None
        if self.num_trials is not None:
            num_trials = int(self.num_trials)
            end_state = int(self.end_state)
            k_from_triplet = int(
                self.derive_k_segments(
                    num_trials=num_trials,
                    start_state=int(self.start_state),
                    end_state=end_state,
                )
            )
            self.num_trials = num_trials
            self.end_state = end_state

        if self.k_segments is None:
            if k_from_triplet is None:
                raise ValueError("Provide k_segments or (num_trials with start_state/end_state)")
            self.k_segments = int(k_from_triplet)
        else:
            self.k_segments = int(self.k_segments)
            if self.k_segments < 1:
                raise ValueError("k_segments must be >= 1")
            if k_from_triplet is not None and self.k_segments != k_from_triplet:
                raise ValueError(
                    "k_segments mismatch: "
                    f"provided {self.k_segments}, derived {k_from_triplet} from "
                    f"(num_trials={self.num_trials}, start_state={self.start_state}, end_state={self.end_state})"
                )

        if self.num_trials is None:
            if self.end_state is None:
                self.end_state = int(self.start_state if (self.k_segments % 2 == 1) else (1 - self.start_state))
            else:
                expected_end = int(self.start_state if (self.k_segments % 2 == 1) else (1 - self.start_state))
                if int(self.end_state) != expected_end:
                    raise ValueError(
                        "end_state is inconsistent with k_segments/start_state under alternating segments: "
                        f"k_segments={self.k_segments}, start_state={self.start_state}, expected_end_state={expected_end}"
                    )
            if int(self.start_state) == 1:
                self.num_trials = int((self.k_segments + 1) // 2)
            else:
                self.num_trials = int(self.k_segments // 2)
        if int(self.num_trials) < 1:
            raise ValueError(
                "num_trials must be >= 1; this configuration implies zero trial segments."
            )

        has_legacy_lambda = (self.lambda_non_contact is not None) or (self.lambda_contact is not None)
        has_new_lambda = (self.lambda_non_contact_per_sec is not None) or (self.lambda_contact_per_sec is not None)
        if has_legacy_lambda:
            if self.lambda_non_contact is None or self.lambda_contact is None:
                raise ValueError("Provide both lambda_non_contact and lambda_contact for legacy API")
            if float(self.lambda_non_contact) <= 0 or float(self.lambda_contact) <= 0:
                raise ValueError("lambda_non_contact and lambda_contact must be > 0")
        if has_new_lambda:
            if self.lambda_non_contact_per_sec is None or self.lambda_contact_per_sec is None:
                raise ValueError(
                    "Provide both lambda_non_contact_per_sec and lambda_contact_per_sec"
                )
            if float(self.lambda_non_contact_per_sec) <= 0 or float(self.lambda_contact_per_sec) <= 0:
                raise ValueError("lambda_non_contact_per_sec and lambda_contact_per_sec must be > 0")
            if self.fps is None or float(self.fps) <= 0:
                raise ValueError("fps must be provided and > 0 when using *_per_sec lambda priors")

        if not has_legacy_lambda and not has_new_lambda:
            raise ValueError(
                "Provide lambda priors via either legacy frame-scale fields "
                "(lambda_non_contact/lambda_contact) or seconds-scale fields "
                "(lambda_non_contact_per_sec/lambda_contact_per_sec with fps)."
            )

        fps_resolved = float(self.fps) if self.fps is not None else 1.0
        if has_new_lambda:
            lambda_non_contact_per_sec = float(self.lambda_non_contact_per_sec)
            lambda_contact_per_sec = float(self.lambda_contact_per_sec)
            lambda_non_contact_per_frame = lambda_non_contact_per_sec / fps_resolved
            lambda_contact_per_frame = lambda_contact_per_sec / fps_resolved
            if has_legacy_lambda:
                legacy_nc = float(self.lambda_non_contact)
                legacy_c = float(self.lambda_contact)
                if not math.isclose(legacy_nc, lambda_non_contact_per_frame, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(
                        "lambda_non_contact mismatch between legacy and seconds-scale priors: "
                        f"legacy={legacy_nc}, from_seconds={lambda_non_contact_per_frame}"
                    )
                if not math.isclose(legacy_c, lambda_contact_per_frame, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(
                        "lambda_contact mismatch between legacy and seconds-scale priors: "
                        f"legacy={legacy_c}, from_seconds={lambda_contact_per_frame}"
                    )
        else:
            lambda_non_contact_per_frame = float(self.lambda_non_contact)
            lambda_contact_per_frame = float(self.lambda_contact)
            lambda_non_contact_per_sec = lambda_non_contact_per_frame * fps_resolved
            lambda_contact_per_sec = lambda_contact_per_frame * fps_resolved
            self.lambda_non_contact_per_sec = lambda_non_contact_per_sec
            self.lambda_contact_per_sec = lambda_contact_per_sec

        for key, value in (
            ("alpha_non_contact", self.alpha_non_contact),
            ("alpha_contact", self.alpha_contact),
            ("lambda_non_contact_per_sec", self.lambda_non_contact_per_sec),
            ("lambda_contact_per_sec", self.lambda_contact_per_sec),
        ):
            if value <= 0:
                raise ValueError(f"{key} must be > 0; got {value}")
        for key, value in (("fpr", self.fpr), ("fnr", self.fnr)):
            if not (0.0 < value < 1.0):
                raise ValueError(f"{key} must satisfy 0 < {key} < 1; got {value}")
        if self.duration_weight < 0:
            raise ValueError("duration_weight must be >= 0")
        if self.emission_weight < 0:
            raise ValueError("emission_weight must be >= 0")
        if self.max_segment_length_frames is not None:
            if int(self.max_segment_length_frames) < 1:
                raise ValueError("max_segment_length_frames must be >= 1 when provided")
            self.max_segment_length_frames = int(self.max_segment_length_frames)
        self.numba_mode = str(self.numba_mode).strip().lower()
        if self.numba_mode not in {"auto", "on", "off"}:
            raise ValueError("numba_mode must be one of: 'auto', 'on', 'off'")
        if not (0 < self.eps < 0.5):
            raise ValueError("eps must satisfy 0 < eps < 0.5")

        self.fps = float(fps_resolved)
        self._k_segments_resolved = int(self.k_segments)
        self._lambda_non_contact_per_frame = float(lambda_non_contact_per_frame)
        self._lambda_contact_per_frame = float(lambda_contact_per_frame)

    @property
    def resolved_k_segments(self) -> int:
        return int(self._k_segments_resolved)

    @property
    def lambda_non_contact_per_frame(self) -> float:
        return float(self._lambda_non_contact_per_frame)

    @property
    def lambda_contact_per_frame(self) -> float:
        return float(self._lambda_contact_per_frame)
