from __future__ import annotations

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
