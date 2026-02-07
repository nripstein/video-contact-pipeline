from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from binary_refinement.types import RefinementResult


class BinaryRefinementStrategy(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def predict(self, observations: np.ndarray, **kwargs) -> RefinementResult:
        raise NotImplementedError
