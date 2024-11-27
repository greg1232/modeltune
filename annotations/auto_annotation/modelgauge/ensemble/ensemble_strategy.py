from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)


class EnsembleStrategies(Enum):
    MAJORITY_TIE_SAFE = "majority_vote_tie_safe"
    MAJORITY_TIE_UNSAFE = "majority_vote_tie_unsafe"


class EnsembleStrategy(ABC):

    @abstractmethod
    def compute_response(
        self, annotations: Dict[str, SafetyModelResponse]
    ) -> EnsembleSafetyModelResponse:
        pass
