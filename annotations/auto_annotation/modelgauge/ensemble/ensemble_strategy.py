from abc import ABC, abstractmethod
from typing import Dict

from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)


class EnsembleStrategy(ABC):

    @abstractmethod
    def compute_response(
        self, annotations: Dict[str, SafetyModelResponse]
    ) -> EnsembleSafetyModelResponse:
        pass
