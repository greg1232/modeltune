from typing import Dict

from modelgauge.ensemble.ensemble_strategy import EnsembleStrategy
from modelgauge.safety_model_response import SafetyModelResponse


class Ensemble:
    def __init__(self, strategy: EnsembleStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: EnsembleStrategy):
        self._strategy = strategy

    def compute_response(self, annotations: Dict[str, SafetyModelResponse]):
        return self._strategy.compute_response(annotations)
