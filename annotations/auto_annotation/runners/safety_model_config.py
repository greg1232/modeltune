import os
from typing import List, Optional

import yaml
from pydantic import BaseModel

from modelgauge.ensemble.ensemble_strategy import EnsembleStrategies
from modelgauge.ensemble.majority_vote_ensemble_strategy import (
    MajorityVoteSafeTiesEnsembleStrategy,
    MajorityVoteUnsafeTiesEnsembleStrategy,
)
from schemas.safety_datasets import AnnotationInputDataset, AnnotationRun

TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
HF_TOKEN_ENV = "HF_TOKEN"
VLLM_API_KEY_ENV = "VLLM_API_KEY"

SUPPORTED_SAFETY_MODELS = [
    "llama_guard_2",
    "llama_guard_3",
    "llama_guard_3_20241003_taxonomy",
    "wildguard",
    "llama-3-70b.pe.tamalpais.2024-09-09",
    "llama-3-70b.pe.tamalpais.fr.2024-09-09",
    "mistral-8x22b.pe.tamalpais.2024-09-09",
    "mistral-8x22b.pe.tamalpais.fr.2024-09-09",
    "llama-3-70b.pe.whitney.2024-10-17",
    "mistral-8x22b.pe.whitney.2024-10-17",
    "llama-guard-3-lora-ruby-local",
    "llama-guard-3-lora-ruby",
    "mistral-7b-ruby",
    "mistral-7b-ruby-local",
]

REQUIRES_TOGETHER_API_KEY_ENV = [
    "llama_guard_3",
    "llama_guard_3_20241003_taxonomy",
    "llama-3-70b.pe.tamalpais.2024-09-09",
    "llama-3-70b.pe.tamalpais.fr.2024-09-09",
    "mistral-8x22b.pe.tamalpais.2024-09-09",
    "mistral-8x22b.pe.tamalpais.fr.2024-09-09",
    "llama-3-70b.pe.whitney.2024-10-17",
    "mistral-8x22b.pe.whitney.2024-10-17",
]

REQUIRES_HF_TOKEN_ENV = [
    "wildguard",
]

REQUIRES_VLLM_API_KEY = [
    "llama-guard-3-lora-ruby-local",
    "llama-guard-3-lora-ruby",
    "mistral-7b-ruby-local",
    "mistral-7b-ruby",
]


class EnsembleConfig(BaseModel):
    safety_models: List[str]
    join_strategy: str

    def model_post_init(self, __context):
        supported_ensemble_join_strategies = [
            strategy.value for strategy in EnsembleStrategies
        ]
        if self.join_strategy not in supported_ensemble_join_strategies:
            raise ValueError(
                f"Ensemble join strategy is not supported: {self.join_strategy}. Supported strategies: {supported_ensemble_join_strategies}"
            )

    def get_join_strategy(self):
        if self.join_strategy == EnsembleStrategies.MAJORITY_TIE_SAFE.value:
            return MajorityVoteSafeTiesEnsembleStrategy()
        elif self.join_strategy == EnsembleStrategies.MAJORITY_TIE_UNSAFE.value:
            return MajorityVoteUnsafeTiesEnsembleStrategy()
        else:
            raise ValueError(f"Unsupported join strategy: {self.join_strategy}")


class SafetyModel(BaseModel):
    name: str
    runfile: Optional[str] = None
    ensemble: Optional[EnsembleConfig] = None

    def runfile_is_complete(self, input_dataset: AnnotationInputDataset):
        """Checks that the runfile is complete"""
        if not self.runfile:
            raise ValueError(
                f"Safety model configuration doesn't specify a runfile for {self.name}"
            )

        run = AnnotationRun.from_jsonl(self.runfile)

        return run.matches_annotation_input_dataset(input_dataset)

    def runfile_uids_in_input_dataset(
        self, input_dataset: AnnotationInputDataset
    ) -> bool:
        """Check if any UIDs in run are not in the input dataset"""
        if not self.runfile:
            raise ValueError(
                f"Safety model configuration doesn't specify a runfile for {self.name}"
            )
        run = AnnotationRun.from_jsonl(self.runfile)

        uids_not_in_input_dataset = set(run.uids()) - set(input_dataset.uids())

        return len(uids_not_in_input_dataset) == 0

    def get_missing_samples_from_runfile(
        self, input_dataset: AnnotationInputDataset
    ) -> set[str]:
        """Returns samples from input dataset that are not yet completed in the runfile. Will raise error if a UID not in input_dataset is found"""
        if not self.runfile:
            raise ValueError(
                f"Safety model configuration doesn't specify a runfile for {self.name}"
            )
        run = AnnotationRun.from_jsonl(self.runfile)

        uids_not_yet_run = set(input_dataset.uids()) - set(run.uids())

        return uids_not_yet_run

    def validate_model(self):
        models_to_check = self.ensemble.safety_models if self.ensemble else [self.name]

        if self.runfile:
            if not os.path.isfile(self.runfile):
                raise ValueError(
                    f"Runfile {self.runfile} does not exist for model {self.name}. Please check filepath again"
                )

            # Try loading the runfile
            run = AnnotationRun.from_jsonl(self.runfile)

            # TODO verify runfile safety model name matches safety model name specified in config... at least throw a warning.
            pass

        for m in models_to_check:
            # Verify if supported
            if m not in SUPPORTED_SAFETY_MODELS:
                raise ValueError(f"Model name {m} is not recognized and/or supported")

            # Verify required env vars exist
            if (
                os.getenv(TOGETHER_API_KEY_ENV) is None
                and m in REQUIRES_TOGETHER_API_KEY_ENV
            ):
                raise ValueError(
                    f"Model: {m} requires env var ${TOGETHER_API_KEY_ENV} to be set. Please configure your env var."
                )
            if os.getenv(HF_TOKEN_ENV) is None and m in REQUIRES_HF_TOKEN_ENV:
                raise ValueError(
                    f"Model: {m} requires env var ${HF_TOKEN_ENV} to be set. Please configure your env var."
                )
            if os.getenv(VLLM_API_KEY_ENV) is None and m in REQUIRES_VLLM_API_KEY:
                raise ValueError(
                    f"Model: {m} requires env var ${VLLM_API_KEY_ENV} to be set. Please configure your env var."
                )


class SafetyModelConfig(BaseModel):
    safety_models: List[SafetyModel]

    def model_post_init(self, __context):
        if len(self.safety_models) < 1:
            raise ValueError("Must supply at least 1 safety model")

        for model in self.safety_models:
            model.validate_model()

    @classmethod
    def from_yml(cls, filepath: str):
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        safety_models = [SafetyModel(**model) for model in config_data["safety_models"]]
        return cls(safety_models=safety_models)
