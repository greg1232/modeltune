import os
from typing import List

import yaml
from pydantic import BaseModel

TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
HF_TOKEN_ENV = "HF_TOKEN"
VLLM_API_KEY_ENV = "VLLM_API_KEY"

SUPPORTED_SAFETY_MODELS = [
    "llama_guard_2",
    "llama_guard_3",
    "wildguard",
    "llama-3-70b.pe.tamalpais.2024-09-09",
    "mistral-8x22b.pe.tamalpais.2024-09-09",
    "llama-3-70b.pe.whitney.2024-10-17",
    "mistral-8x22b.pe.whitney.2024-10-17",
]

REQUIRES_TOGETHER_API_KEY_ENV = [
    "llama_guard_3",
    "llama-3-70b.pe.tamalpais.2024-09-09",
    "mistral-8x22b.pe.tamalpais.2024-09-09",
    "llama-3-70b.pe.whitney.2024-10-17",
    "mistral-8x22b.pe.whitney.2024-10-17",
]

REQUIRES_HF_TOKEN_ENV = [
    "wildguard",
]

REQUIRES_VLLM_API_KEY = []


class SafetyModelConfig(BaseModel):
    safety_models: List[str]

    def model_post_init(self, __context):
        if len(self.safety_models) < 1:
            raise ValueError("Must supply at least 1 safety model")

        for model_name in self.safety_models:
            # Verify if supported
            if model_name not in SUPPORTED_SAFETY_MODELS:
                raise ValueError(
                    f"Model name {model_name} is not recognized and/or supported"
                )

            # Verify required env vars exist
            if (
                os.getenv(TOGETHER_API_KEY_ENV) == None
                and model_name in REQUIRES_TOGETHER_API_KEY_ENV
            ):
                raise ValueError(
                    f"Model: {model_name} requires env var ${TOGETHER_API_KEY_ENV} to be set. Please configure your env var."
                )
            if os.getenv(HF_TOKEN_ENV) == None and model_name in REQUIRES_HF_TOKEN_ENV:
                raise ValueError(
                    f"Model: {model_name} requires env var ${HF_TOKEN_ENV} to be set. Please configure your env var."
                )
            if (
                os.getenv(VLLM_API_KEY_ENV) == None
                and model_name in REQUIRES_VLLM_API_KEY
            ):
                raise ValueError(
                    f"Model: {model_name} requires env var ${VLLM_API_KEY_ENV} to be set. Please configure your env var."
                )

    @classmethod
    def from_yml(cls, filepath: str):
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)
