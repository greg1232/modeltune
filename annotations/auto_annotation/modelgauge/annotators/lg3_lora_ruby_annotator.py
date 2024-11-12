import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.llama_guard_3_lora_annotator import (
    LlamaGuard3LoRAAnnotator,
    LlamaGuard3LoRAConfig,
)
from modelgauge.annotators.llama_guard_annotator import SafetyCategory

LOCAL_MODELGAUGE_ANNOTATOR_NAME = "llama-guard-3-lora-ruby-local"
LOCAL_DEPLOYED_VLLM_ENDPOINT_URL = os.getenv(
    "DEPLOYED_VLLM_ENDPOINT_URL", "http://localhost:8000/v1"
)
LOCAL_DEPLOYED_LORA_MODEL_NAME = "llama-guard-3-lora"

local_config = LlamaGuard3LoRAConfig(
    name=LOCAL_MODELGAUGE_ANNOTATOR_NAME,
    model=LOCAL_DEPLOYED_LORA_MODEL_NAME,
    base_url=LOCAL_DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_API_KEY", os.getenv("VLLM_LOCAL_API_KEY", "")),
)

PROD_MODELGAUGE_ANNOTATOR_NAME = "llama-guard-3-lora-ruby"
PROD_DEPLOYED_VLLM_ENDPOINT_URL = os.getenv(
    "DEPLOYED_VLLM_ENDPOINT_URL", "http://34.45.146.64:8000/v1"
)
PROD_DEPLOYED_LORA_MODEL_NAME = "llama-guard-3-lora"

prod_config = LlamaGuard3LoRAConfig(
    name=PROD_MODELGAUGE_ANNOTATOR_NAME,
    model=PROD_DEPLOYED_LORA_MODEL_NAME,
    base_url=PROD_DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_API_KEY", os.getenv("VLLM_LOCAL_API_KEY", "")),
)

ANNOTATORS.register(LlamaGuard3LoRAAnnotator, local_config.name, local_config)
ANNOTATORS.register(LlamaGuard3LoRAAnnotator, prod_config.name, prod_config)
