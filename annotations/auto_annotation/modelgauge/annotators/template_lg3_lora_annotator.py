import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.llama_guard_3_lora_annotator import (
    LlamaGuard3LoRAAnnotator,
    LlamaGuard3LoRAConfig,
)
from modelgauge.annotators.llama_guard_annotator import SafetyCategory

# Arbitrary name to register annotator with modelgauge
MODELGAUGE_ANNOTATOR_NAME = "sample-lg3-lora"
# Update the following endpoint to point to the deployed vllm
DEPLOYED_VLLM_ENDPOINT_URL = os.getenv(
    "DEPLOYED_VLLM_ENDPOINT_URL", "http://localhost:8000/v1"
)
# This value is NOT arbitrary. The docker image sets this value
DEPLOYED_LORA_MODEL_NAME = "llama-guard-3-lora"
# Define Llama Guard 3 style safety categories

config = LlamaGuard3LoRAConfig(
    name=MODELGAUGE_ANNOTATOR_NAME,
    model=DEPLOYED_LORA_MODEL_NAME,
    base_url=DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_API_KEY", os.getenv("VLLM_LOCAL_API_KEY", "")),
)

ANNOTATORS.register(LlamaGuard3LoRAAnnotator, config.name, config)
