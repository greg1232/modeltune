import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.llama_guard_3_lora_annotator import (
    LlamaGuard3LoRAAnnotator,
    LlamaGuard3LoRAConfig,
)

LOCAL_MODELGAUGE_ANNOTATOR_NAME = "mistral-7b-ruby-local"
LOCAL_DEPLOYED_VLLM_ENDPOINT_URL = os.getenv(
    "DEPLOYED_VLLM_ENDPOINT_URL", "http://localhost:8000/v1"
)
LOCAL_DEPLOYED_LORA_MODEL_NAME = "mistral7b-lora"

# NOTE the mistral annotator purposely uses LlamaGuard3LoRA logic because the
# input and output formats are (currently) the same despite the underlying
# backbone having changed
local_config = LlamaGuard3LoRAConfig(
    name=LOCAL_MODELGAUGE_ANNOTATOR_NAME,
    model=LOCAL_DEPLOYED_LORA_MODEL_NAME,
    base_url=LOCAL_DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_API_KEY", os.getenv("VLLM_LOCAL_API_KEY", "")),
)

ANNOTATORS.register(LlamaGuard3LoRAAnnotator, local_config.name, local_config)
