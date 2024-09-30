import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.ensemble_annotator import EnsembleAnnotator
from modelgauge.annotators.llama_3_70b_chat_annotator import (
    LLAMA_3_70B_PE_SUTRO_2024_07_01_CONFIG,
)
from modelgauge.annotators.llama_guard_2_lora_annotator import (
    LlamaGuard2LoRAAnnotator,
    LlamaGuard2LoRAConfig,
)
from modelgauge.annotators.llama_guard_annotator import LlamaGuardAnnotator
from modelgauge.annotators.mistral_8x22b_instruct_annotator import (
    MISTRAL_8x22B_PE_SUTRO_2024_07_01_CONFIG,
)
from modelgauge.annotators.prompt_engineered_annotator import PromptEngineeredAnnotator
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole
from modelgauge.safety_model_constants import DEFAULT_MINIMAL_SAFETY_CATEGORIES

# Update the following endpoint to point to the deployed vllm
DEPLOYED_VLLM_ENDPOINT_URL = "http://evaluator.dev.modelmodel.org:8000/v1"
# This value is NOT arbitrary. The docker image sets this value
DEPLOYED_LORA_MODEL_NAME = "llamaguard-lora"

vllm_config = LlamaGuard2LoRAConfig(
    name="vllm",
    model=DEPLOYED_LORA_MODEL_NAME,
    base_url=DEPLOYED_VLLM_ENDPOINT_URL,
    api_key=os.getenv("VLLM_LOCAL_API_KEY", ""),
    safety_categories=DEFAULT_MINIMAL_SAFETY_CATEGORIES,
)

a1 = PromptEngineeredAnnotator(
    MISTRAL_8x22B_PE_SUTRO_2024_07_01_CONFIG.name,
    MISTRAL_8x22B_PE_SUTRO_2024_07_01_CONFIG,
)
a2 = PromptEngineeredAnnotator(
    LLAMA_3_70B_PE_SUTRO_2024_07_01_CONFIG.name, LLAMA_3_70B_PE_SUTRO_2024_07_01_CONFIG
)
a3 = LlamaGuard2LoRAAnnotator(vllm_config.name, vllm_config)

ANNOTATORS.register(EnsembleAnnotator, "ensemble", [a1, a2, a3])
