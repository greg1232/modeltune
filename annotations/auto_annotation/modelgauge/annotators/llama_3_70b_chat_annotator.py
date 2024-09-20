import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.prompt_engineered_annotator import (
    PromptEngineeredAnnotator,
    PromptEngineeredAnnotatorConfig,
    TogetherModelConfig,
)
from modelgauge.prompts.labeler_v1 import LABELER_SYS_PROMPT_V1, LABELER_USER_PROMPT_V1
from modelgauge.safety_model_constants import ASSISTANT_ROLE
from modelgauge.suts.together_client import TogetherApiKey
from modelgauge.taxonomy.v0_5_2024_04_12_taxonomy import V0_5_2024_04_12_TAXONOMY
from modelgauge.taxonomy.v1_0_2024_09_09_taxonomy import V1_0_2024_09_09_TAXONOMY

LLAMA_3_70B_PE_SUTRO_2024_07_01_CONFIG = PromptEngineeredAnnotatorConfig(
    name="llama-3-70b.pe.sutro.2024-07-01",
    description="First Llama 3 70b model, using the v0.5 April 2024 taxonomy",
    llm_config=TogetherModelConfig(
        path="meta-llama/Llama-3-70b-chat-hf",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT_V1,
    labeler_user_prompt=LABELER_USER_PROMPT_V1,
    taxonomy=V0_5_2024_04_12_TAXONOMY,
    role_to_measure=ASSISTANT_ROLE,
)

LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG = PromptEngineeredAnnotatorConfig(
    name="llama-3-70b.pe.tamalpais.2024-09-09",
    description="Llama 3 70b model, using the v1.0 September 2024 taxonomy",
    llm_config=TogetherModelConfig(
        path="meta-llama/Llama-3-70b-chat-hf",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT_V1,
    labeler_user_prompt=LABELER_USER_PROMPT_V1,
    taxonomy=V1_0_2024_09_09_TAXONOMY,
    role_to_measure=ASSISTANT_ROLE,
)

ANNOTATORS.register(
    PromptEngineeredAnnotator,
    LLAMA_3_70B_PE_SUTRO_2024_07_01_CONFIG.name,
    LLAMA_3_70B_PE_SUTRO_2024_07_01_CONFIG,
)
ANNOTATORS.register(
    PromptEngineeredAnnotator,
    LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG.name,
    LLAMA_3_70B_PE_TAMALPAIS_2024_09_09_CONFIG,
)
