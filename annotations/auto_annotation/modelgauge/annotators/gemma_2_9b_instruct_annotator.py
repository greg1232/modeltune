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

GEMMA_2_9B_PE_SUTRO_2024_08_01_CONFIG = PromptEngineeredAnnotatorConfig(
    name="gemma-2-9b.pe.sutro.2024-08-01",
    description="",
    llm_config=TogetherModelConfig(
        path="google/gemma-2-9b-it",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT_V1,
    labeler_user_prompt=LABELER_USER_PROMPT_V1,
    taxonomy=V0_5_2024_04_12_TAXONOMY,
    role_to_measure=ASSISTANT_ROLE,
)

ANNOTATORS.register(
    PromptEngineeredAnnotator,
    GEMMA_2_9B_PE_SUTRO_2024_08_01_CONFIG.name,
    GEMMA_2_9B_PE_SUTRO_2024_08_01_CONFIG,
)
