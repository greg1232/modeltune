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
from modelgauge.taxonomy.v1_0_2024_09_09_taxonomy import V1_0_2024_09_09_TAXONOMY

GEMMA_2_27B_PE_TAMALPAIS_2024_09_17_CONFIG = PromptEngineeredAnnotatorConfig(
    name="gemma-2-27b.pe.tamalpais.2024-09-17",
    description="Gemma 2 27b model using v1.0 September 2024 taxonomy",
    llm_config=TogetherModelConfig(
        path="google/gemma-2-27b-it",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT_V1,
    labeler_user_prompt=LABELER_USER_PROMPT_V1,
    taxonomy=V1_0_2024_09_09_TAXONOMY,
    role_to_measure=ASSISTANT_ROLE,
)

ANNOTATORS.register(
    PromptEngineeredAnnotator,
    GEMMA_2_27B_PE_TAMALPAIS_2024_09_17_CONFIG.name,
    GEMMA_2_27B_PE_TAMALPAIS_2024_09_17_CONFIG,
)
