import os

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.together_pe_annotator import (
    TogetherModelConfig,
    TogetherPEAnnotator,
    TogetherPEAnnotatorConfig,
)
from modelgauge.safety_model_constants import ASSISTANT_ROLE
from modelgauge.safety_model_prompts import (
    LABELER_SYS_PROMPT,
    LABELER_USER_PROMPT,
    MLC_TAXONOMY_0_5,
)
from modelgauge.suts.together_client import TogetherApiKey

MISTRAL_8x22B_CONFIG = TogetherPEAnnotatorConfig(
    name="mistral_8x22b",
    llm_config=TogetherModelConfig(
        path="mistralai/Mixtral-8x22B-Instruct-v0.1",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT,
    labeler_user_prompt=LABELER_USER_PROMPT,
    taxonomy=MLC_TAXONOMY_0_5,
    role_to_measure=ASSISTANT_ROLE,
)

ANNOTATORS.register(
    TogetherPEAnnotator, MISTRAL_8x22B_CONFIG.name, MISTRAL_8x22B_CONFIG
)
