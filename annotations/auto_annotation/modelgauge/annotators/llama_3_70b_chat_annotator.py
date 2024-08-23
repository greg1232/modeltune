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

LLAMA_3_70B_CONFIG = TogetherPEAnnotatorConfig(
    name="llama_3_70b",
    llm_config=TogetherModelConfig(
        path="meta-llama/Llama-3-70b-chat-hf",
        api_key=TogetherApiKey(os.environ.get("TOGETHER_API_KEY")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT,
    labeler_user_prompt=LABELER_USER_PROMPT,
    taxonomy=MLC_TAXONOMY_0_5,
    role_to_measure=ASSISTANT_ROLE,
)

ANNOTATORS.register(TogetherPEAnnotator, LLAMA_3_70B_CONFIG.name, LLAMA_3_70B_CONFIG)
