import json
import os
from typing import Callable, Dict, Optional

from modelgauge.annotator import CompletionAnnotator
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotators.prompt_engineered_annotator import (
    OpenAIModelConfig,
    PromptEngineeredAnnotator,
    PromptEngineeredAnnotatorConfig,
)
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.prompts.labeler_v1 import LABELER_SYS_PROMPT_V1, LABELER_USER_PROMPT_V1
from modelgauge.safety_model_constants import ASSISTANT_ROLE
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.safety_model_utils import flatten_conversation_to_str
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.suts.openai_client import (
    OpenAIApiKey,
    OpenAIChat,
    OpenAIChatRequest,
    OpenAIOrgId,
)
from modelgauge.taxonomy.v0_5_2024_04_12_taxonomy import V0_5_2024_04_12_TAXONOMY
from modelgauge.taxonomy.v1_0_2024_09_09_taxonomy import V1_0_2024_09_09_TAXONOMY

GPT_4_PE_SUTRO_2024_08_01_CONFIG = PromptEngineeredAnnotatorConfig(
    name="gpt-4.pe.sutro.2024-08-01",
    description="First gpt4 model, using the v0.5 April 2024 taxonomy",
    llm_config=OpenAIModelConfig(
        path="gpt-4",
        api_key=OpenAIApiKey(os.environ.get("OPENAI_API_KEY")),
        org_id=OpenAIOrgId(os.environ.get("OPENAI_ORG_ID")),
    ),
    labeler_system_prompt=LABELER_SYS_PROMPT_V1,
    labeler_user_prompt=LABELER_USER_PROMPT_V1,
    taxonomy=V0_5_2024_04_12_TAXONOMY,
    role_to_measure=ASSISTANT_ROLE,
)


ANNOTATORS.register(
    PromptEngineeredAnnotator,
    GPT_4_PE_SUTRO_2024_08_01_CONFIG.name,
    GPT_4_PE_SUTRO_2024_08_01_CONFIG,
)
