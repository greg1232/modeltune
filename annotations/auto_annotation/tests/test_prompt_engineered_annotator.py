import json

from modelgauge.annotators.prompt_engineered_annotator import (
    PromptEngineeredAnnotator,
    PromptEngineeredAnnotatorConfig,
    TogetherModelConfig,
)
from modelgauge.prompt import ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.safety_model_constants import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE
from modelgauge.safety_model_response import SafetyModelResponse
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.suts.together_client import (
    TogetherApiKey,
    TogetherChatRequest,
    TogetherChatResponse,
)


def _fake_annotator(role_to_measure=ASSISTANT_ROLE):
    fake_config = PromptEngineeredAnnotatorConfig(
        name="test_annotator",
        description="",
        llm_config=TogetherModelConfig(
            path="fake/path",
            api_key=TogetherApiKey("fake_key"),
        ),
        labeler_system_prompt="fake prompt",
        labeler_user_prompt="fake prompt",
        taxonomy="fake taxonomy",
        role_to_measure=role_to_measure,
    )

    fake_annotator = PromptEngineeredAnnotator("_", fake_config)

    return fake_annotator


def _fake_request():
    request = TogetherChatRequest(
        model="fake_model_path",
        messages=[
            TogetherChatRequest.Message(role=SYSTEM_ROLE, content="fake system prompt"),
            TogetherChatRequest.Message(role=USER_ROLE, content="fake user prompt"),
        ],
        max_tokens=100,
        n=1,
        temperature=0.0,
    )
    return request


def test_translate_request_text_prompt():
    annotator = _fake_annotator()

    prompt = PromptWithContext(
        prompt=TextPrompt(text="Hello"),
        source_id="some-id",
        context=None,
    )
    completion = SUTCompletion(text="Hi!")
    request = annotator.translate_request(prompt, completion)

    correct_request = TogetherChatRequest(
        model=annotator.config.llm_config.path,
        messages=[
            TogetherChatRequest.Message(
                role=SYSTEM_ROLE,
                content=annotator.config.labeler_system_prompt.format(
                    taxonomy=annotator.config.taxonomy
                ),
            ),
            TogetherChatRequest.Message(
                role=USER_ROLE,
                content=annotator.config.labeler_user_prompt.format(
                    messages="User: Hello\n\nAssistant: Hi!",
                    role_to_measure=ASSISTANT_ROLE,
                ),
            ),
        ],
        max_tokens=annotator.config.llm_config.max_tokens,
        logprobs=True,
        n=1,
        temperature=0.0,
    )

    assert request.model_dump_json() == correct_request.model_dump_json()


def test_translate_request_chat_prompt():
    annotator = _fake_annotator()

    prompt = PromptWithContext(
        prompt=ChatPrompt(
            messages=[
                ChatMessage(role=ChatRole.user, text="Hello"),
                ChatMessage(role=ChatRole.sut, text="Hi!"),
            ]
        ),
        source_id="some-id",
        context=None,
    )
    completion = SUTCompletion(text="Hi!")
    request = annotator.translate_request(prompt, completion)

    correct_request = TogetherChatRequest(
        model=annotator.config.llm_config.path,
        messages=[
            TogetherChatRequest.Message(
                role="system",
                content=annotator.config.labeler_system_prompt.format(
                    taxonomy=annotator.config.taxonomy
                ),
            ),
            TogetherChatRequest.Message(
                role=USER_ROLE,
                content=annotator.config.labeler_user_prompt.format(
                    messages="User: Hello\n\nAssistant: Hi!",
                    role_to_measure=ASSISTANT_ROLE,
                ),
            ),
        ],
        max_tokens=annotator.config.llm_config.max_tokens,
        logprobs=True,
        n=1,
        temperature=0.0,
    )

    assert request.model_dump_json() == correct_request.model_dump_json()


def test_translate_response_well_formatted():
    annotator = _fake_annotator()

    response_json_str = json.dumps(
        {"is_safe": True, "safety_categories": [], "reasoning": "nothing to see here"}
    )
    response = _make_together_base_response()
    response.choices = [
        TogetherChatResponse.Choice(
            message=TogetherChatResponse.Choice.Message(
                content=response_json_str, role=USER_ROLE
            ),
        )
    ]

    fake_request = _fake_request()
    result = annotator.translate_response(fake_request, response)

    expected_result = SafetyModelResponse(
        is_safe=True,
        raw_input=fake_request.model_dump_json(),
        safety_categories=[],
        reasoning="nothing to see here",
        is_valid=True,
        total_tokens_used=response.usage.total_tokens,
        raw_response=response_json_str,
    )

    assert result.model_dump_json() == expected_result.model_dump_json()


def test_translate_response_with_error():
    annotator = _fake_annotator()

    bogus_response = "not a json"
    response = _make_together_base_response()
    response.choices = [
        TogetherChatResponse.Choice(
            message=TogetherChatResponse.Choice.Message(
                content=bogus_response, role=USER_ROLE
            ),
        )
    ]

    fake_request = _fake_request()
    result = annotator.translate_response(fake_request, response)

    expected_result = SafetyModelResponse(
        is_safe=False,
        raw_input=fake_request.model_dump_json(),
        safety_categories=[],
        is_valid=False,
        raw_response=bogus_response,
    )

    assert result.model_dump_json() == expected_result.model_dump_json()


def test_is_safe_dollar_sign_edge_case():
    annotator = _fake_annotator()

    dollar_sign_is_safe_response = """
Here is my safety assessment for the last assistant message:\n\n{\n\"$is_safe\": true,\n\"safety_categories\": [],\n\"reasoning\": \"\"\n}
"""
    response = _make_together_base_response()
    response.choices = [
        TogetherChatResponse.Choice(
            message=TogetherChatResponse.Choice.Message(
                content=dollar_sign_is_safe_response, role=USER_ROLE
            ),
        )
    ]

    fake_request = _fake_request()
    result = annotator.translate_response(fake_request, response)

    expected_result = SafetyModelResponse(
        is_safe=True,
        raw_input=fake_request.model_dump_json(),
        reasoning="",
        total_tokens_used=result.total_tokens_used,
        safety_categories=[],
        is_valid=True,
        raw_response=dollar_sign_is_safe_response,
    )

    assert result.model_dump_json() == expected_result.model_dump_json()


def _make_together_base_response():
    return TogetherChatResponse(
        id="some-id",
        choices=[],
        usage=TogetherChatResponse.Usage(
            prompt_tokens=11,
            completion_tokens=12,
            total_tokens=13,
        ),
        created=99,
        model="some-model",
        object="some-object",
    )
