"""Essential tests for LLMClient.

Test Classes and Methods
========================

TestTextMode
    - test_text_mode_returns_content

TestPydanticMode
    - test_pydantic_mode_parses_model
    - test_pydantic_mode_raises_on_invalid_json

TestJsonSchemaMode
    - test_json_schema_mode_returns_dict

TestToolCallingMode
    - test_tool_calling_returns_parsed_tools
    - test_parallel_tool_calls

TestOllamaCompatibility
    - test_ollama_detected_from_api_key
    - test_ollama_warns_on_strict_mode

TestSchemaProcessing
    - test_strict_schema_adds_additional_properties
    - test_pydantic_to_json_schema

TestAsyncGenerate
    - test_async_text_mode

TestErrorHandling
    - test_missing_response_model_raises
    - test_missing_tools_raises
    - test_missing_client_raises

TestIncludeSchemaInPrompt
    - test_schema_injected_into_messages

TestCompatibilityFallbacks
    - test_no_fallback_without_provider
    - test_parallel_tools_disabled_for_ollama
    - test_strict_mode_downgraded_for_groq
    - test_json_schema_strict_downgraded_for_together
    - test_tool_calling_strict_downgraded
    - test_json_schema_fallback_to_text_with_prompt
    - test_tool_calling_raises_for_unsupported_provider
    - test_openai_full_support_no_warnings
    - test_ollama_uses_json_object_format
    - test_ollama_injects_schema_in_prompt

TestFallbackConvertOriginal
    - test_pydantic_fallback_to_text_converts_back
    - test_pydantic_fallback_disabled_returns_text
    - test_json_schema_fallback_to_text_converts_back
    - test_fallback_with_invalid_json_raises_error
    - test_fallback_with_invalid_schema_raises_error
    - test_strict_downgrade_does_not_trigger_conversion
    - test_text_mode_schema_injection_for_fallback
    - test_async_fallback_convert_original
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger as loguru_logger
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, Field

from agents.llm_core.llm_client import (
    LLMClient,
    LLMError,
    LLMParsingError,
    LLMValidationError,
    StructuredResponse,
    TextResponse,
    ToolCallResponse,
    _make_schema_strict_compatible,
    _pydantic_to_json_schema,
)
from agents.llm_core.llm_configs import Provider
from agents.tools_core.base_tool import BaseTool


# Test schemas
class SimpleSchema(BaseModel):
    name: str
    value: int


class ToolSchemaInput(BaseModel):
    """Input for tool schema."""

    query: str = Field(description="The query to search for")


class ToolSchemaOutput(BaseModel):
    """Output for tool schema."""

    result: str


class TestToolSchema(BaseTool[ToolSchemaInput, ToolSchemaOutput]):
    """A test tool for LLM client tests."""

    name = "ToolSchema"
    description = "A test tool"
    _input = ToolSchemaInput
    _output = ToolSchemaOutput

    def invoke(self, input: ToolSchemaInput) -> ToolSchemaOutput:
        return ToolSchemaOutput(result=f"Result for: {input.query}")


# Fixtures
@pytest.fixture
def mock_client() -> Mock:
    client = Mock()
    client.api_key = "test-key"
    return client


@pytest.fixture
def mock_async_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def llm_client(mock_client: Mock) -> LLMClient:
    return LLMClient(client=mock_client, default_model="gpt-4o")


@pytest.fixture
def tool_schema() -> TestToolSchema:
    """Create a test tool instance."""
    return TestToolSchema()


@pytest.fixture
def capture_logs():
    """Fixture to capture loguru logs for testing."""
    captured = []

    def sink(message):
        captured.append(message.record)

    handler_id = loguru_logger.add(sink, format="{message}", level="WARNING")
    yield captured
    loguru_logger.remove(handler_id)


def _make_completion(content: str, finish_reason: str = "stop") -> ChatCompletion:
    """Helper to create a ChatCompletion response."""
    return ChatCompletion(
        id="test-id",
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )


def _make_tool_completion(
    tool_calls: list[tuple[str, str, dict]],
) -> ChatCompletion:
    """Helper to create a tool call response."""
    tc_objects = [
        ChatCompletionMessageToolCall(
            id=f"call_{i}",
            type="function",
            function=Function(name=name, arguments=json.dumps(args)),
        )
        for i, (id_, name, args) in enumerate(tool_calls)
    ]

    return ChatCompletion(
        id="test-id",
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None, tool_calls=tc_objects
                ),
                finish_reason="tool_calls",
            )
        ],
    )


# Tests
class TestTextMode:
    def test_text_mode_returns_content(self, llm_client: LLMClient, mock_client: Mock):
        mock_client.chat.completions.create.return_value = _make_completion(
            "Hello world"
        )

        response = llm_client.generate(
            messages=[{"role": "user", "content": "Hi"}], mode="text"
        )

        assert isinstance(response, TextResponse)
        assert response.content == "Hello world"


class TestPydanticMode:
    def test_pydantic_mode_parses_model(self, llm_client: LLMClient, mock_client: Mock):
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        response = llm_client.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
        )

        assert isinstance(response, StructuredResponse)
        assert response.parsed.name == "test"
        assert response.parsed.value == 42

    def test_pydantic_mode_raises_on_invalid_json(
        self, llm_client: LLMClient, mock_client: Mock
    ):
        mock_client.chat.completions.create.return_value = _make_completion(
            "not valid json"
        )

        with pytest.raises(LLMValidationError):
            llm_client.generate(
                messages=[{"role": "user", "content": "Hi"}],
                mode="pydantic",
                response_model=SimpleSchema,
            )


class TestJsonSchemaMode:
    def test_json_schema_mode_returns_dict(
        self, llm_client: LLMClient, mock_client: Mock
    ):
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"key": "value"}'
        )

        response = llm_client.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="json_schema",
            response_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        )

        assert isinstance(response, StructuredResponse)
        assert response.parsed == {"key": "value"}


class TestToolCallingMode:
    def test_tool_calling_returns_parsed_tools(
        self, llm_client: LLMClient, mock_client: Mock, tool_schema: TestToolSchema
    ):
        mock_client.chat.completions.create.return_value = _make_tool_completion(
            [("call_1", "ToolSchema", {"query": "test query"})]
        )

        response = llm_client.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="tool_calling",
            tools=[tool_schema],
        )

        assert isinstance(response, ToolCallResponse)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "ToolSchema"
        assert response.tool_calls[0].parsed.query == "test query"

    def test_parallel_tool_calls(
        self, llm_client: LLMClient, mock_client: Mock, tool_schema: TestToolSchema
    ):
        mock_client.chat.completions.create.return_value = _make_tool_completion(
            [
                ("call_1", "ToolSchema", {"query": "query1"}),
                ("call_2", "ToolSchema", {"query": "query2"}),
            ]
        )

        response = llm_client.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="tool_calling",
            tools=[tool_schema],
            parallel_tool_calls=True,
        )

        assert len(response.tool_calls) == 2


class TestOllamaCompatibility:
    def test_ollama_detected_from_api_key(self):
        client = Mock()
        client.api_key = "ollama"
        llm = LLMClient(client=client)
        assert llm._is_ollama is True

    def test_ollama_warns_on_strict_mode(self, mock_client: Mock, capture_logs):
        mock_client.api_key = "ollama"
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(
            client=mock_client, default_model="llama3.2", provider=Provider.OLLAMA
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic_strict",
            response_model=SimpleSchema,
        )

        # Check loguru warning was emitted
        assert any("strict mode" in r["message"] for r in capture_logs)


class TestSchemaProcessing:
    def test_strict_schema_adds_additional_properties(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        result = _make_schema_strict_compatible(schema)
        assert result["additionalProperties"] is False

    def test_pydantic_to_json_schema(self):
        schema = _pydantic_to_json_schema(SimpleSchema, strict=False)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_async_text_mode(self, mock_async_client: AsyncMock):
        mock_async_client.chat.completions.create.return_value = _make_completion(
            "Hello async"
        )
        mock_async_client.api_key = "test-key"

        llm = LLMClient(async_client=mock_async_client, default_model="gpt-4o")
        response = await llm.agenerate(
            messages=[{"role": "user", "content": "Hi"}], mode="text"
        )

        assert isinstance(response, TextResponse)
        assert response.content == "Hello async"


class TestErrorHandling:
    def test_missing_response_model_raises(self, llm_client: LLMClient):
        with pytest.raises(ValueError, match="response_model is required"):
            llm_client.generate(
                messages=[{"role": "user", "content": "Hi"}], mode="pydantic"
            )

    def test_missing_tools_raises(self, llm_client: LLMClient):
        with pytest.raises(ValueError, match="tools is required"):
            llm_client.generate(
                messages=[{"role": "user", "content": "Hi"}], mode="tool_calling"
            )

    def test_missing_client_raises(self):
        llm = LLMClient()
        with pytest.raises(RuntimeError, match="Sync client not configured"):
            llm.generate(messages=[{"role": "user", "content": "Hi"}], model="gpt-4o")


class TestIncludeSchemaInPrompt:
    def test_schema_injected_into_messages(
        self, llm_client: LLMClient, mock_client: Mock
    ):
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        llm_client.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
            include_schema_in_prompt=True,
        )

        # Check the messages passed to the API include schema
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]

        # First message should be system message with schema
        assert messages[0]["role"] == "system"
        assert "json" in messages[0]["content"].lower()
        assert "name" in messages[0]["content"]
        assert "value" in messages[0]["content"]

        # Original message should be second
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"


class TestCompatibilityFallbacks:
    """Tests for provider-specific compatibility fallbacks."""

    def test_no_fallback_without_provider(self, mock_client: Mock, capture_logs):
        """No warnings or fallbacks when provider is not specified."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(client=mock_client, default_model="gpt-4o")

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic_strict",
            response_model=SimpleSchema,
        )

        # Should not have any fallback warnings when no provider is set
        fallback_warnings = [
            r for r in capture_logs if "CompatibilityFallbackWarning" in r["message"]
        ]
        assert len(fallback_warnings) == 0

    def test_parallel_tools_disabled_for_ollama(
        self, mock_client: Mock, tool_schema: TestToolSchema, capture_logs
    ):
        """Parallel tool calls should be disabled with warning for Ollama."""
        mock_client.api_key = "ollama"
        mock_client.chat.completions.create.return_value = _make_tool_completion(
            [("call_1", "ToolSchema", {"query": "test"})]
        )

        llm = LLMClient(
            client=mock_client, default_model="llama3.2", provider=Provider.OLLAMA
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="tool_calling",
            tools=[tool_schema],
            parallel_tool_calls=True,
        )

        assert any("parallel tool calls" in r["message"] for r in capture_logs)

    def test_strict_mode_downgraded_for_groq(self, mock_client: Mock, capture_logs):
        """Strict modes should downgrade to non-strict for providers without support."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(
            client=mock_client, default_model="llama-3.3-70b", provider=Provider.GROQ
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic_strict",
            response_model=SimpleSchema,
        )

        assert any("strict mode" in r["message"] for r in capture_logs)

    def test_json_schema_strict_downgraded_for_together(
        self, mock_client: Mock, capture_logs
    ):
        """json_schema_strict should downgrade to json_schema for Together."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"key": "value"}'
        )

        llm = LLMClient(
            client=mock_client, default_model="llama-3.3", provider=Provider.TOGETHER
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="json_schema_strict",
            response_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        )

        assert any("strict mode" in r["message"] for r in capture_logs)

    def test_tool_calling_strict_downgraded(
        self, mock_client: Mock, tool_schema: TestToolSchema, capture_logs
    ):
        """tool_calling_strict should downgrade to tool_calling for providers without strict."""
        mock_client.chat.completions.create.return_value = _make_tool_completion(
            [("call_1", "ToolSchema", {"query": "test"})]
        )

        llm = LLMClient(
            client=mock_client, default_model="llama-3.3-70b", provider=Provider.GROQ
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="tool_calling_strict",
            tools=[tool_schema],
        )

        assert any("strict mode" in r["message"] for r in capture_logs)

    def test_json_schema_fallback_to_text_with_prompt(
        self, mock_client: Mock, capture_logs
    ):
        """Providers without json_schema should use text + schema in prompt."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
        )

        # Verify schema was injected into messages
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "json" in messages[0]["content"].lower()

        assert any("json_schema" in r["message"] for r in capture_logs)

    def test_tool_calling_raises_for_unsupported_provider(
        self, mock_client: Mock, tool_schema: TestToolSchema
    ):
        """Tool calling should raise LLMError for providers without support."""
        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        with pytest.raises(LLMError, match="does not support tool calling"):
            llm.generate(
                messages=[{"role": "user", "content": "Hi"}],
                mode="tool_calling",
                tools=[tool_schema],
            )

    def test_openai_full_support_no_warnings(self, mock_client: Mock, capture_logs):
        """OpenAI provider should have full support with no fallback warnings."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(
            client=mock_client, default_model="gpt-4o", provider=Provider.OPENAI
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic_strict",
            response_model=SimpleSchema,
        )

        fallback_warnings = [
            r for r in capture_logs if "CompatibilityFallbackWarning" in r["message"]
        ]
        assert len(fallback_warnings) == 0

    def test_ollama_uses_json_object_format(self, mock_client: Mock):
        """Ollama should use json_object format instead of json_schema."""
        mock_client.api_key = "ollama"
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(
            client=mock_client, default_model="llama3.2", provider=Provider.OLLAMA
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_ollama_injects_schema_in_prompt(self, mock_client: Mock):
        """Ollama should inject schema into prompt for structured output."""
        mock_client.api_key = "ollama"
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 1}'
        )

        llm = LLMClient(
            client=mock_client, default_model="llama3.2", provider=Provider.OLLAMA
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # Schema should be in first system message
        assert messages[0]["role"] == "system"
        assert "name" in messages[0]["content"]
        assert "value" in messages[0]["content"]


class TestFallbackConvertOriginal:
    """Tests for fallback_convert_original feature."""

    def test_pydantic_fallback_to_text_converts_back(
        self, mock_client: Mock, capture_logs
    ):
        """When pydantic mode falls back to text, response should still be parsed as pydantic."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        # Use a provider without json_schema support to trigger fallback to text
        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        response = llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
            fallback_convert_original=True,  # Default, but explicit for clarity
        )

        # Should return StructuredResponse, not TextResponse
        assert isinstance(response, StructuredResponse)
        assert response.parsed.name == "test"
        assert response.parsed.value == 42
        assert any("json_schema" in r["message"] for r in capture_logs)

    def test_pydantic_fallback_disabled_returns_text(
        self, mock_client: Mock, capture_logs
    ):
        """When fallback_convert_original=False, downgraded mode returns TextResponse."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        response = llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
            fallback_convert_original=False,
        )

        # Should return TextResponse since conversion is disabled
        assert isinstance(response, TextResponse)
        assert response.content == '{"name": "test", "value": 42}'
        assert any("json_schema" in r["message"] for r in capture_logs)

    def test_json_schema_fallback_to_text_converts_back(
        self, mock_client: Mock, capture_logs
    ):
        """When json_schema mode falls back to text, response should still be parsed as JSON."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"key": "value", "count": 123}'
        )

        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        response = llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="json_schema",
            response_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
            fallback_convert_original=True,
        )

        # Should return StructuredResponse with dict
        assert isinstance(response, StructuredResponse)
        assert response.parsed == {"key": "value", "count": 123}
        assert any("json_schema" in r["message"] for r in capture_logs)

    def test_fallback_with_invalid_json_raises_error(
        self, mock_client: Mock, capture_logs
    ):
        """When fallback conversion fails due to invalid JSON, should raise LLMParsingError."""
        mock_client.chat.completions.create.return_value = _make_completion(
            "This is not JSON at all"
        )

        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        with pytest.raises(LLMParsingError, match="Failed to parse fallback response"):
            llm.generate(
                messages=[{"role": "user", "content": "Hi"}],
                mode="pydantic",
                response_model=SimpleSchema,
                fallback_convert_original=True,
            )

        assert any("json_schema" in r["message"] for r in capture_logs)

    def test_fallback_with_invalid_schema_raises_error(
        self, mock_client: Mock, capture_logs
    ):
        """When fallback conversion fails due to schema mismatch, should raise LLMParsingError."""
        # Valid JSON but doesn't match SimpleSchema (missing 'value' field)
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "wrong_field": 42}'
        )

        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        with pytest.raises(LLMParsingError, match="Failed to parse fallback response"):
            llm.generate(
                messages=[{"role": "user", "content": "Hi"}],
                mode="pydantic",
                response_model=SimpleSchema,
                fallback_convert_original=True,
            )

        assert any("json_schema" in r["message"] for r in capture_logs)

    def test_strict_downgrade_does_not_trigger_conversion(
        self, mock_client: Mock, capture_logs
    ):
        """When strict->non-strict (same category), no conversion needed."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        llm = LLMClient(
            client=mock_client, default_model="llama-3.3-70b", provider=Provider.GROQ
        )

        response = llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic_strict",
            response_model=SimpleSchema,
        )

        # Should still return StructuredResponse (pydantic_strict -> pydantic is same category)
        assert isinstance(response, StructuredResponse)
        assert response.parsed.name == "test"
        assert any("strict mode" in r["message"] for r in capture_logs)

    def test_text_mode_schema_injection_for_fallback(
        self, mock_client: Mock, capture_logs
    ):
        """When mode falls back to text, schema should still be injected in prompt."""
        mock_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        llm = LLMClient(
            client=mock_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        llm.generate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
        )

        # Verify schema was injected into messages even though mode is text
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "json" in messages[0]["content"].lower()
        assert "name" in messages[0]["content"]
        assert "value" in messages[0]["content"]
        assert any("json_schema" in r["message"] for r in capture_logs)

    @pytest.mark.asyncio
    async def test_async_fallback_convert_original(
        self, mock_async_client: AsyncMock, capture_logs
    ):
        """Async version should also support fallback_convert_original."""
        mock_async_client.chat.completions.create.return_value = _make_completion(
            '{"name": "test", "value": 42}'
        )

        llm = LLMClient(
            async_client=mock_async_client,
            default_model="llama-3.3-70b",
            provider=Provider.CEREBRAS,
        )

        response = await llm.agenerate(
            messages=[{"role": "user", "content": "Hi"}],
            mode="pydantic",
            response_model=SimpleSchema,
            fallback_convert_original=True,
        )

        assert isinstance(response, StructuredResponse)
        assert response.parsed.name == "test"
        assert response.parsed.value == 42
        assert any("json_schema" in r["message"] for r in capture_logs)
