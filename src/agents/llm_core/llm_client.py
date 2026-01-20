"""
Unified LLM Client for OpenAI-compatible APIs.

Supports 14+ LLM providers (OpenAI, Ollama, Groq, Together, Mistral, etc.)
with automatic capability-based fallbacks.

Output Modes:
    - text: Free text response (returns TextResponse)
    - json_schema: Structured output via JSON schema (returns StructuredResponse[dict])
    - json_schema_strict: Strict JSON schema with constrained decoding
    - pydantic: Structured output via Pydantic model (returns StructuredResponse[T])
    - pydantic_strict: Strict Pydantic mode with constrained decoding
    - tool_calling: Function/tool calling (returns ToolCallResponse)
    - tool_calling_strict: Strict tool calling with constrained decoding

Features:
    - Multi-provider support with automatic fallbacks
    - Sync and async execution (generate/agenerate)
    - Parallel tool calls (where supported)
    - Schema injection for providers without native json_schema support

Provider Fallbacks:
    When a Provider is specified, unsupported features are automatically downgraded:
    - strict modes -> non-strict (with CompatibilityFallbackWarning)
    - parallel_tool_calls -> sequential (with warning)
    - json_schema -> text + schema in prompt (for Cerebras, SambaNova, etc.)
    - Ollama: uses json_object format + schema injection

┌─────────────────────────────────────────────────────────────┐
│ Request: mode="pydantic", response_model=Answer             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │     Is it Ollama?       │
              └─────────────────────────┘
                     │           │
                    Yes          No
                     │           │
                     ▼           ▼
        ┌──────────────────┐  ┌──────────────────────────┐
        │ response_format: │  │ response_format:         │
        │ {"type":         │  │ {"type": "json_schema",  │
        │  "json_object"}  │  │  "json_schema": {...}}   │
        │                  │  │                          │
        │ + Schema in      │  │ API enforces schema      │
        │   prompt         │  │ automatically            │
        └──────────────────┘  └──────────────────────────┘

Usage:
    # Using factory functions (recommended)
    from agents.llm_core.llm_client import create_openai_client, create_ollama_client

    client = create_openai_client()
    response = client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        mode="text"
    )

    # Structured output with Pydantic
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str
        confidence: float

    response = client.generate(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        mode="pydantic",
        response_model=Answer
    )
    print(response.parsed.answer)

    # Tool calling with BaseTool
    from agents.tools_core.base_tool import BaseTool, create_fn_tool

    @create_fn_tool(name="search", description="Search the web")
    def search(query: str) -> str:
        return f"Results for: {query}"

    response = client.generate(
        messages=[{"role": "user", "content": "Search for Python tutorials"}],
        mode="tool_calling",
        tools=[search],  # Pass BaseTool instances
        parallel_tool_calls=True
    )

    # Ollama with automatic fallbacks
    ollama = create_ollama_client()  # Provider.OLLAMA auto-configured
    response = ollama.generate(
        messages=[{"role": "user", "content": "Hi"}],
        mode="pydantic_strict",  # Auto-downgraded to pydantic with warning
        response_model=Answer
        include_schema_in_prompt=True
    )

Factory Functions:
    - create_openai_client(): OpenAI with full feature support
    - create_ollama_client(): Ollama with automatic fallbacks
    - create_azure_client(): Azure OpenAI

See llm_configs.py for full provider capability matrix.
"""

from __future__ import annotations

import typing as t
import json

from loguru import logger
from copy import deepcopy
from dataclasses import dataclass

from openai import APIConnectionError, APIError, AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, ValidationError

from agents.llm_core.llm_configs import Provider
from agents.settings import get_api_key, get_endpoint
from agents.tools_core.base_tool import BaseTool
from agents.utilities.pydantic_utils import process_schema

# Type definitions
T = t.TypeVar("T", bound=BaseModel)

OutputMode = t.Literal[
    "text",
    "json_schema",
    "json_schema_strict",
    "pydantic",
    "pydantic_strict",
    "tool_calling",
    "tool_calling_strict",
]

# Mode constants
JSON_MODES: tuple[str, ...] = ("json_schema", "json_schema_strict")
PYDANTIC_MODES: tuple[str, ...] = ("pydantic", "pydantic_strict")
TOOL_MODES: tuple[str, ...] = ("tool_calling", "tool_calling_strict")
STRICT_MODES: tuple[str, ...] = (
    "json_schema_strict",
    "pydantic_strict",
    "tool_calling_strict",
)

# Schema prompt template
SCHEMA_PROMPT_TEMPLATE = """You must respond with valid JSON that conforms to this schema:

```json
{schema}
```

Respond ONLY with valid JSON matching this schema, no additional text."""


# Exceptions
class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class LLMAPIError(LLMError):
    """API-level error from LLM provider."""

    pass


class LLMParsingError(LLMError):
    """Failed to parse LLM response into expected format."""

    def __init__(self, message: str, raw_content: str | None = None):
        super().__init__(message)
        self.raw_content = raw_content


class LLMValidationError(LLMError):
    """Schema validation failed."""

    def __init__(self, message: str, validation_error: ValidationError | None = None):
        super().__init__(message)
        self.validation_error = validation_error


class CompatibilityFallbackWarning(UserWarning):
    """Warning when mode is automatically downgraded due to provider limitations."""

    pass


# Response types
@dataclass
class TextResponse:
    """Response for text mode."""

    content: str
    finish_reason: str | None = None
    raw_response: ChatCompletion | None = None


@dataclass
class StructuredResponse(t.Generic[T]):
    """Response for structured output modes."""

    parsed: T
    raw_content: str | None = None
    finish_reason: str | None = None
    raw_response: ChatCompletion | None = None


@dataclass
class ToolCall:
    """Represents a single tool call."""

    id: str
    tool_name: str
    arguments: dict[str, t.Any]
    parsed: BaseModel | None = None


@dataclass
class ToolCallResponse:
    """Response for tool calling modes."""

    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    raw_response: ChatCompletion | None = None


# Utility functions
def _is_ollama(api_key: str | None) -> bool:
    """Check if the API key indicates Ollama."""
    return api_key == get_api_key(Provider.OLLAMA)


def _make_schema_strict_compatible(schema: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Transform a JSON schema to be compatible with OpenAI strict mode.

    Strict mode requires:
    - additionalProperties: false on all objects
    - All fields must be required
    - No default values in schema
    """
    schema = deepcopy(schema)
    return process_schema(
        schema,
        hide_defaults=True,
        enforce_additional_properties=True,
        make_fields_required=True,
    )


def _make_schema_non_strict_compatible(schema: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Transform a JSON schema for non-strict mode.

    Non-strict mode benefits from:
    - Hidden defaults (mentioned in description)
    - Original required fields preserved
    """
    schema = deepcopy(schema)
    return process_schema(schema, hide_defaults=True)


def _pydantic_to_json_schema(
    model: type[BaseModel], strict: bool = False
) -> dict[str, t.Any]:
    """Convert a Pydantic model to a JSON schema dict."""
    schema = model.model_json_schema()
    if strict:
        return _make_schema_strict_compatible(schema)
    return _make_schema_non_strict_compatible(schema)


def _inject_schema_into_messages(
    messages: list[ChatCompletionMessageParam],
    schema: dict[str, t.Any],
) -> list[ChatCompletionMessageParam]:
    """
    Inject schema instructions into messages for models that don't support
    server-side schema enforcement (e.g., Ollama).

    Adds schema as a system message at the beginning of the conversation.
    """

    schema_str = json.dumps(schema, indent=2)
    schema_message: ChatCompletionMessageParam = {
        "role": "system",
        "content": SCHEMA_PROMPT_TEMPLATE.format(schema=schema_str),
    }

    # Create new list with schema message first
    return [schema_message] + list(messages)


def _create_tool_schema(
    tool: BaseTool[BaseModel, BaseModel], strict: bool = False
) -> ChatCompletionToolParam:
    """Create an OpenAI tool schema from a BaseTool."""
    schema = tool.input_schema()

    # Process schema for strict mode if needed
    if strict:
        schema = _make_schema_strict_compatible(schema)
    else:
        schema = _make_schema_non_strict_compatible(schema)

    # Remove title from schema (it goes in the function name)
    schema.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
            "strict": strict,
        },
    }


def _parse_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    tools: t.Sequence[BaseTool[BaseModel, BaseModel]],
) -> list[ToolCall]:
    """Parse raw tool calls into ToolCall objects with validated Pydantic models.

    Unknown tools (not in the provided tools list) are skipped with a warning.
    """
    import warnings

    # Build tool name -> BaseTool mapping
    tool_map = {tool.name: tool for tool in tools}

    result = []
    for tc in tool_calls:
        # Skip unknown tools
        if tc.function.name not in tool_map:
            warnings.warn(
                f"LLM returned unknown tool '{tc.function.name}', skipping",
                RuntimeWarning,
            )
            continue

        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            raise LLMParsingError(
                f"Failed to parse tool arguments as JSON: {e}",
                raw_content=tc.function.arguments,
            )

        tool = tool_map[tc.function.name]
        try:
            parsed = tool._input.model_validate(args)
        except ValidationError as e:
            raise LLMValidationError(
                f"Tool arguments failed validation for {tc.function.name}: {e}",
                validation_error=e,
            )

        result.append(
            ToolCall(
                id=tc.id,
                tool_name=tc.function.name,
                arguments=args,
                parsed=parsed,
            )
        )

    return result


class LLMClient:
    """
    Unified LLM client supporting multiple output modes.

    Args:
        client: Sync OpenAI client for blocking calls
        async_client: Async OpenAI client for non-blocking calls
        default_model: Default model to use if not specified in generate()
        provider: Provider enum for automatic fallback behavior. When set,
            unsupported features are automatically downgraded with warnings:
            - strict modes → non-strict if not supported
            - parallel tool calls → sequential if not supported
            - json_schema → text + schema in prompt if not supported
            - Ollama: uses json_object format + schema in prompt

    Example:
        >>> client = LLMClient(OpenAI(), default_model="gpt-4o")
        >>> response = client.generate(messages, mode="text")
        >>> print(response.content)

        >>> # With provider fallbacks
        >>> client = LLMClient(
        ...     OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        ...     provider=Provider.OLLAMA
        ... )
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        default_model: str | None = None,
        provider: Provider | None = None,
    ):
        self._client = client
        self._async_client = async_client
        self._default_model = default_model
        self._is_ollama = _is_ollama(client.api_key) if client else False
        self._provider = provider

    def _apply_compatibility_fallback(
        self,
        mode: OutputMode,
        parallel_tool_calls: bool,
    ) -> tuple[OutputMode, bool, bool]:
        """
        Apply automatic fallbacks based on provider capabilities.

        Returns:
            Tuple of (adjusted_mode, adjusted_parallel_tool_calls, include_schema_in_prompt)

        Raises:
            LLMError: If tool_calling is requested but not supported
        """
        if self._provider is None:
            return mode, parallel_tool_calls, False

        provider = self._provider
        adjusted_mode = mode
        adjusted_parallel = parallel_tool_calls
        include_schema_in_prompt = False

        # Handle tool calling modes
        if mode in TOOL_MODES:
            # Check tool calling support first
            if not provider.supports_tool_calling:
                raise LLMError(
                    f"Provider {provider.name} does not support tool calling. "
                    f"Cannot use mode '{mode}'."
                )

            # Disable parallel if not supported
            if parallel_tool_calls and not provider.supports_parallel_tool_calls:
                logger.warning(
                    f"[{CompatibilityFallbackWarning.__name__}] "
                    f"Provider {provider.name} does not support parallel tool calls. "
                    "Falling back to sequential tool calls."
                )
                adjusted_parallel = False

            # Downgrade strict to non-strict if not supported
            if mode == "tool_calling_strict" and not provider.supports_strict_mode:
                logger.warning(
                    f"[{CompatibilityFallbackWarning.__name__}] "
                    f"Provider {provider.name} does not support strict mode. "
                    "Falling back to 'tool_calling' (non-strict)."
                )
                adjusted_mode = "tool_calling"

        # Handle pydantic/json_schema modes
        elif mode in PYDANTIC_MODES or mode in JSON_MODES:
            # Downgrade strict to non-strict if not supported
            if mode in STRICT_MODES and not provider.supports_strict_mode:
                if mode == "pydantic_strict":
                    adjusted_mode = "pydantic"
                elif mode == "json_schema_strict":
                    adjusted_mode = "json_schema"
                logger.warning(
                    f"[{CompatibilityFallbackWarning.__name__}] "
                    f"Provider {provider.name} does not support strict mode. "
                    f"Falling back to '{adjusted_mode}' (non-strict)."
                )

            # Ollama supports json_object format but needs schema in prompt
            if self._is_ollama:
                include_schema_in_prompt = True
            # Fall back to text + schema in prompt if json_schema not supported
            elif not provider.supports_json_schema:
                logger.warning(
                    f"[{CompatibilityFallbackWarning.__name__}] "
                    f"Provider {provider.name} does not support json_schema. "
                    "Falling back to 'text' mode with schema injected in prompt."
                )
                adjusted_mode = "text"
                include_schema_in_prompt = True

        return adjusted_mode, adjusted_parallel, include_schema_in_prompt

    def _build_chat_kwargs(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        mode: OutputMode,
        response_model: type[T] | None = None,
        response_schema: dict[str, t.Any] | None = None,
        tools: t.Sequence[BaseTool[BaseModel, BaseModel]] | None = None,
        parallel_tool_calls: bool = False,
        include_schema_in_prompt: bool = False,
        **extra_kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Build kwargs for chat.completions.create()."""
        strict = mode in STRICT_MODES
        is_ollama = self._is_ollama
        final_messages = list(messages)
        schema_for_prompt: dict[str, t.Any] | None = None

        if mode == "text":
            # For text mode with schema injection (fallback from pydantic/json modes),
            # we still need to prepare the schema for prompt injection
            if include_schema_in_prompt:
                if response_model is not None:
                    schema_for_prompt = _pydantic_to_json_schema(
                        response_model, strict=False
                    )
                elif response_schema is not None:
                    schema_for_prompt = _make_schema_non_strict_compatible(
                        response_schema
                    )

        elif mode in JSON_MODES:
            if response_schema is None:
                raise ValueError(f"response_schema is required for mode '{mode}'")

            schema = (
                _make_schema_strict_compatible(response_schema)
                if strict
                else _make_schema_non_strict_compatible(response_schema)
            )
            schema_for_prompt = schema

        elif mode in PYDANTIC_MODES:
            if response_model is None:
                raise ValueError(f"response_model is required for mode '{mode}'")

            schema = _pydantic_to_json_schema(response_model, strict=strict)
            schema_for_prompt = schema

        # Inject schema into prompt if requested
        if include_schema_in_prompt and schema_for_prompt is not None:
            final_messages = _inject_schema_into_messages(
                final_messages, schema_for_prompt
            )

        kwargs: dict[str, t.Any] = {
            "messages": final_messages,
            "model": model,
            **extra_kwargs,
        }

        # Set response_format for structured modes
        if mode in JSON_MODES and response_schema:
            schema = (
                _make_schema_strict_compatible(response_schema)
                if strict
                else _make_schema_non_strict_compatible(response_schema)
            )
            if is_ollama:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("title", "response"),
                        "strict": strict,
                        "schema": schema,
                    },
                }

        elif mode in PYDANTIC_MODES and response_model:
            schema = _pydantic_to_json_schema(response_model, strict=strict)
            if is_ollama:
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("title", response_model.__name__),
                        "strict": strict,
                        "schema": schema,
                    },
                }

        elif mode in TOOL_MODES:
            if not tools:
                raise ValueError(f"tools is required for mode '{mode}'")

            tool_schemas = [_create_tool_schema(tool, strict=strict) for tool in tools]
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "required"

            if parallel_tool_calls:
                kwargs["parallel_tool_calls"] = True

        return kwargs

    def _process_response(
        self,
        response: ChatCompletion,
        mode: OutputMode,
        original_mode: OutputMode | None = None,
        response_model: type[T] | None = None,
        tools: t.Sequence[BaseTool[BaseModel, BaseModel]] | None = None,
        first_tool_only: bool = True,
        fallback_convert_original: bool = True,
    ) -> TextResponse | StructuredResponse[T] | ToolCallResponse:
        """Process the API response based on mode.

        Args:
            response: Raw API response
            mode: The effective mode used for the API call
            original_mode: The originally requested mode (before fallbacks)
            response_model: Pydantic model for validation
            tools: Tools for tool calling modes
            first_tool_only: Return only first tool call
            fallback_convert_original: If True and mode was downgraded,
                attempt to parse response according to original mode
        """
        choice = response.choices[0]
        message = choice.message

        # Determine if we should convert back to original format
        should_convert = (
            fallback_convert_original
            and original_mode is not None
            and mode != original_mode
        )

        if mode == "text":
            # If original mode was pydantic/json, try to parse as structured
            if should_convert and original_mode in PYDANTIC_MODES and response_model:
                content = message.content or ""
                try:
                    parsed_dict = json.loads(content)
                    parsed = response_model.model_validate(parsed_dict)
                    return StructuredResponse(
                        parsed=parsed,
                        raw_content=content,
                        finish_reason=choice.finish_reason,
                        raw_response=response,
                    )
                except (json.JSONDecodeError, ValidationError) as e:
                    raise LLMParsingError(
                        f"Failed to parse fallback response as {response_model.__name__}: {e}",
                        raw_content=content,
                    )
            elif should_convert and original_mode in JSON_MODES:
                content = message.content or ""
                try:
                    parsed_dict = json.loads(content)
                    return StructuredResponse(
                        parsed=parsed_dict,
                        raw_content=content,
                        finish_reason=choice.finish_reason,
                        raw_response=response,
                    )
                except json.JSONDecodeError as e:
                    raise LLMParsingError(
                        f"Failed to parse fallback response as JSON: {e}",
                        raw_content=content,
                    )

            return TextResponse(
                content=message.content or "",
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        elif mode in JSON_MODES:
            content = message.content or ""
            try:
                parsed_dict = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMParsingError(
                    f"Failed to parse response as JSON: {e}", raw_content=content
                )

            # If response_model provided, validate against it (fallback from pydantic)
            if response_model is not None:
                try:
                    parsed = response_model.model_validate(parsed_dict)
                except ValidationError as e:
                    raise LLMValidationError(
                        f"Response failed schema validation: {e}",
                        validation_error=e,
                    )
            else:
                parsed = parsed_dict

            return StructuredResponse(
                parsed=parsed,
                raw_content=content,
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        elif mode in PYDANTIC_MODES:
            if response_model is None:
                raise ValueError("response_model is required for pydantic modes")

            content = message.content or ""
            try:
                parsed = response_model.model_validate_json(content)
            except ValidationError as e:
                raise LLMValidationError(
                    f"Response failed schema validation: {e}",
                    validation_error=e,
                )

            return StructuredResponse(
                parsed=parsed,
                raw_content=content,
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        elif mode in TOOL_MODES:
            if not message.tool_calls:
                raise LLMParsingError(
                    "Expected tool calls but none were returned",
                    raw_content=message.content,
                )

            tool_calls = _parse_tool_calls(message.tool_calls, tools or [])

            return ToolCallResponse(
                tool_calls=tool_calls if not first_tool_only else tool_calls[:1],
                finish_reason=choice.finish_reason,
                raw_response=response,
            )

        raise ValueError(f"Unknown mode: {mode}")

    # Overloads for type hints
    @t.overload
    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["text"] = "text",
        **kwargs: t.Any,
    ) -> TextResponse: ...

    @t.overload
    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["json_schema", "json_schema_strict"],
        response_schema: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> StructuredResponse[dict[str, t.Any]]: ...

    @t.overload
    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["pydantic", "pydantic_strict"],
        response_model: type[T],
        **kwargs: t.Any,
    ) -> StructuredResponse[T]: ...

    @t.overload
    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["tool_calling", "tool_calling_strict"],
        tools: t.Sequence[BaseTool[BaseModel, BaseModel]],
        parallel_tool_calls: bool = False,
        **kwargs: t.Any,
    ) -> ToolCallResponse: ...

    def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: OutputMode = "text",
        response_model: type[T] | None = None,
        response_schema: dict[str, t.Any] | None = None,
        tools: t.Sequence[BaseTool[BaseModel, BaseModel]] | None = None,
        parallel_tool_calls: bool = False,
        include_schema_in_prompt: bool = False,
        fallback_convert_original: bool = True,
        **kwargs: t.Any,
    ) -> TextResponse | StructuredResponse[T] | ToolCallResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of chat messages
            model: Model name (uses default_model if not specified)
            mode: Output mode - one of:
                - "text": Free text response
                - "json_schema": Structured output via JSON schema
                - "json_schema_strict": Strict JSON schema mode
                - "pydantic": Structured output via Pydantic model
                - "pydantic_strict": Strict Pydantic mode
                - "tool_calling": Function calling
                - "tool_calling_strict": Strict function calling
            response_model: Pydantic model for pydantic modes
            response_schema: JSON schema dict for json_schema modes
            tools: List of BaseTool instances for tool calling modes
            parallel_tool_calls: Allow parallel tool calls (tool modes only)
            include_schema_in_prompt: If True, inject schema into prompt as system
                message. Useful for Ollama and models without native schema support.
            fallback_convert_original: If True (default), when mode is downgraded
                due to provider limitations, still attempt to parse the response
                according to the original mode (e.g., parse text as JSON when
                original mode was pydantic but was downgraded to text).
            **kwargs: Additional kwargs passed to chat.completions.create()

        Returns:
            TextResponse, StructuredResponse, or ToolCallResponse based on mode

        Raises:
            RuntimeError: If sync client not configured
            LLMAPIError: API connection or rate limit issues
            LLMParsingError: Failed to parse response
            LLMValidationError: Schema validation failed
        """
        if self._client is None:
            raise RuntimeError("Sync client not configured. Pass 'client' to __init__.")

        model = model or self._default_model
        if model is None:
            raise ValueError("model must be specified or set default_model")

        # Apply compatibility fallbacks if provider is specified
        effective_mode, effective_parallel, force_schema_in_prompt = (
            self._apply_compatibility_fallback(mode, parallel_tool_calls)
        )
        effective_include_schema = include_schema_in_prompt or force_schema_in_prompt

        chat_kwargs = self._build_chat_kwargs(
            messages=messages,
            model=model,
            mode=effective_mode,
            include_schema_in_prompt=effective_include_schema,
            response_model=response_model,
            response_schema=response_schema,
            tools=tools,
            parallel_tool_calls=effective_parallel,
            **kwargs,
        )

        try:
            response = self._client.chat.completions.create(**chat_kwargs)
        except (APIConnectionError, RateLimitError) as e:
            raise LLMAPIError(f"API error: {e}") from e
        except APIError as e:
            raise LLMAPIError(f"OpenAI API error: {e}") from e

        return self._process_response(
            response=response,
            mode=effective_mode,
            original_mode=mode,
            response_model=response_model,
            tools=tools,
            first_tool_only=not effective_parallel,
            fallback_convert_original=fallback_convert_original,
        )

    # Async overloads
    @t.overload
    async def agenerate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["text"] = "text",
        **kwargs: t.Any,
    ) -> TextResponse: ...

    @t.overload
    async def agenerate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["json_schema", "json_schema_strict"],
        response_schema: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> StructuredResponse[dict[str, t.Any]]: ...

    @t.overload
    async def agenerate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["pydantic", "pydantic_strict"],
        response_model: type[T],
        **kwargs: t.Any,
    ) -> StructuredResponse[T]: ...

    @t.overload
    async def agenerate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: t.Literal["tool_calling", "tool_calling_strict"],
        tools: t.Sequence[BaseTool[BaseModel, BaseModel]],
        parallel_tool_calls: bool = False,
        **kwargs: t.Any,
    ) -> ToolCallResponse: ...

    async def agenerate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        *,
        mode: OutputMode = "text",
        response_model: type[T] | None = None,
        response_schema: dict[str, t.Any] | None = None,
        tools: t.Sequence[BaseTool[BaseModel, BaseModel]] | None = None,
        parallel_tool_calls: bool = False,
        include_schema_in_prompt: bool = False,
        fallback_convert_original: bool = True,
        **kwargs: t.Any,
    ) -> TextResponse | StructuredResponse[T] | ToolCallResponse:
        """
        Async version of generate().

        See generate() for full documentation.
        """
        if self._async_client is None:
            raise RuntimeError(
                "Async client not configured. Pass 'async_client' to __init__."
            )

        model = model or self._default_model
        if model is None:
            raise ValueError("model must be specified or set default_model")

        # Apply compatibility fallbacks if provider is specified
        effective_mode, effective_parallel, force_schema_in_prompt = (
            self._apply_compatibility_fallback(mode, parallel_tool_calls)
        )
        effective_include_schema = include_schema_in_prompt or force_schema_in_prompt

        chat_kwargs = self._build_chat_kwargs(
            messages=messages,
            model=model,
            mode=effective_mode,
            include_schema_in_prompt=effective_include_schema,
            response_model=response_model,
            response_schema=response_schema,
            tools=tools,
            parallel_tool_calls=effective_parallel,
            **kwargs,
        )

        try:
            response = await self._async_client.chat.completions.create(**chat_kwargs)
        except (APIConnectionError, RateLimitError) as e:
            raise LLMAPIError(f"API error: {e}") from e
        except APIError as e:
            raise LLMAPIError(f"OpenAI API error: {e}") from e

        return self._process_response(
            response=response,
            mode=effective_mode,
            original_mode=mode,
            response_model=response_model,
            tools=tools,
            first_tool_only=not effective_parallel,
            fallback_convert_original=fallback_convert_original,
        )


# Factory functions for common configurations
def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    default_model: str | None = None,
    async_client: bool = False,
) -> LLMClient:
    """
    Create an LLMClient configured for OpenAI.

    Args:
        api_key: OpenAI API key (uses settings if not set)
        base_url: Optional base URL override
        timeout: Request timeout in seconds
        default_model: Default model to use
        async_client: If True, also create async client

    Returns:
        Configured LLMClient
    """
    client_kwargs: dict[str, t.Any] = {"timeout": timeout}
    client_kwargs["api_key"] = api_key or get_api_key(Provider.OPENAI) or None
    client_kwargs["base_url"] = base_url or get_endpoint(Provider.OPENAI)

    sync_client = OpenAI(**client_kwargs)
    async_client_obj = AsyncOpenAI(**client_kwargs) if async_client else None

    return LLMClient(
        client=sync_client,
        async_client=async_client_obj,
        default_model=default_model or Provider.OPENAI.default_model,
        provider=Provider.OPENAI,
    )


def create_ollama_client(
    base_url: str | None = None,
    timeout: float = 120.0,
    default_model: str | None = None,
    async_client: bool = False,
) -> LLMClient:
    """
    Create an LLMClient configured for Ollama with automatic fallbacks.

    Args:
        base_url: Ollama API URL (default from ProviderConfig)
        timeout: Request timeout in seconds
        default_model: Default model to use (default from ProviderConfig)
        async_client: If True, also create async client

    Returns:
        Configured LLMClient with provider=Provider.OLLAMA for fallbacks

    Note:
        Automatic fallbacks applied:
        - Uses json_object format instead of json_schema
        - Injects schema into prompt for structured output
        - Strict modes downgraded to non-strict
        - Parallel tool calls disabled
    """
    client_kwargs: dict[str, t.Any] = {
        "base_url": base_url or get_endpoint(Provider.OLLAMA),
        "api_key": get_api_key(Provider.OLLAMA),
        "timeout": timeout,
    }

    sync_client = OpenAI(**client_kwargs)
    async_client_obj = AsyncOpenAI(**client_kwargs) if async_client else None

    return LLMClient(
        client=sync_client,
        async_client=async_client_obj,
        default_model=default_model or Provider.OLLAMA.default_model,
        provider=Provider.OLLAMA,
    )


def create_azure_client(
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    default_model: str | None = None,
    async_client: bool = False,
) -> LLMClient:
    """
    Create an LLMClient configured for Azure OpenAI.

    Args:
        endpoint: Azure OpenAI endpoint (uses settings if not set)
        api_key: Azure API key (uses settings if not set)
        timeout: Request timeout in seconds
        default_model: Default deployment name
        async_client: If True, also create async client

    Returns:
        Configured LLMClient for Azure
    """
    resolved_endpoint = endpoint or get_endpoint(Provider.AZURE_OPENAI)
    resolved_api_key = api_key or get_api_key(Provider.AZURE_OPENAI)

    if not resolved_endpoint:
        raise ValueError("endpoint or AZURE_OPENAI_ENDPOINT env var required")
    if not resolved_api_key:
        raise ValueError("api_key or AZURE_OPENAI_API_KEY env var required")

    client_kwargs: dict[str, t.Any] = {
        "base_url": resolved_endpoint.rstrip("/"),
        "api_key": resolved_api_key,
        "timeout": timeout,
    }

    sync_client = OpenAI(**client_kwargs)
    async_client_obj = AsyncOpenAI(**client_kwargs) if async_client else None

    return LLMClient(
        client=sync_client,
        async_client=async_client_obj,
        default_model=default_model or Provider.AZURE_OPENAI.default_model,
        provider=Provider.AZURE_OPENAI,
    )
