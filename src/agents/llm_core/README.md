# LLM Core

Unified LLM client abstraction supporting 14+ providers with automatic fallback.

## Concept

LLM Core provides a single interface for interacting with multiple LLM providers. Key features:
- Provider-agnostic API
- Automatic capability detection and fallback
- Multiple output modes (text, JSON, Pydantic, tool calling)

## Components

| File | Purpose |
|------|---------|
| `llm_client.py` | Main `LLMClient` class |
| `llm_configs.py` | Provider configurations and capabilities |

## Usage

### Creating a Client

```python
from agents.llm_core.llm_client import create_openai_client, create_ollama_client

# OpenAI client
client = create_openai_client(model="gpt-4o-mini")

# Ollama client (local)
client = create_ollama_client(model="llama3.2")

# Custom provider
from agents.llm_core.llm_configs import Provider
client = LLMClient(
    provider=Provider.GROQ,
    model="llama-3.1-70b-versatile",
)
```

### Text Generation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

response = await client.generate(messages, mode="text")
print(response.text)
```

### Structured Output (JSON Schema)

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

response = await client.generate(
    messages,
    mode="pydantic",
    response_model=Answer
)
print(response.parsed.answer)
```

### Tool Calling

```python
from agents.tools.calculator import Calculator

calc = Calculator()
response = await client.generate(
    messages,
    mode="tool_calling",
    tools=[calc]
)

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call.name}")
    print(f"Input: {tool_call.parsed}")
```

## Output Modes

| Mode | Description |
|------|-------------|
| `text` | Free-form text response |
| `json_schema` | JSON matching a schema |
| `json_schema_strict` | Strict JSON schema validation |
| `pydantic` | Parsed Pydantic model |
| `pydantic_strict` | Strict Pydantic parsing |
| `tool_calling` | Function/tool calls |
| `tool_calling_strict` | Strict tool calling |

## Supported Providers

| Provider | JSON Schema | Tool Calling | Parallel Tools |
|----------|-------------|--------------|----------------|
| OpenAI | Yes | Yes | Yes |
| Azure OpenAI | Yes | Yes | Yes |
| Ollama | Partial | Yes | No |
| Groq | Yes | Yes | Yes |
| Together | Yes | Yes | Yes |
| Mistral | Yes | Yes | No |
| DeepSeek | Yes | Yes | Yes |
| Fireworks | Yes | Yes | Yes |

## Automatic Fallbacks

When a provider doesn't support a feature, LLMClient automatically falls back:
- Strict mode → non-strict mode
- `json_schema` → schema injected in prompt (Ollama)
- Parallel tool calls → sequential

## API Reference

### LLMClient

```python
class LLMClient:
    def __init__(
        self,
        provider: Provider,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    )

    async def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        mode: str = "text",
        response_model: type[BaseModel] | None = None,
        tools: list[BaseTool] | None = None,
        parallel_tool_calls: bool = True,
    ) -> TextResponse | StructuredResponse | ToolCallResponse
```

### Response Types

```python
class TextResponse:
    text: str
    usage: CompletionUsage

class StructuredResponse[T]:
    parsed: T
    usage: CompletionUsage

class ToolCallResponse:
    tool_calls: list[ToolCall]
    usage: CompletionUsage

class ToolCall:
    id: str
    name: str
    arguments: str
    parsed: BaseModel  # Validated input model
```

## See Also

- [Agent Tool](../agent_tool/README.md) - Uses LLMClient for planning
- [Tools Core](../tools_core/README.md) - Tool definitions for tool calling
