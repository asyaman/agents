"""
LLM Provider Configurations.

Centralized base URLs and default models for various LLM providers.
All providers use OpenAI-compatible API format.

Provider Capability Matrix
==========================

| Provider              | json_schema | strict | tools | parallel | Notes                    |
|-----------------------|-------------|--------|-------|----------|--------------------------|
| OPENAI                | Yes         | Yes    | Yes   | Yes      | Full support             |
| AZURE_OPENAI          | Yes         | Yes    | Yes   | Yes      | Full support             |
| OLLAMA                | No          | No     | Yes   | No       | json_object only         |
| HUGGINGFACE_ROUTER    | No          | No     | No    | No       | Depends on backend       |
| HUGGINGFACE_INFERENCE | No          | No     | No    | No       | TGI limitations          |
| GROQ                  | Yes         | No     | Yes   | Yes      |                          |
| TOGETHER              | Yes         | No     | Yes   | Yes      |                          |
| MISTRAL               | Yes         | No     | Yes   | Yes      |                          |
| FIREWORKS             | Yes         | Yes    | Yes   | Yes      | Grammar-based decoding   |
| ANYSCALE              | Yes         | No     | Yes   | Yes      |                          |
| DEEPSEEK              | Yes         | No     | Yes   | Yes      |                          |
| CEREBRAS              | No          | No     | No    | No       | Limited support          |
| SAMBANOVA             | No          | No     | No    | No       | Limited support          |
| VLLM                  | Yes         | Yes    | Yes   | Yes      | Outlines integration     |

"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider.

    Note: API keys are managed centrally via Settings, not here.
    Use get_api_key(provider) from agents.settings to get the API key.
    """

    base_url: str
    default_model: str
    supports_json_schema: bool = True  # Does API support response_format json_schema?
    supports_strict_mode: bool = (
        False  # Does API enforce schema via constrained decoding?
    )
    supports_tool_calling: bool = True  # Does API support tools/function calling?
    supports_parallel_tool_calls: bool = True  # Does API support parallel_tool_calls?


class Provider(Enum):
    """LLM Provider configurations."""

    OPENAI = ProviderConfig(
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        supports_json_schema=True,
        supports_strict_mode=True,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    AZURE_OPENAI = ProviderConfig(
        base_url="",  # Set via settings.azure_openai_endpoint
        default_model="gpt-5-mini",
        supports_json_schema=True,
        supports_strict_mode=True,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    OLLAMA = ProviderConfig(
        base_url="http://localhost:11434/v1",
        default_model="llama3.2",
        supports_json_schema=True,  # with prompt schema injection
        supports_strict_mode=False,
        supports_tool_calling=True,
        supports_parallel_tool_calls=False,
    )

    # Hugging Face
    HUGGINGFACE_ROUTER = ProviderConfig(
        base_url="https://router.huggingface.co/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct",
        supports_json_schema=False,
        supports_strict_mode=False,
        supports_tool_calling=False,
        supports_parallel_tool_calls=False,
    )

    HUGGINGFACE_INFERENCE = ProviderConfig(
        base_url="https://api-inference.huggingface.co/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct",
        supports_json_schema=False,
        supports_strict_mode=False,
        supports_tool_calling=False,
        supports_parallel_tool_calls=False,
    )

    # Other providers
    GROQ = ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        supports_json_schema=True,
        supports_strict_mode=False,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    TOGETHER = ProviderConfig(
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        supports_json_schema=True,
        supports_strict_mode=False,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    MISTRAL = ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        supports_json_schema=True,
        supports_strict_mode=False,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    FIREWORKS = ProviderConfig(
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        supports_json_schema=True,
        supports_strict_mode=True,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    ANYSCALE = ProviderConfig(
        base_url="https://api.endpoints.anyscale.com/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct",
        supports_json_schema=True,
        supports_strict_mode=False,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    DEEPSEEK = ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        supports_json_schema=True,
        supports_strict_mode=False,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    CEREBRAS = ProviderConfig(
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        supports_json_schema=False,
        supports_strict_mode=False,
        supports_tool_calling=False,
        supports_parallel_tool_calls=False,
    )

    SAMBANOVA = ProviderConfig(
        base_url="https://api.sambanova.ai/v1",
        default_model="Meta-Llama-3.3-70B-Instruct",
        supports_json_schema=False,
        supports_strict_mode=False,
        supports_tool_calling=False,
        supports_parallel_tool_calls=False,
    )

    # Self-hosted
    VLLM = ProviderConfig(
        base_url="http://localhost:8000/v1",
        default_model="default",
        supports_json_schema=True,
        supports_strict_mode=True,
        supports_tool_calling=True,
        supports_parallel_tool_calls=True,
    )

    @property
    def base_url(self) -> str:
        return self.value.base_url

    @property
    def default_model(self) -> str:
        return self.value.default_model

    @property
    def supports_json_schema(self) -> bool:
        return self.value.supports_json_schema

    @property
    def supports_tool_calling(self) -> bool:
        return self.value.supports_tool_calling

    @property
    def supports_strict_mode(self) -> bool:
        return self.value.supports_strict_mode

    @property
    def supports_parallel_tool_calls(self) -> bool:
        return self.value.supports_parallel_tool_calls
