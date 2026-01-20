# noqa: E402

from __future__ import annotations

import typing as t
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402

from agents.llm_core.llm_configs import Provider

# For development mode it's more convenient to be able to modify the .env file directly and override the environment.
# Since docker-compose loads the `.env` file and sets the variables when the dev container is created.
# This should go before the `BaseSettings` import
# .parents[2] goes: settings.py -> agents/ -> src/ -> project_root/
_DOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
loaded = load_dotenv(_DOT_ENV_PATH, override=True)


if t.TYPE_CHECKING:
    from agents.llm_core.llm_configs import Provider


class Settings(BaseSettings):
    """Centralized settings for the agents SDK.

    All API keys and configuration should be accessed through get_settings().
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str = ""

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""

    # HuggingFace
    huggingface_api_token: str = ""

    # Other LLM providers
    groq_api_key: str = ""
    together_api_key: str = ""
    mistral_api_key: str = ""
    fireworks_api_key: str = ""
    anyscale_api_key: str = ""
    deepseek_api_key: str = ""
    cerebras_api_key: str = ""
    sambanova_api_key: str = ""

    # Services
    tavily_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False

    # Development
    development_mode: bool = False


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the current settings instance.

    Initializes settings from environment variables if not already configured.
    """
    if _settings is None:
        configure_settings()
    return t.cast(Settings, _settings)


def configure_settings(**kwargs: t.Any) -> None:
    """Configure settings with optional overrides.

    Args:
        **kwargs: Optional setting overrides (e.g., openai_api_key="sk-...")
    """
    global _settings
    _settings = Settings(
        **kwargs
    )  # pyright: ignore[reportCallIssue, reportArgumentType]


def get_api_key(provider: Provider) -> str:
    """Get the API key for a given provider.

    Args:
        provider: The LLM provider to get the API key for.

    Returns:
        The API key string, or empty string if not configured.

    Example:
        >>> from agents.llm_core.llm_configs import Provider
        >>> api_key = get_api_key(Provider.OPENAI)
    """

    settings = get_settings()

    provider_key_map: dict[Provider, str] = {
        Provider.OPENAI: settings.openai_api_key,
        Provider.AZURE_OPENAI: settings.azure_openai_api_key,
        Provider.OLLAMA: "ollama",  # Ollama doesn't need a real API key
        Provider.HUGGINGFACE_ROUTER: settings.huggingface_api_token,
        Provider.HUGGINGFACE_INFERENCE: settings.huggingface_api_token,
        Provider.GROQ: settings.groq_api_key,
        Provider.TOGETHER: settings.together_api_key,
        Provider.MISTRAL: settings.mistral_api_key,
        Provider.FIREWORKS: settings.fireworks_api_key,
        Provider.ANYSCALE: settings.anyscale_api_key,
        Provider.DEEPSEEK: settings.deepseek_api_key,
        Provider.CEREBRAS: settings.cerebras_api_key,
        Provider.SAMBANOVA: settings.sambanova_api_key,
        Provider.VLLM: "",  # Local VLLM doesn't need API key
    }

    return provider_key_map.get(provider, "")


def get_endpoint(provider: Provider) -> str:
    """Get the endpoint/base_url for a given provider.

    For most providers, returns the static base_url from ProviderConfig.
    For Azure OpenAI, returns the user-configured endpoint from settings.

    Args:
        provider: The LLM provider to get the endpoint for.

    Returns:
        The endpoint URL string.

    Example:
        >>> from agents.llm_core.llm_configs import Provider
        >>> endpoint = get_endpoint(Provider.OPENAI)
        'https://api.openai.com/v1'
    """
    from agents.llm_core.llm_configs import Provider

    # Azure OpenAI has a user-specific endpoint
    if provider == Provider.AZURE_OPENAI:
        settings = get_settings()
        return settings.azure_openai_endpoint

    # All other providers use static base_url from ProviderConfig
    return provider.base_url
