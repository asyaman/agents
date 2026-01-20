import copy
import os

import pytest

from agents.settings import Settings

API_KEY_VARS = [
    "openai_api_key",
    "azure_openai_api_key",
    "groq_api_key",
    "together_api_key",
    "mistral_api_key",
    "fireworks_api_key",
    "anyscale_api_key",
    "deepseek_api_key",
    "cerebras_api_key",
    "sambanova_api_key",
    "tavily_api_key",
    "huggingface_api_token",
    "langchain_api_key",
]


@pytest.fixture(scope="session", autouse=True)
def clear_api_keys():
    """Clear API keys to prevent accidental use during testing."""
    saved = {k: os.environ.pop(k, None) for k in API_KEY_VARS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture(scope="session", autouse=True)
def disable_env_file():
    """Prevent .env file from being loaded during tests."""
    original = copy.copy(Settings.model_config)
    Settings.model_config["env_file"] = None
    yield
    Settings.model_config = original
