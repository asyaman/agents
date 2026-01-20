import agents.settings as settings_module
import pytest

from agents.settings import Settings, configure_settings, get_settings


@pytest.fixture(autouse=True)
def reset_settings():
    original = settings_module._settings
    yield
    settings_module._settings = original


def test_default_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("openai_api_key", "test-123")
    # Override .env file value to test defaults
    monkeypatch.setenv("huggingface_api_token".upper(), "")
    settings_module._settings = None  # Reset to force reload
    settings = get_settings()
    assert settings.openai_api_key == "test-123"
    assert settings.huggingface_api_token == ""
    assert settings.langchain_tracing_v2 is False


def test_configure_settings():
    configure_settings(openai_api_key="configured-key")
    assert get_settings().openai_api_key == "configured-key"


def test_settings_equality():
    s1 = Settings(openai_api_key="key1")
    s2 = Settings(openai_api_key="key1")
    s3 = Settings(openai_api_key="key2")
    assert s1 == s2
    assert s1 != s3
