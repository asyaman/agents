"""
Configuration settings for the webapp.

Uses environment variables with sensible defaults.
Minimal configuration - only what's needed for Chainlit + AgentTool.
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Webapp settings loaded from environment variables."""

    # Agent selection
    AGENT_NAME: str = Field(
        description="Name of the agent to run (from registry)",
    )

    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./data/chat_history.db",
        description="SQLAlchemy database URL for chat history",
    )

    # Development mode
    ENVIRONMENT: str = Field(
        default="local",
        description="Environment: local, staging, production",
    )

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "local"

    # Chainlit settings
    CHAINLIT_AUTH_SECRET: str | None = Field(
        default=None,
        description="Secret for Chainlit auth (optional for single-user)",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience accessor
settings = get_settings()


# Ensure data directory exists
def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir
