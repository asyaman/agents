"""Pytest configuration and shared fixtures for llm_tools tests."""

# Import fixtures from common_fixtures to make them available to all tests in this dir
from agents.tools.llm_tools.tests.common_fixtures import (
    mock_llm_client,
    simple_tool,
)

__all__ = ["mock_llm_client", "simple_tool"]
