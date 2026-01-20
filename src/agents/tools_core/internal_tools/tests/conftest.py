"""Pytest configuration and shared fixtures for tools_core tests."""

# Import fixtures from common_fixtures to make them available to all tests
from agents.tools_core.internal_tools.tests.common_fixtures import (
    mock_llm_client,
    simple_tool,
)

__all__ = ["mock_llm_client", "simple_tool"]
