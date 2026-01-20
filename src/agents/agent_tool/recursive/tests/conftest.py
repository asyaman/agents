"""Pytest configuration and shared fixtures for recursive agent tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.agent_tool.planning_strategies import PlanningStrategy
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.agent_tool.tests.common_fixtures import (
    CalculatorTool,
    SearchTool,
    calculator_tool,
    mock_llm_client,
    search_tool,
)
from agents.llm_core.llm_client import ToolCall, ToolCallResponse

# Re-export fixtures from common_fixtures
__all__ = [
    "mock_llm_client",
    "search_tool",
    "calculator_tool",
    "SearchTool",
    "CalculatorTool",
    "mock_strategy_factory",
    "mock_tool_selector",
    "simple_tools",
]


@pytest.fixture
def mock_strategy_factory(mock_llm_client: MagicMock):
    """Create a factory that returns DirectStrategy with mock client."""

    def factory() -> PlanningStrategy:
        return DirectStrategy(llm_client=mock_llm_client)

    return factory


@pytest.fixture
def mock_tool_selector(mock_llm_client: MagicMock):
    """Create a mock ToolSelector."""
    from agents.tools_core.internal_tools.tool_selector import ToolSelector

    selector = MagicMock(spec=ToolSelector)
    selector.afilter_tools = AsyncMock()
    return selector


@pytest.fixture
def simple_tools(search_tool: SearchTool, calculator_tool: CalculatorTool):
    """Create a list of simple tools for testing."""
    return [search_tool, calculator_tool]


@pytest.fixture
def finish_tool_call():
    """Create a finish tool call response."""
    return ToolCallResponse(
        tool_calls=[
            ToolCall(
                id="finish-1",
                tool_name="finish",
                arguments={"result": "Task completed successfully", "success": True},
            )
        ]
    )


@pytest.fixture
def search_tool_call():
    """Create a search tool call response."""
    return ToolCallResponse(
        tool_calls=[
            ToolCall(
                id="search-1",
                tool_name="search",
                arguments={"query": "test query"},
            )
        ]
    )
