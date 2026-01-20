"""Tests for AdaptiveReflexionStrategy."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.adaptive_reflexion_strategy import AdaptiveReflexionStrategy
from agents.agent_tool.planning_strategies import StrategyOutput
from agents.llm_core.llm_client import ToolCall, ToolCallResponse
from agents.agent_tool.tests.common_fixtures import SearchTool


class TestAdaptiveReflexionStrategy:
    """Tests for AdaptiveReflexionStrategy."""

    def test_init_defaults(self, mock_llm_client: MagicMock):
        strategy = AdaptiveReflexionStrategy(llm_client=mock_llm_client)
        assert strategy.max_reflections == 2
        assert strategy.max_direct_attempts == 2
        assert strategy.finish_tool_name == "finish"

    def test_init_custom_params(self, mock_llm_client: MagicMock):
        strategy = AdaptiveReflexionStrategy(
            llm_client=mock_llm_client,
            max_reflections=5,
            max_direct_attempts=3,
        )
        assert strategy.max_reflections == 5
        assert strategy.max_direct_attempts == 3

    @pytest.mark.asyncio
    async def test_plan_returns_tool_calls(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test plan returns tool calls normally."""
        strategy = AdaptiveReflexionStrategy(llm_client=mock_llm_client)

        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(id="1", tool_name="search", arguments={"query": "test"})
            ]
        )

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Search"}],
            tools=[search_tool],
        )

        assert isinstance(result, StrategyOutput)
        assert len(result.tool_calls) == 1
        assert not result.finished

    @pytest.mark.asyncio
    async def test_plan_finish_tool_completes(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test finish tool signals completion."""
        strategy = AdaptiveReflexionStrategy(llm_client=mock_llm_client)

        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="1",
                    tool_name="finish",
                    arguments={"result": "Done", "success": True},
                )
            ]
        )

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Complete"}],
            tools=[search_tool],
        )

        assert result.finished
        assert result.success is True
        assert result.result == "Done"
