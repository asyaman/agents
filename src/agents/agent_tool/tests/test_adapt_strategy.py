"""Tests for AdaptStrategy."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.adapt_strategy import AdaptStrategy
from agents.agent_tool.base_strategy import StrategyOutput
from agents.agent_tool.tests.common_fixtures import SearchTool
from agents.llm_core.llm_client import ToolCall, ToolCallResponse


class TestAdaptStrategy:
    """Tests for AdaptStrategy."""

    def test_init_defaults(self, mock_llm_client: MagicMock):
        strategy = AdaptStrategy(llm_client=mock_llm_client)
        assert strategy.max_direct_attempts == 3
        assert strategy.error_threshold == 2
        assert strategy.stagnation_window == 2

    def test_init_custom_params(self, mock_llm_client: MagicMock):
        strategy = AdaptStrategy(
            llm_client=mock_llm_client,
            max_direct_attempts=5,
            error_threshold=3,
            stagnation_window=4,
        )
        assert strategy.max_direct_attempts == 5
        assert strategy.error_threshold == 3
        assert strategy.stagnation_window == 4

    @pytest.mark.asyncio
    async def test_plan_returns_tool_calls(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test plan returns tool calls normally."""
        strategy = AdaptStrategy(llm_client=mock_llm_client)

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

    @pytest.mark.asyncio
    async def test_plan_finish_tool_passes_through(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Strategy passes finish through; AgentTool detects + terminates."""
        strategy = AdaptStrategy(llm_client=mock_llm_client)

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

        # finish stays in tool_calls; strategy doesn't extract args.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "finish"
        assert result.tool_calls[0].arguments["result"] == "Done"
