"""Tests for ReflexionStrategy."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.base_strategy import StrategyOutput
from agents.agent_tool.reflexion_strategy import (
    ReflectionInsight,
    ReflexionMemory,
    ReflexionStrategy,
)
from agents.agent_tool.tests.common_fixtures import SearchTool
from agents.llm_core.llm_client import ToolCall, ToolCallResponse


class TestReflexionMemory:
    """Tests for ReflexionMemory."""

    def test_add_insight(self):
        memory = ReflexionMemory()
        insight = ReflectionInsight(
            iteration=1,
            failure_description="Query failed",
            insight="Try more specific terms",
        )
        memory.add_insight(insight)

        assert len(memory.task_insights) == 1
        assert memory.task_insights[0].insight == "Try more specific terms"

    def test_get_context_prompt_empty(self):
        memory = ReflexionMemory()
        prompt = memory.get_context_prompt()
        assert prompt == ""

    def test_get_context_prompt_with_insights(self):
        memory = ReflexionMemory()
        insight = ReflectionInsight(
            iteration=1,
            failure_description="Error",
            insight="Learned something",
        )
        memory.add_insight(insight)

        prompt = memory.get_context_prompt()
        assert "Learned something" in prompt

    def test_reset_task(self):
        memory = ReflexionMemory()
        insight = ReflectionInsight(
            iteration=1,
            failure_description="Error",
            insight="Test",
        )
        memory.add_insight(insight)
        memory.reset_task()

        assert memory.task_insights == []


class TestReflexionStrategy:
    """Tests for ReflexionStrategy."""

    def test_init_defaults(self, mock_llm_client: MagicMock):
        strategy = ReflexionStrategy(llm_client=mock_llm_client)
        assert strategy.max_reflections == 3
        assert strategy.persist_insights is False

    def test_init_custom_params(self, mock_llm_client: MagicMock):
        strategy = ReflexionStrategy(
            llm_client=mock_llm_client,
            max_reflections=5,
            persist_insights=True,
        )
        assert strategy.max_reflections == 5
        assert strategy.persist_insights is True

    @pytest.mark.asyncio
    async def test_plan_returns_tool_calls(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test plan returns tool calls normally."""
        strategy = ReflexionStrategy(llm_client=mock_llm_client)

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
        strategy = ReflexionStrategy(llm_client=mock_llm_client)

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

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "finish"
        assert result.tool_calls[0].arguments["result"] == "Done"
