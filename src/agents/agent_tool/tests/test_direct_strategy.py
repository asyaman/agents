"""Tests for DirectStrategy."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.base_strategy import StrategyOutput
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.llm_core.llm_client import ToolCall, ToolCallResponse
from agents.agent_tool.tests.common_fixtures import SearchTool


class TestDirectStrategy:
    """Tests for DirectStrategy."""

    def test_init_default_finish_tool(self, mock_llm_client: MagicMock):
        strategy = DirectStrategy(llm_client=mock_llm_client)
        assert strategy.finish_tool_name == "finish"

    def test_init_custom_finish_tool(self, mock_llm_client: MagicMock):
        strategy = DirectStrategy(
            llm_client=mock_llm_client, finish_tool_name="complete"
        )
        assert strategy.finish_tool_name == "complete"

    def test_init_default_direct_prompt(self, mock_llm_client: MagicMock):
        strategy = DirectStrategy(llm_client=mock_llm_client)
        assert strategy.direct_prompt is None

    def test_init_custom_direct_prompt(self, mock_llm_client: MagicMock):
        strategy = DirectStrategy(
            llm_client=mock_llm_client, direct_prompt="Custom prompt here"
        )
        assert strategy.direct_prompt == "Custom prompt here"

    @pytest.mark.asyncio
    async def test_plan_uses_custom_direct_prompt(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that custom direct_prompt is passed to LLM."""
        custom_prompt = "Execute the next action based on the objective."
        strategy = DirectStrategy(
            llm_client=mock_llm_client, direct_prompt=custom_prompt
        )

        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="test-id-1", tool_name="search", arguments={"query": "test"}
                )
            ]
        )

        await strategy.plan(
            messages=[{"role": "user", "content": "Search for test"}],
            tools=[search_tool],
        )

        # Verify the custom prompt was included in messages
        call_args = mock_llm_client.agenerate.call_args
        messages_arg = call_args.kwargs["messages"]
        # Last message should be the direct prompt
        assert messages_arg[-1]["role"] == "user"
        assert messages_arg[-1]["content"] == custom_prompt

    @pytest.mark.asyncio
    async def test_plan_with_tool_calls(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that tool calls are returned for execution."""
        strategy = DirectStrategy(llm_client=mock_llm_client)

        # Mock LLM returning a tool call
        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="test-id-1", tool_name="search", arguments={"query": "test"}
                )
            ]
        )

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Search for test"}],
            tools=[search_tool],
        )

        assert isinstance(result, StrategyOutput)
        assert not result.finished
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_plan_with_finish_tool(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that finish tool signals completion."""
        strategy = DirectStrategy(llm_client=mock_llm_client)

        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="test-id-1",
                    tool_name="finish",
                    arguments={"result": "Task done", "success": True},
                )
            ]
        )

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Complete task"}],
            tools=[search_tool],
        )

        assert result.finished
        assert result.result == "Task done"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_plan_with_finish_tool_success_false(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that finish tool with success=false propagates correctly."""
        strategy = DirectStrategy(llm_client=mock_llm_client)

        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="test-id-1",
                    tool_name="finish",
                    arguments={"result": "Task failed", "success": False},
                )
            ]
        )

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Complete task"}],
            tools=[search_tool],
        )

        assert result.finished
        assert result.result == "Task failed"
        assert result.success is False

    @pytest.mark.asyncio
    async def test_plan_no_tool_calls_returns_unsuccessful_finish(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that empty tool calls = unsuccessful finish (LLM didn't use finish tool)."""
        strategy = DirectStrategy(llm_client=mock_llm_client)

        # ToolCallResponse with empty tool_calls list
        mock_llm_client.agenerate.return_value = ToolCallResponse(tool_calls=[])

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Do something"}],
            tools=[search_tool],
        )

        assert result.finished
        assert result.success is False  # No tool calls = unsuccessful
        assert result.result == "No tool calls returned by LLM"
