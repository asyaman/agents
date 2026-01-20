"""Tests for ReactStrategy."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.react_strategy import ReactStrategy
from agents.llm_core.llm_client import ToolCall, ToolCallResponse, TextResponse
from agents.agent_tool.tests.common_fixtures import SearchTool


class TestReactStrategy:
    """Tests for ReactStrategy."""

    def test_init_defaults(self, mock_llm_client: MagicMock):
        strategy = ReactStrategy(action_client=mock_llm_client)
        assert strategy.reasoning_prompt is None
        assert strategy.action_prompt is None
        assert strategy.finish_tool_name == "finish"
        # reasoning_client defaults to action_client
        assert strategy.reasoning_client is mock_llm_client

    def test_init_custom_prompts(self, mock_llm_client: MagicMock):
        strategy = ReactStrategy(
            action_client=mock_llm_client,
            reasoning_prompt="Think carefully",
            action_prompt="Now act",
            finish_tool_name="done",
        )
        assert strategy.reasoning_prompt == "Think carefully"
        assert strategy.action_prompt == "Now act"
        assert strategy.finish_tool_name == "done"

    @pytest.mark.asyncio
    async def test_plan_two_phase_execution(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that React does reasoning then action."""
        strategy = ReactStrategy(
            action_client=mock_llm_client,
            reasoning_prompt="Think step by step",
            action_prompt="Select a tool",
        )

        # First call: reasoning (text response)
        # Second call: action (tool call)
        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="I should search for information first."),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="test-id-1", tool_name="search", arguments={"query": "test"}
                    )
                ]
            ),
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Find information"}],
            tools=[search_tool],
        )

        # Verify two LLM calls were made
        assert mock_llm_client.agenerate.call_count == 2

        # Check result includes reasoning in messages
        assert not result.finished
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "I should search for information first."

        # Check tool calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_plan_with_finish_includes_reasoning(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that finish includes reasoning message."""
        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="The task is complete."),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="test-id-1",
                        tool_name="finish",
                        arguments={"result": "Done!", "success": True},
                    )
                ]
            ),
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Complete task"}],
            tools=[search_tool],
        )

        assert result.finished
        assert result.result == "Done!"
        assert result.success is True
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "The task is complete."

    @pytest.mark.asyncio
    async def test_plan_no_tool_calls_returns_unsuccessful_finish(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that no tool calls after reasoning = unsuccessful finish."""
        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="I cannot proceed with this task."),
            ToolCallResponse(tool_calls=[]),  # No tool calls
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Do something"}],
            tools=[search_tool],
        )

        assert result.finished
        assert result.success is False  # No tool calls = unsuccessful
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "I cannot proceed with this task."
