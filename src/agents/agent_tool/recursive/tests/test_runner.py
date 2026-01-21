"""Tests for RecursiveAgentRunner."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from agents.agent_tool.base_strategy import PlanningStrategy
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.agent_tool.recursive.context import (
    LevelExecution,
    RecursionStatistics,
)
from agents.agent_tool.recursive.runner import (
    RecursiveAgentOutput,
    RecursiveAgentRunner,
    run_recursive_agent,
)
from agents.agent_tool.tests.common_fixtures import SearchTool
from agents.llm_core.llm_client import ToolCall, ToolCallResponse
from agents.tools_core.base_tool import BaseTool


class TestRecursiveAgentOutput:
    """Tests for RecursiveAgentOutput model."""

    def test_create_output(self):
        stats = RecursionStatistics(
            max_depth_reached=2,
            total_iterations=5,
            total_tool_calls=3,
            levels_completed=2,
        )
        history = [
            LevelExecution(
                depth=0,
                objective="Root task",
                iterations_used=3,
                success=True,
                result="Done",
            )
        ]

        output = RecursiveAgentOutput(
            result="Final result",
            success=True,
            iterations_used=3,
            statistics=stats,
            execution_history=history,
        )

        assert output.result == "Final result"
        assert output.success is True
        assert output.iterations_used == 3
        assert output.statistics.max_depth_reached == 2
        assert len(output.execution_history) == 1


class TestRecursiveAgentRunnerInit:
    """Tests for RecursiveAgentRunner initialization."""

    def test_init_with_required_args(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
    ):
        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=mock_strategy_factory,
        )
        assert runner.tools == simple_tools
        assert runner.tool_selector is None
        assert runner.max_depth == 3
        assert runner.max_iterations_per_level == 5
        assert runner.system_prompt is None
        assert runner.parallel_tool_calls is True
        assert runner.include_sub_agent_at_root is True

    def test_init_with_all_args(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
        mock_tool_selector: MagicMock,
    ):
        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=mock_strategy_factory,
            tool_selector=mock_tool_selector,
            max_depth=5,
            max_iterations_per_level=15,
            system_prompt="Custom system prompt",
            parallel_tool_calls=False,
            include_sub_agent_at_root=False,
        )
        assert runner.tool_selector is mock_tool_selector
        assert runner.max_depth == 5
        assert runner.max_iterations_per_level == 15
        assert runner.system_prompt == "Custom system prompt"
        assert runner.parallel_tool_calls is False
        assert runner.include_sub_agent_at_root is False


class TestRecursiveAgentRunnerExecution:
    """Tests for RecursiveAgentRunner execution."""

    @pytest.mark.asyncio
    async def test_run_simple_task(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test running a simple task that completes immediately."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            max_depth=3,
            include_sub_agent_at_root=False,
        )

        result = await runner.run(objective="Simple task", context="Some context")

        assert isinstance(result, RecursiveAgentOutput)
        assert result.success is True
        assert result.result == "Task completed successfully"
        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_run_returns_execution_history(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that execution history includes root level."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            include_sub_agent_at_root=False,
        )

        result = await runner.run(objective="Test objective")

        assert len(result.execution_history) >= 1
        root_execution = result.execution_history[0]
        assert root_execution.depth == 0
        assert root_execution.objective == "Test objective"

    @pytest.mark.asyncio
    async def test_run_updates_statistics(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that statistics are updated after execution."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            include_sub_agent_at_root=False,
        )

        result = await runner.run(objective="Test")

        assert result.statistics.levels_completed >= 1
        assert result.statistics.total_iterations >= 1

    @pytest.mark.asyncio
    async def test_run_includes_sub_agent_tool_at_root(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that SubAgentTool is included when enabled."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            max_depth=3,
            include_sub_agent_at_root=True,
        )

        await runner.run(objective="Test")

        # Check that tools passed to LLM include SubAgentTool
        call_args = mock_llm_client.agenerate.call_args
        tools_arg = call_args.kwargs["tools"]
        tool_names = [t.name.upper() for t in tools_arg]
        assert "DELEGATE_SUBTASK" in tool_names

    @pytest.mark.asyncio
    async def test_run_excludes_sub_agent_when_disabled(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that SubAgentTool is excluded when disabled."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            include_sub_agent_at_root=False,
        )

        await runner.run(objective="Test")

        call_args = mock_llm_client.agenerate.call_args
        tools_arg = call_args.kwargs["tools"]
        tool_names = [t.name for t in tools_arg]
        assert "delegate_subtask" not in tool_names

    @pytest.mark.asyncio
    async def test_run_excludes_sub_agent_when_max_depth_1(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that SubAgentTool is excluded when max_depth=1."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            max_depth=1,
            include_sub_agent_at_root=True,
        )

        await runner.run(objective="Test")

        call_args = mock_llm_client.agenerate.call_args
        tools_arg = call_args.kwargs["tools"]
        tool_names = [t.name for t in tools_arg]
        # SubAgentTool should not be included when max_depth <= 1
        assert "delegate_subtask" not in tool_names

    @pytest.mark.asyncio
    async def test_run_uses_tool_selector(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        mock_tool_selector: MagicMock,
        finish_tool_call: ToolCallResponse,
        search_tool: SearchTool,
    ):
        """Test that tool selector filters root tools."""
        mock_llm_client.agenerate.return_value = finish_tool_call
        mock_tool_selector.afilter_tools.return_value = [search_tool]

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            tool_selector=mock_tool_selector,
            include_sub_agent_at_root=False,
        )

        await runner.run(objective="Search for something")

        mock_tool_selector.afilter_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_failed_task(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
    ):
        """Test handling of failed tasks."""
        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="finish-1",
                    tool_name="finish",
                    arguments={"result": "Could not complete", "success": False},
                )
            ]
        )

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            include_sub_agent_at_root=False,
        )

        result = await runner.run(objective="Failing task")

        assert result.success is False
        assert result.result == "Could not complete"


class TestRunSyncMethod:
    """Tests for synchronous run method."""

    def test_run_sync(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test synchronous wrapper."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            include_sub_agent_at_root=False,
        )

        result = runner.run_sync(objective="Sync test")

        assert isinstance(result, RecursiveAgentOutput)
        assert result.success is True


class TestConvenienceFunction:
    """Tests for run_recursive_agent convenience function."""

    @pytest.mark.asyncio
    async def test_run_recursive_agent(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test convenience function."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        result = await run_recursive_agent(
            objective="Test objective",
            tools=simple_tools,
            strategy_factory=strategy_factory,
            max_depth=2,
        )

        assert isinstance(result, RecursiveAgentOutput)
        assert result.success is True


class TestGuidanceInjection:
    """Tests for sub-agent guidance injection."""

    @pytest.mark.asyncio
    async def test_guidance_injected_when_sub_agent_included(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that guidance is injected when sub-agent is included."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            max_depth=3,
            include_sub_agent_at_root=True,
        )

        result = await runner.run(objective="Test")

        # Check that messages include guidance about delegate_subtask
        system_msgs = [
            msg
            for msg in result.execution_history[0].messages
            if msg.get("role") == "system"
        ]
        # Should have more than 1 system message when guidance is included
        assert len(system_msgs) >= 2
        # One of them should mention delegate_subtask
        all_content = " ".join(msg.get("content", "") for msg in system_msgs)
        assert "delegate_subtask" in all_content.lower()

    @pytest.mark.asyncio
    async def test_no_guidance_when_sub_agent_disabled(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that no guidance is injected when sub-agent is disabled."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            include_sub_agent_at_root=False,
        )

        result = await runner.run(objective="Test")

        # Check that there's only 1 system message (no guidance)
        system_msgs = [
            msg
            for msg in result.execution_history[0].messages
            if msg.get("role") == "system"
        ]
        assert len(system_msgs) == 1

    @pytest.mark.asyncio
    async def test_no_guidance_when_max_depth_1(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that no guidance is injected when max_depth=1."""
        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        runner = RecursiveAgentRunner(
            tools=simple_tools,
            strategy_factory=strategy_factory,
            max_depth=1,
            include_sub_agent_at_root=True,
        )

        result = await runner.run(objective="Test")

        # SubAgentTool not included when max_depth=1, so no guidance
        system_msgs = [
            msg
            for msg in result.execution_history[0].messages
            if msg.get("role") == "system"
        ]
        assert len(system_msgs) == 1
