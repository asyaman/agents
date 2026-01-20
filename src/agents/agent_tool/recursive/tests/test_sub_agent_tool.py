"""Tests for SubAgentTool."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from agents.agent_tool.planning_strategies import PlanningStrategy
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.agent_tool.recursive.context import (
    get_current_depth,
    initialize_recursion_context,
    var_recursion_depth,
)
from agents.agent_tool.recursive.sub_agent_tool import (
    SubAgentInput,
    SubAgentOutput,
    SubAgentTool,
)
from agents.agent_tool.tests.common_fixtures import SearchTool
from agents.llm_core.llm_client import ToolCall, ToolCallResponse
from agents.tools_core.base_tool import BaseTool


class TestSubAgentInput:
    """Tests for SubAgentInput model."""

    def test_create_with_objective_only(self):
        input = SubAgentInput(sub_objective="Do something")
        assert input.sub_objective == "Do something"
        assert input.context is None

    def test_create_with_context(self):
        input = SubAgentInput(
            sub_objective="Research topic", context="Focus on recent developments"
        )
        assert input.sub_objective == "Research topic"
        assert input.context == "Focus on recent developments"


class TestSubAgentOutput:
    """Tests for SubAgentOutput model."""

    def test_create_success(self):
        output = SubAgentOutput(result="Task completed", success=True)
        assert output.result == "Task completed"
        assert output.success is True

    def test_create_failure(self):
        output = SubAgentOutput(result="Failed to complete", success=False)
        assert output.result == "Failed to complete"
        assert output.success is False


class TestSubAgentToolInit:
    """Tests for SubAgentTool initialization."""

    def test_init_with_required_args(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
    ):
        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=mock_strategy_factory,
        )
        assert tool.available_tools == simple_tools
        assert tool.tool_selector is None
        assert tool.max_iterations_per_level == 10
        assert tool.include_self_in_children is True
        assert tool.system_prompt is None
        assert tool.parallel_tool_calls is True

    def test_init_with_all_args(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
        mock_tool_selector: MagicMock,
    ):
        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=mock_strategy_factory,
            tool_selector=mock_tool_selector,
            max_iterations_per_level=5,
            include_self_in_children=False,
            system_prompt="Custom prompt",
            parallel_tool_calls=False,
        )
        assert tool.tool_selector is mock_tool_selector
        assert tool.max_iterations_per_level == 5
        assert tool.include_self_in_children is False
        assert tool.system_prompt == "Custom prompt"
        assert tool.parallel_tool_calls is False

    def test_tool_metadata(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
    ):
        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=mock_strategy_factory,
        )
        assert tool.name.upper() == "DELEGATE_SUBTASK"
        assert "delegate" in tool.description.lower()
        assert tool._input == SubAgentInput
        assert tool._output == SubAgentOutput


class TestSubAgentToolExecution:
    """Tests for SubAgentTool execution."""

    @pytest.mark.asyncio
    async def test_returns_max_depth_error_when_at_limit(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
    ):
        """Test that max depth is enforced."""
        initialize_recursion_context(max_depth=2)
        var_recursion_depth.set(2)  # Already at max

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=mock_strategy_factory,
        )

        result = await tool.ainvoke(SubAgentInput(sub_objective="Should not execute"))

        assert result.success is False
        assert "maximum recursion depth" in result.result.lower()

    @pytest.mark.asyncio
    async def test_increments_depth_during_execution(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that depth is incremented and restored."""
        initialize_recursion_context(max_depth=3)
        var_recursion_depth.set(0)

        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            include_self_in_children=False,
        )

        assert get_current_depth() == 0

        await tool.ainvoke(SubAgentInput(sub_objective="Test task"))

        # Depth should be restored after execution
        assert get_current_depth() == 0

    @pytest.mark.asyncio
    async def test_records_execution_to_history(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that execution is recorded to global history."""
        history, stats = initialize_recursion_context(max_depth=3)

        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            include_self_in_children=False,
        )

        await tool.ainvoke(
            SubAgentInput(sub_objective="Record this task", context="With context")
        )

        assert len(history) == 1
        assert history[0].objective == "Record this task"
        assert history[0].depth == 1
        assert history[0].success is True
        assert stats.levels_completed == 1

    @pytest.mark.asyncio
    async def test_returns_opaque_output(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
    ):
        """Test that output is opaque (no internal messages exposed)."""
        initialize_recursion_context(max_depth=3)

        mock_llm_client.agenerate.return_value = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="finish-1",
                    tool_name="finish",
                    arguments={"result": "The answer is 42", "success": True},
                )
            ]
        )

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            include_self_in_children=False,
        )

        result = await tool.ainvoke(SubAgentInput(sub_objective="Calculate something"))

        # Output should only have result and success
        assert isinstance(result, SubAgentOutput)
        assert result.result == "The answer is 42"
        assert result.success is True
        # No messages attribute exposed
        assert not hasattr(result, "messages")

    @pytest.mark.asyncio
    async def test_uses_tool_selector_when_provided(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        mock_tool_selector: MagicMock,
        finish_tool_call: ToolCallResponse,
        search_tool: SearchTool,
    ):
        """Test that tool selector is used to filter tools."""
        initialize_recursion_context(max_depth=3)

        mock_llm_client.agenerate.return_value = finish_tool_call
        mock_tool_selector.afilter_tools.return_value = [search_tool]

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            tool_selector=mock_tool_selector,
            include_self_in_children=False,
        )

        await tool.ainvoke(SubAgentInput(sub_objective="Search for something"))

        mock_tool_selector.afilter_tools.assert_called_once()
        call_args = mock_tool_selector.afilter_tools.call_args
        assert call_args.kwargs["objective"] == "Search for something"

    @pytest.mark.asyncio
    async def test_adds_self_to_children_when_enabled(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that self is added to child tools when include_self_in_children=True."""
        initialize_recursion_context(max_depth=3)
        var_recursion_depth.set(0)

        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            include_self_in_children=True,
        )

        await tool.ainvoke(SubAgentInput(sub_objective="Test"))

        # Check that agenerate was called with tools including SubAgentTool
        call_args = mock_llm_client.agenerate.call_args
        tools_arg = call_args.kwargs["tools"]
        tool_names = [t.name.upper() for t in tools_arg]
        assert "DELEGATE_SUBTASK" in tool_names

    @pytest.mark.asyncio
    async def test_does_not_add_self_at_max_depth_minus_one(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that self is not added when at max_depth - 1."""
        initialize_recursion_context(max_depth=2)
        var_recursion_depth.set(1)  # At max_depth - 1

        mock_llm_client.agenerate.return_value = finish_tool_call

        def strategy_factory():
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            include_self_in_children=True,
        )

        await tool.ainvoke(SubAgentInput(sub_objective="Test"))

        # Should not include self since we're at depth 1 and max is 2
        call_args = mock_llm_client.agenerate.call_args
        tools_arg = call_args.kwargs["tools"]
        tool_names = [t.name for t in tools_arg]
        assert "delegate_subtask" not in tool_names

    @pytest.mark.asyncio
    async def test_creates_fresh_strategy_per_invocation(
        self,
        simple_tools: list[BaseTool],
        mock_llm_client: MagicMock,
        finish_tool_call: ToolCallResponse,
    ):
        """Test that strategy factory is called for each invocation."""
        initialize_recursion_context(max_depth=3)

        mock_llm_client.agenerate.return_value = finish_tool_call
        factory_call_count = 0

        def strategy_factory():
            nonlocal factory_call_count
            factory_call_count += 1
            return DirectStrategy(llm_client=mock_llm_client)

        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=strategy_factory,
            include_self_in_children=False,
        )

        await tool.ainvoke(SubAgentInput(sub_objective="First"))
        await tool.ainvoke(SubAgentInput(sub_objective="Second"))

        assert factory_call_count == 2


class TestSubAgentToolInputOutput:
    """Tests for SubAgentTool input/output models."""

    def test_input_schema(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
    ):
        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=mock_strategy_factory,
        )
        schema = tool.input_schema()
        assert "sub_objective" in schema["properties"]
        assert "context" in schema["properties"]

    def test_output_schema(
        self,
        simple_tools: list[BaseTool],
        mock_strategy_factory: Callable[[], PlanningStrategy],
    ):
        tool = SubAgentTool(
            available_tools=simple_tools,
            strategy_factory=mock_strategy_factory,
        )
        schema = tool.output_schema()
        assert "result" in schema["properties"]
        assert "success" in schema["properties"]


class TestSubAgentToolGuidance:
    """Tests for SubAgentTool guidance prompt functionality."""

    def test_get_guidance_prompt_returns_string(self):
        """Test that get_guidance_prompt returns a non-empty string."""
        guidance = SubAgentTool.get_guidance_prompt()
        assert isinstance(guidance, str)
        assert len(guidance) > 0

    def test_guidance_prompt_contains_key_sections(self):
        """Test that guidance prompt contains expected content."""
        guidance = SubAgentTool.get_guidance_prompt()
        # Should mention the tool name
        assert "delegate_subtask" in guidance.lower()
        # Should explain when to use
        assert "when to use" in guidance.lower()
        # Should explain when NOT to use
        assert "when not to use" in guidance.lower() or "not to" in guidance.lower()

    def test_guidance_prompt_is_class_method(self):
        """Test that get_guidance_prompt works as a class method."""
        # Should work without instantiation
        guidance1 = SubAgentTool.get_guidance_prompt()

        # Should also work from instance
        from agents.agent_tool.direct_strategy import DirectStrategy
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        tool = SubAgentTool(
            available_tools=[],
            strategy_factory=lambda: DirectStrategy(llm_client=mock_client),
        )
        guidance2 = tool.get_guidance_prompt()

        assert guidance1 == guidance2
