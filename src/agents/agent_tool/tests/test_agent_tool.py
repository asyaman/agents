"""Tests for AgentTool."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.agent_tool import (
    AgentTool,
    AgentToolInput,
    AgentToolOutput,
    create_finish_tool,
)
from agents.agent_tool.base_strategy import StrategyOutput, PlanningStrategy
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.llm_core.llm_client import ToolCall
from agents.agent_tool.tests.common_fixtures import (
    SearchTool,
    CalculatorTool,
    SearchInput,
    CalculatorInput,
)


class MockStrategy(PlanningStrategy):
    """Mock strategy for testing AgentTool."""

    def __init__(self, outputs: list[StrategyOutput]):
        self.outputs = outputs
        self.call_count = 0

    async def plan(self, messages, tools, parallel_tool_calls=True) -> StrategyOutput:
        output = self.outputs[self.call_count]
        self.call_count += 1
        return output


class TestAgentToolInput:
    """Tests for AgentToolInput model."""

    def test_required_objective(self):
        input = AgentToolInput(objective="Do something")
        assert input.objective == "Do something"
        assert input.context is None
        assert input.max_iterations == 10

    def test_with_context(self):
        input = AgentToolInput(
            objective="Do something",
            context="Additional info",
            max_iterations=5,
        )
        assert input.context == "Additional info"
        assert input.max_iterations == 5


class TestAgentToolOutput:
    """Tests for AgentToolOutput model."""

    def test_basic_output(self):
        output = AgentToolOutput(
            result="Done",
            success=True,
            iterations_used=3,
        )
        assert output.result == "Done"
        assert output.success
        assert output.iterations_used == 3
        assert output.messages == []


class TestCreateFinishTool:
    """Tests for finish tool creation."""

    def test_finish_tool_attributes(self):
        finish = create_finish_tool()
        assert finish.name == "FINISH"  # normalized to uppercase
        assert (
            "finish" in finish.description.lower()
            or "complete" in finish.description.lower()
        )

    def test_finish_tool_invoke(self):
        finish = create_finish_tool()
        result = finish.invoke({"result": "Task done", "success": True})
        assert result.acknowledged


class TestAgentTool:
    """Tests for AgentTool."""

    def test_init_with_strategy(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        strategy = DirectStrategy(llm_client=mock_llm_client)
        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
        )
        assert agent.strategy is strategy
        # Should have search_tool + auto-added finish tool
        assert len(agent.tools) == 2

    def test_init_without_finish_tool(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        agent = AgentTool(
            tools=[search_tool],
            strategy=DirectStrategy(llm_client=mock_llm_client),
            include_finish_tool=False,
        )
        assert len(agent.tools) == 1

    @pytest.mark.asyncio
    async def test_agent_loop_finishes_on_first_iteration(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that agent finishes when strategy returns finished=True."""
        strategy = MockStrategy(
            outputs=[StrategyOutput(finished=True, result="Immediate result")]
        )

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
        )

        result = await agent.ainvoke(AgentToolInput(objective="Simple task"))

        assert result.success
        assert result.result == "Immediate result"
        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_agent_loop_propagates_success_false(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that agent propagates success=False from strategy."""
        strategy = MockStrategy(
            outputs=[StrategyOutput(finished=True, success=False, result="Task failed")]
        )

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
        )

        result = await agent.ainvoke(AgentToolInput(objective="Failing task"))

        assert not result.success
        assert result.result == "Task failed"
        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_agent_loop_executes_tools(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that agent executes tool calls from strategy."""
        strategy = MockStrategy(
            outputs=[
                # First iteration: call search tool
                StrategyOutput(
                    tool_calls=[
                        ToolCall(
                            tool_name="search",
                            arguments={"query": "test"},
                            id="call-1",
                            parsed=SearchInput(query="test"),
                        )
                    ]
                ),
                # Second iteration: finish
                StrategyOutput(finished=True, result="Found results"),
            ]
        )

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
        )

        result = await agent.ainvoke(AgentToolInput(objective="Search for test"))

        assert result.success
        assert result.result == "Found results"
        assert result.iterations_used == 2
        # Messages should include tool call and result
        assert len(result.messages) > 2  # system, user, + tool messages

    @pytest.mark.asyncio
    async def test_agent_loop_max_iterations(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that agent stops at max iterations."""
        # Strategy never finishes
        strategy = MockStrategy(
            outputs=[
                StrategyOutput(
                    tool_calls=[
                        ToolCall(
                            tool_name="search",
                            arguments={"query": "test"},
                            id=f"call-{i}",
                            parsed=SearchInput(query="test"),
                        )
                    ]
                )
                for i in range(10)
            ]
        )

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
        )

        result = await agent.ainvoke(
            AgentToolInput(objective="Infinite task", max_iterations=3)
        )

        assert not result.success
        assert result.iterations_used == 3
        assert "max iterations" in result.result.lower()

    @pytest.mark.asyncio
    async def test_agent_adds_strategy_messages(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that strategy-generated messages are added to history."""
        strategy = MockStrategy(
            outputs=[
                StrategyOutput(
                    messages=[{"role": "assistant", "content": "Let me think..."}],
                    tool_calls=[
                        ToolCall(
                            tool_name="search",
                            arguments={"query": "test"},
                            id="call-1",
                            parsed=SearchInput(query="test"),
                        )
                    ],
                ),
                StrategyOutput(finished=True, result="Found it"),
            ]
        )

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
        )

        result = await agent.ainvoke(AgentToolInput(objective="Think and search"))

        # Check that reasoning message is in history
        assert any(
            msg.get("content") == "Let me think..."
            for msg in result.messages
            if msg.get("role") == "assistant"
        )

    @pytest.mark.asyncio
    async def test_agent_executes_parallel_tool_calls(
        self,
        mock_llm_client: MagicMock,
        search_tool: SearchTool,
        calculator_tool: CalculatorTool,
    ):
        """Test that multiple tool calls are executed in parallel with correct message format."""
        strategy = MockStrategy(
            outputs=[
                # Return multiple tool calls at once
                StrategyOutput(
                    tool_calls=[
                        ToolCall(
                            tool_name="search",
                            arguments={"query": "test"},
                            id="call-1",
                            parsed=SearchInput(query="test"),
                        ),
                        ToolCall(
                            tool_name="calculator",
                            arguments={"expression": "2+2"},
                            id="call-2",
                            parsed=CalculatorInput(expression="2+2"),
                        ),
                    ]
                ),
                StrategyOutput(finished=True, result="Done with parallel calls"),
            ]
        )

        agent = AgentTool(
            tools=[search_tool, calculator_tool],
            strategy=strategy,
        )

        result = await agent.ainvoke(AgentToolInput(objective="Search and calculate"))

        assert result.success
        assert result.iterations_used == 2

        # Find the assistant message with tool_calls
        assistant_msgs_with_tools = [
            msg
            for msg in result.messages
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        ]
        assert len(assistant_msgs_with_tools) == 1

        # Check it has both tool calls in a single message
        tool_calls = list(assistant_msgs_with_tools[0]["tool_calls"])
        assert len(tool_calls) == 2
        tool_names = {tc["function"]["name"] for tc in tool_calls}
        assert tool_names == {"search", "calculator"}

        # Check there are two tool result messages
        tool_result_msgs = [msg for msg in result.messages if msg.get("role") == "tool"]
        assert len(tool_result_msgs) == 2


class TestAgentToolGuidanceMessages:
    """Tests for AgentTool guidance_messages functionality."""

    @pytest.mark.asyncio
    async def test_guidance_messages_injected_into_conversation(
        self, search_tool: SearchTool
    ):
        """Test that guidance messages are injected as system messages."""
        strategy = MockStrategy(outputs=[StrategyOutput(finished=True, result="Done")])

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
            guidance_messages=["This is guidance about tools."],
        )

        result = await agent.ainvoke(AgentToolInput(objective="Test"))

        # Find system messages
        system_msgs = [msg for msg in result.messages if msg.get("role") == "system"]
        # Should have at least 2 system messages (main + guidance)
        assert len(system_msgs) >= 2
        # Guidance should be in one of them
        assert any(
            "This is guidance about tools." in msg.get("content", "")
            for msg in system_msgs
        )

    @pytest.mark.asyncio
    async def test_multiple_guidance_messages(self, search_tool: SearchTool):
        """Test that multiple guidance messages are all injected."""
        strategy = MockStrategy(outputs=[StrategyOutput(finished=True, result="Done")])

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
            guidance_messages=["First guidance.", "Second guidance."],
        )

        result = await agent.ainvoke(AgentToolInput(objective="Test"))

        system_msgs = [msg for msg in result.messages if msg.get("role") == "system"]
        contents = [msg.get("content", "") for msg in system_msgs]

        assert any("First guidance." in c for c in contents)
        assert any("Second guidance." in c for c in contents)

    @pytest.mark.asyncio
    async def test_guidance_messages_appear_before_user_message(
        self, search_tool: SearchTool
    ):
        """Test that guidance messages appear before the user task message."""
        strategy = MockStrategy(outputs=[StrategyOutput(finished=True, result="Done")])

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
            guidance_messages=["Guidance content here."],
        )

        result = await agent.ainvoke(AgentToolInput(objective="Test objective"))

        # Find indices of guidance and user message
        guidance_idx = None
        user_idx = None
        for i, msg in enumerate(result.messages):
            if msg.get("content") == "Guidance content here.":
                guidance_idx = i
            if msg.get("role") == "user" and "Test objective" in msg.get("content", ""):
                user_idx = i

        assert guidance_idx is not None, "Guidance message not found"
        assert user_idx is not None, "User message not found"
        assert guidance_idx < user_idx, "Guidance should appear before user msg"

    @pytest.mark.asyncio
    async def test_no_guidance_messages_by_default(self, search_tool: SearchTool):
        """Test that no extra system messages when guidance_messages is None."""
        strategy = MockStrategy(outputs=[StrategyOutput(finished=True, result="Done")])

        agent = AgentTool(
            tools=[search_tool],
            strategy=strategy,
            # No guidance_messages provided
        )

        result = await agent.ainvoke(AgentToolInput(objective="Test"))

        # Should have exactly 1 system message (the main system prompt)
        system_msgs = [msg for msg in result.messages if msg.get("role") == "system"]
        assert len(system_msgs) == 1
