"""Tests for planning strategies base classes."""

from agents.agent_tool.base_strategy import StrategyOutput
from agents.llm_core.llm_client import ToolCall


class TestStrategyOutput:
    """Tests for StrategyOutput model."""

    def test_default_values(self):
        output = StrategyOutput()
        assert output.messages == []
        assert output.tool_calls == []
        assert output.finished is False
        assert output.success is True  # Default to success
        assert output.result is None

    def test_with_tool_calls(self):
        output = StrategyOutput(
            tool_calls=[ToolCall(tool_name="search", arguments={"q": "test"}, id="1")]
        )
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].tool_name == "search"

    def test_finished_with_result(self):
        output = StrategyOutput(finished=True, result="Task complete")
        assert output.finished
        assert output.success is True
        assert output.result == "Task complete"

    def test_finished_with_failure(self):
        output = StrategyOutput(finished=True, success=False, result="Task failed")
        assert output.finished
        assert output.success is False
        assert output.result == "Task failed"
