"""Tests for planning strategies base classes."""

from agents.agent_tool.base_strategy import StrategyOutput
from agents.llm_core.llm_client import ToolCall


class TestStrategyOutput:
    """Tests for StrategyOutput model.

    StrategyOutput now signals termination implicitly: empty tool_calls
    means stop, non-empty means run them. The success/result fields are
    only consulted on the empty-tool_calls path.
    """

    def test_default_values(self):
        output = StrategyOutput()
        assert output.messages == []
        assert output.tool_calls == []
        assert output.success is True  # Default to success
        assert output.result is None

    def test_with_tool_calls(self):
        output = StrategyOutput(
            tool_calls=[ToolCall(tool_name="search", arguments={"q": "test"}, id="1")]
        )
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].tool_name == "search"

    def test_implicit_terminate_with_success(self):
        """Empty tool_calls + success=True = strategy-internal terminate ok."""
        output = StrategyOutput(tool_calls=[], success=True, result="Task complete")
        assert output.tool_calls == []
        assert output.success is True
        assert output.result == "Task complete"

    def test_implicit_terminate_with_failure(self):
        """Empty tool_calls + success=False = strategy-internal terminate bad."""
        output = StrategyOutput(tool_calls=[], success=False, result="Task failed")
        assert output.tool_calls == []
        assert output.success is False
        assert output.result == "Task failed"
