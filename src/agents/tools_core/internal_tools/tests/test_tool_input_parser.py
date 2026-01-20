"""Tests for ParseToolInput - parses natural language to tool input schema."""

from unittest.mock import MagicMock

from agents.tools_core.internal_tools.tests.common_fixtures import (
    SimpleTool,
    SimpleInput,
)
from agents.tools_core.internal_tools.nl_models import NLInput
from agents.tools_core.internal_tools.tool_input_parser import (
    ParseResult,
    ParseToolInput,
)


class TestParseToolInput:
    def test_name_and_description(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        wrapper = ParseToolInput(simple_tool, mock_llm_client)
        assert wrapper.name == "LLM_SIMPLE_TOOL"
        assert "SIMPLE_TOOL" in wrapper.description

    def test_format_messages(self, simple_tool: SimpleTool, mock_llm_client: MagicMock):
        wrapper = ParseToolInput(simple_tool, mock_llm_client)
        input_data = NLInput(task="Find something", context="Test context")
        messages = wrapper.format_messages(input_data)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = str(messages[0]["content"])
        assert "Find something" in content
        assert "Test context" in content

    def test_get_tool_input_success(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        wrapper = ParseToolInput(simple_tool, mock_llm_client)
        result = ParseResult(success=True, tool_input={"query": "test"})
        tool_input = wrapper.get_tool_input(result)

        assert tool_input is not None
        assert isinstance(tool_input, SimpleInput)
        assert tool_input.query == "test"

    def test_get_tool_input_failure(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        wrapper = ParseToolInput(simple_tool, mock_llm_client)
        result = ParseResult(success=False, error=None)
        tool_input = wrapper.get_tool_input(result)

        assert tool_input is None
