"""Tests for NLTool and ToolWithFormatter - combined NL pipeline tools."""

from unittest.mock import MagicMock

from agents.tools_core.internal_tools.tests.common_fixtures import (
    SimpleTool,
    SimpleInput,
)
from agents.tools_core.internal_tools.nl_tool import NLTool, ToolWithFormatter


class TestNLTool:
    def test_name_and_description(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        nl_tool = NLTool(simple_tool, mock_llm_client)
        assert nl_tool.name == simple_tool.name
        assert nl_tool.description == simple_tool.description

    def test_with_args_in_description(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        nl_tool = NLTool(simple_tool, mock_llm_client, include_args_in_description=True)
        assert "query" in nl_tool.description


class TestToolWithFormatter:
    def test_name_and_input_type(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        wrapper = ToolWithFormatter(simple_tool, "Test task", mock_llm_client)
        assert wrapper.name == simple_tool.name
        assert wrapper._input == SimpleInput
