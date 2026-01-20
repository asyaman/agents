"""Tests for FormatToolOutput and SimplifyToolOutput - format tool output as NL."""

from unittest.mock import MagicMock

import pytest

from agents.tools_core.internal_tools.tests.common_fixtures import (
    SimpleTool,
    SimpleOutput,
)
from agents.tools_core.internal_tools.nl_models import NLOutput
from agents.tools_core.internal_tools.tool_output_formatter import (
    FormatToolOutput,
    OutputTooLargeError,
    SimplifyToolOutput,
)


class TestFormatToolOutput:
    def test_name_and_description(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        wrapper = FormatToolOutput(simple_tool, "Test task", mock_llm_client)
        assert wrapper.name == "LLM_FORMAT_SIMPLE_TOOL"
        assert "format" in wrapper.description.lower()

    def test_format_messages(self, simple_tool: SimpleTool, mock_llm_client: MagicMock):
        wrapper = FormatToolOutput(simple_tool, "Test task", mock_llm_client)
        output_data = SimpleOutput(result="Test result")
        messages = wrapper.format_messages(output_data)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = str(messages[0]["content"])
        assert "Test task" in content
        assert "Test result" in content


class TestSimplifyToolOutput:
    def test_name_and_description(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        simplifier = SimplifyToolOutput(simple_tool, "Test task", mock_llm_client)
        assert "simplify" in simplifier.name.lower()
        assert "simplif" in simplifier.description.lower()

    def test_default_thresholds(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        simplifier = SimplifyToolOutput(simple_tool, "Test task", mock_llm_client)
        assert simplifier.token_lower_bound == 5000
        assert simplifier.token_upper_bound == 90000
        assert simplifier.chunk_size == 30000
        assert simplifier.chunk_overlap == 1000
        assert simplifier.parallel_chunks is True

    def test_custom_thresholds(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        simplifier = SimplifyToolOutput(
            simple_tool,
            "Test task",
            mock_llm_client,
            token_lower_bound=100,
            token_upper_bound=1000,
            chunk_size=500,
            chunk_overlap=50,
            parallel_chunks=False,
        )
        assert simplifier.token_lower_bound == 100
        assert simplifier.token_upper_bound == 1000
        assert simplifier.chunk_size == 500
        assert simplifier.chunk_overlap == 50
        assert simplifier.parallel_chunks is False

    def test_count_tokens(self, simple_tool: SimpleTool, mock_llm_client: MagicMock):
        simplifier = SimplifyToolOutput(simple_tool, "Test task", mock_llm_client)
        token_count = simplifier._count_tokens("Hello world")
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_simplify_small_output_passes_through(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        # Set very high lower bound so output passes through
        simplifier = SimplifyToolOutput(
            simple_tool,
            "Test task",
            mock_llm_client,
            token_lower_bound=100000,
        )
        output = SimpleOutput(result="Small result")
        result = simplifier.simplify(output)

        # Should return raw output (pass through)
        assert result == output
        mock_llm_client.generate.assert_not_called()

    def test_simplify_too_large_raises_error(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        # Set very low upper bound
        simplifier = SimplifyToolOutput(
            simple_tool,
            "Test task",
            mock_llm_client,
            token_upper_bound=1,  # Very low
        )
        output = SimpleOutput(result="This output is too large")

        with pytest.raises(OutputTooLargeError) as exc_info:
            simplifier.simplify(output)

        assert simple_tool.name in str(exc_info.value)

    def test_format_messages(self, simple_tool: SimpleTool, mock_llm_client: MagicMock):
        simplifier = SimplifyToolOutput(simple_tool, "Test task", mock_llm_client)
        output = SimpleOutput(result="Test result")
        messages = simplifier.format_messages(output)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = str(messages[0]["content"])
        assert "Test task" in content
        assert "Test result" in content

    def test_format_chunk_messages(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        simplifier = SimplifyToolOutput(simple_tool, "Test task", mock_llm_client)
        messages = simplifier._format_chunk_messages("chunk content here")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = str(messages[0]["content"])
        assert "Test task" in content
        assert "chunk content here" in content

    def test_simplify_medium_output_calls_llm(
        self, simple_tool: SimpleTool, mock_llm_client: MagicMock
    ):
        # Set thresholds so output needs simplification
        mock_llm_client.generate.return_value.parsed = NLOutput(
            success=True, result="Simplified result"
        )
        simplifier = SimplifyToolOutput(
            simple_tool,
            "Test task",
            mock_llm_client,
            token_lower_bound=1,  # Very low so it triggers simplification
            token_upper_bound=100000,
            chunk_size=100000,  # High so no chunking
        )
        output = SimpleOutput(result="Medium sized result")
        result = simplifier.simplify(output)

        assert isinstance(result, NLOutput)
        assert result.success is True
        mock_llm_client.generate.assert_called_once()
