"""
Tests for agents/tools/llm_tools/summarizer.py

Tests:
- test_summarizer_name_and_description: Summarizer naming and description
- test_summarizer_format_messages: Message formatting for summarization
- test_summarizer_input_output_schemas: Input/output schema structure
"""

from unittest.mock import MagicMock

from agents.tools.llm_tools.summarizer import (
    Summarizer,
    SummarizerInput,
    SummarizerOutput,
)
from agents.tools.llm_tools.tests.common_fixtures import mock_llm_client

mock_llm_client = mock_llm_client


class TestSummarizer:
    def test_name_and_description(self, mock_llm_client: MagicMock):
        summarizer = Summarizer(mock_llm_client)
        assert summarizer.name == "SUMMARIZER"
        assert "summarize" in summarizer.description.lower()

    def test_format_messages(self, mock_llm_client: MagicMock):
        summarizer = Summarizer(mock_llm_client)
        input_data = SummarizerInput(
            input="Text to summarize",
            description="A test document",
            objective="Extract key points",
        )
        messages = summarizer.format_messages(input_data)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = str(messages[0]["content"])
        assert "Text to summarize" in content
        assert "A test document" in content
        assert "Extract key points" in content

    def test_input_output_schemas(self, mock_llm_client: MagicMock):
        summarizer = Summarizer(mock_llm_client)

        input_schema = summarizer.input_schema()
        assert "input" in input_schema["properties"]
        assert "description" in input_schema["properties"]
        assert "objective" in input_schema["properties"]

        output_schema = summarizer.output_schema()
        assert "summary" in output_schema["properties"]

    def test_example_inputs_outputs(self, mock_llm_client: MagicMock):
        summarizer = Summarizer(mock_llm_client)
        assert len(summarizer.example_inputs) > 0
        assert len(summarizer.example_outputs) > 0
        assert isinstance(summarizer.example_inputs[0], SummarizerInput)
        assert isinstance(summarizer.example_outputs[0], SummarizerOutput)
