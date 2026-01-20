"""
Tests for agents/tools/llm_tools/llmcall.py

Tests:
- test_llmcall_name_derived_from_model: Name includes model name
- test_llmcall_description_includes_model: Description mentions model
- test_llmcall_format_messages: Message formatting
- test_llmcall_input_output_schemas: Schema structure
"""

from unittest.mock import MagicMock

import pytest

from agents.tools.llm_tools.llmcall import LLMCall, LLMInput, LLMOutput


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock()
    client._default_model = "gpt-4"
    return client


class TestLLMCall:
    def test_name_derived_from_model(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client, model="gpt-4-turbo")
        assert "gpt_4_turbo" in tool.name.lower()
        assert "call" in tool.name.lower()

    def test_name_uses_default_model(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client)
        assert "gpt_4" in tool.name.lower()

    def test_description_includes_model(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client, model="gpt-4-turbo")
        assert "gpt-4-turbo" in tool.description

    def test_format_messages(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client)
        input_data = LLMInput(message="What is 2+2?")
        messages = tool.format_messages(input_data)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 2+2?"

    def test_input_output_schemas(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client)

        input_schema = tool.input_schema()
        assert "message" in input_schema["properties"]

        output_schema = tool.output_schema()
        assert "answer" in output_schema["properties"]

    def test_example_inputs_outputs(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client)
        assert len(tool.example_inputs) > 0
        assert len(tool.example_outputs) > 0
        assert isinstance(tool.example_inputs[0], LLMInput)
        assert isinstance(tool.example_outputs[0], LLMOutput)

    def test_sanitizes_model_name(self, mock_llm_client: MagicMock):
        tool = LLMCall(mock_llm_client, model="openai/gpt-4.5-preview")
        # Should not contain /, -, or . in the name
        assert "/" not in tool._name
        assert "-" not in tool._name
        assert "." not in tool._name
