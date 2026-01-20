"""Tests for the LLM-based tool implementation."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from agents.tools_core.llm_base_tool import LLMTool


class SummarizeInput(BaseModel):
    text: str


class SummarizeOutput(BaseModel):
    summary: str


class SummarizeTool(LLMTool[SummarizeInput, SummarizeOutput]):
    _name = "summarize"
    description = "Summarizes text"
    _input = SummarizeInput
    _output = SummarizeOutput

    def format_messages(
        self, input: SummarizeInput
    ) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": "You are a summarizer."},
            {"role": "user", "content": f"Summarize: {input.text}"},
        ]


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock()
    response = Mock()
    response.parsed = SummarizeOutput(summary="This is a summary.")
    client.generate.return_value = response
    client.agenerate = AsyncMock(return_value=response)
    return client


@pytest.fixture
def summarize_tool(mock_llm_client: MagicMock) -> SummarizeTool:
    return SummarizeTool(llm_client=mock_llm_client)


class TestLLMTool:
    def test_init_stores_client_and_model(self, mock_llm_client: MagicMock):
        tool = SummarizeTool(llm_client=mock_llm_client, model="gpt-4")
        assert tool.llm_client is mock_llm_client
        assert tool._model == "gpt-4"

    def test_init_model_defaults_to_none(self, mock_llm_client: MagicMock):
        tool = SummarizeTool(llm_client=mock_llm_client)
        assert tool._model is None

    def test_format_messages(self, summarize_tool: SummarizeTool):
        input_data = SummarizeInput(text="Hello world")
        messages = summarize_tool.format_messages(input_data)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Hello world" in messages[1]["content"]

    def test_invoke(self, summarize_tool: SummarizeTool, mock_llm_client: MagicMock):
        result = summarize_tool.invoke(SummarizeInput(text="Test text"))

        assert result == SummarizeOutput(summary="This is a summary.")
        mock_llm_client.generate.assert_called_once()
        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert call_kwargs["mode"] == "pydantic"
        assert call_kwargs["response_model"] is SummarizeOutput

    def test_invoke_with_dict(
        self, summarize_tool: SummarizeTool, mock_llm_client: MagicMock
    ):
        result = summarize_tool({"text": "Test text"})

        assert result == SummarizeOutput(summary="This is a summary.")
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke(
        self, summarize_tool: SummarizeTool, mock_llm_client: MagicMock
    ):
        result = await summarize_tool.ainvoke(SummarizeInput(text="Test text"))

        assert result == SummarizeOutput(summary="This is a summary.")
        mock_llm_client.agenerate.assert_called_once()
        call_kwargs = mock_llm_client.agenerate.call_args.kwargs
        assert call_kwargs["mode"] == "pydantic"
        assert call_kwargs["response_model"] is SummarizeOutput

    def test_invoke_uses_custom_model(self, mock_llm_client: MagicMock):
        tool = SummarizeTool(llm_client=mock_llm_client, model="gpt-4-turbo")
        tool.invoke(SummarizeInput(text="Test"))

        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4-turbo"

    def test_name_and_description(self, summarize_tool: SummarizeTool):
        assert summarize_tool.name == "SUMMARIZE"
        assert summarize_tool.description == "Summarizes text"
