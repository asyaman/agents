"""
LLM-based tools that use an LLMClient to process input and produce structured output.
"""

import typing as t
from abc import abstractmethod

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool


class LLMTool[InputT: BaseModel, OutputT: BaseModel](BaseTool[InputT, OutputT]):
    """
    Tool that uses an LLM to process input and produce structured output.

    Subclasses must implement `format_messages()` to convert tool input to LLM messages.
    The tool automatically uses the `_output` model for structured response parsing.

    Example:
        class SummarizeTool(LLMTool[SummarizeInput, SummarizeOutput]):
            _name = "summarize"
            description = "Summarize text"
            _input = SummarizeInput
            _output = SummarizeOutput

            def format_messages(self, input: SummarizeInput) -> Messages:
                return [
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": f"Summarize: {input.text}"},
                ]

        tool = SummarizeTool(llm_client=my_client)
        result = tool(SummarizeInput(text="Long text..."))
    """

    def __init__(self, llm_client: LLMClient, model: str | None = None) -> None:
        super().__init__()
        self.llm_client = llm_client
        self._model = model  # None = use client's default_model

    @abstractmethod
    def format_messages(self, input: InputT) -> list[ChatCompletionMessageParam]:
        """
        Convert tool input to LLM messages.

        Args:
            input: Validated input model

        Returns:
            List of ChatCompletionMessageParam dicts
        """
        raise NotImplementedError()

    def invoke(self, input: InputT) -> OutputT:
        """Execute the tool synchronously."""
        validated = self._validate_input(input)
        messages = self.format_messages(validated)
        response = self.llm_client.generate(
            messages=messages,
            model=self._model,
            mode="pydantic",
            response_model=self._output,
        )
        return t.cast(OutputT, response.parsed)

    async def ainvoke(self, input: InputT) -> OutputT:
        """Execute the tool asynchronously."""
        validated = self._validate_input(input)
        messages = self.format_messages(validated)
        response = await self.llm_client.agenerate(
            messages=messages,
            model=self._model,
            mode="pydantic",
            response_model=self._output,
        )
        return t.cast(OutputT, response.parsed)
