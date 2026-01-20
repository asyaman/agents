"""Summarizer tool - summarizes text with given objectives."""

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.llm_core.llm_client import LLMClient
from agents.tools_core.llm_base_tool import LLMTool


class SummarizerInput(BaseModel):
    input: str = Field(description="Input to summarize")
    description: str = Field(description="Description of the input")
    objective: str = Field(description="Objective of the summarization")


class SummarizerOutput(BaseModel):
    summary: str = Field(description="The summary of the input")


class Summarizer(LLMTool[SummarizerInput, SummarizerOutput]):
    """Summarizes text based on provided objectives and description."""

    _name = "summarizer"
    description = "Summarize given input with specified objectives."
    _input = SummarizerInput
    _output = SummarizerOutput

    example_inputs = (
        SummarizerInput(
            input="input string",
            description="Description of characteristics of the input string",
            objective="focus of the summarization",
        ),
    )
    example_outputs = (SummarizerOutput(summary="output string"),)

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
    ) -> None:
        super().__init__(llm_client, model=model)

    def format_messages(
        self, input: SummarizerInput
    ) -> list[ChatCompletionMessageParam]:
        prompt = (
            f"Given an input text with provided description, summarize the text with provided objectives.\n"
            f"Objectives: {input.objective}\n"
            f"Input description: {input.description}\n"
            f"Input text: {input.input}"
        )
        return [{"role": "user", "content": prompt}]
