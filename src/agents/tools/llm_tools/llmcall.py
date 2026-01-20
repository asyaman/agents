"""Simple LLM call tool - passes a message to an LLM and returns the response."""

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.llm_core.llm_client import LLMClient
from agents.tools_core.llm_base_tool import LLMTool


class LLMInput(BaseModel):
    message: str = Field(description="Message to pass to the LLM.")


class LLMOutput(BaseModel):
    answer: str = Field(description="The answer from the LLM.")


class LLMCall(LLMTool[LLMInput, LLMOutput]):
    """Simple tool that passes a message to an LLM and returns the response."""

    _name = "llm_call"
    description = "Ask a question to the LLM."
    _input = LLMInput
    _output = LLMOutput
    example_inputs = (LLMInput(message="What is the capital of France?"),)
    example_outputs = (LLMOutput(answer="The capital of France is Paris."),)

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
    ) -> None:
        super().__init__(llm_client, model=model)

        # Derive tool name from model, sanitizing special characters
        model_name = model or llm_client._default_model or "llm"
        safe_name = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        self._name = f"{safe_name}_call"
        self.description = f"Ask a question to {model_name}."

    def format_messages(self, input: LLMInput) -> list[ChatCompletionMessageParam]:
        return [{"role": "user", "content": input.message}]
