"""
LLM wrapper that parses natural language task into tool input schema.

ParseToolInput: Parses natural language task into tool input schema
    - Input: NLInput (task, context)
    - Output: ParseResult (success, tool_input, error)
    - Base: LLMTool
    - Use case: "Search for Python tutorials" → SearchInput(query="...")
"""

import typing as t

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, create_model

from agents.configs import get_tools_core_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.llm_base_tool import LLMTool
from agents.tools_core.internal_tools.nl_models import LLMError, NLInput

# Type variables for tool input/output
ToolInputT = t.TypeVar("ToolInputT", bound=BaseModel)
ToolOutputT = t.TypeVar("ToolOutputT", bound=BaseModel)

# Load templates
_templates = get_tools_core_template_module("tool_input_parser.jinja")


class ParseResult(BaseModel):
    """Result from ParseToolInput - either parsed tool input or error."""

    success: bool = Field(description="Whether parsing succeeded")
    tool_input: dict[str, t.Any] | None = Field(
        default=None, description="Parsed tool input as dict (if success=True)"
    )
    error: LLMError | None = Field(
        default=None, description="Error details (if success=False)"
    )


class ParseToolInput(LLMTool[NLInput, ParseResult]):
    """
    Wraps a tool with LLM that transforms natural language input into tool arguments.

    Takes a task description and context, uses LLM to parse into the tool's input schema.
    """

    _input = NLInput
    _output = ParseResult

    def __init__(
        self,
        tool: BaseTool[ToolInputT, ToolOutputT],
        llm_client: LLMClient,
        model: str | None = None,
        example_inputs: t.Sequence[NLInput] | None = None,
    ) -> None:
        super().__init__(llm_client, model=model)

        self.tool = tool
        self._tool_input_type = tool._input

        # Create unique input schema class for this wrapper
        self._input_schema_cls: type[NLInput] = create_model(
            tool.name + "WrapperInput", __base__=NLInput
        )

        self._name = f"llm_{tool.name}"
        self.description = (
            f"LLM calls the {tool.name} after parsing natural language task: "
            f"{tool.description}"
        )

        self.example_inputs = example_inputs or (
            NLInput(task=f"Natural language task for {tool.name}.", context=None),
        )

        # Build example outputs for the prompt
        self._example_outputs = [
            *[ex.model_dump_json() for ex in tool.example_inputs],
            LLMError(
                error="...", type_of_error="...", content="...", suggested_fix="..."
            ).model_dump_json(),
        ]

    def format_messages(self, input: NLInput) -> list[ChatCompletionMessageParam]:
        prompt = _templates.tool_input_wrapper(
            tool_name=self.tool.name,
            tool_description=self.tool.description,
            input_schema=self.tool.input_schema(),
            task=input.task,
            context=input.context or "None",
            examples=self._example_outputs,
        )
        return [{"role": "user", "content": prompt}]

    def get_tool_input(self, result: ParseResult) -> ToolInputT | None:
        """Parse the result into the actual tool input type."""
        if result.success and result.tool_input:
            return self._tool_input_type.model_validate(result.tool_input)
        return None
