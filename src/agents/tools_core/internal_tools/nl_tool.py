"""
Combined NL pipeline tools that chain parsing, execution, and formatting.

NLTool: Full 3-step pipeline - NL task parse → execute → format
    - Input: NLInput (task, context)
    - Output: NLToolResult extends NLOutput (+ stage)
    - Base: BaseTool
    - Use case: End-to-end NL interface for any tool

ToolWithFormatter: 2-step pipeline - execute tool with native input, format output
    - Input: Tool's input type (e.g., SearchInput)
    - Output: ToolWithFormatterResult extends NLOutput (+ stage)
    - Base: BaseTool
    - Use case: Direct tool input, but want formatted NL output
"""

import typing as t

from pydantic import BaseModel, Field

from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.internal_tools.tool_output_formatter import FormatToolOutput
from agents.tools_core.internal_tools.nl_models import LLMError, NLInput, NLOutput
from agents.tools_core.internal_tools.tool_input_parser import ParseToolInput

# Type variables for tool input/output
ToolInputT = t.TypeVar("ToolInputT", bound=BaseModel)
ToolOutputT = t.TypeVar("ToolOutputT", bound=BaseModel)


class NLToolResult(NLOutput):
    """Result from NLTool - full 3-step pipeline result.

    Extends NLOutput with stage tracking.
    """

    stage: str = Field(
        default="complete",
        description=(
            "Stage where error occurred: 'input_parsing', 'tool_execution', "
            "'output_formatting', or 'complete'"
        ),
    )


class NLTool(BaseTool[NLInput, NLToolResult]):
    """
    Full NL pipeline: NL task → parse to tool input → execute tool → format output.

    Combines ParseToolInput and FormatToolOutput for end-to-end processing.
    """

    _input = NLInput
    _output = NLToolResult

    example_inputs = (NLInput(task="...", context="..."),)
    example_outputs = (
        NLToolResult(success=True, result="...", stage="complete"),
        NLToolResult(
            success=False,
            error=LLMError(
                error="...", type_of_error="...", content="...", suggested_fix="..."
            ),
            stage="input_parsing",
        ),
    )

    def __init__(
        self,
        tool: BaseTool[ToolInputT, ToolOutputT],
        llm_client: LLMClient,
        model: str | None = None,
        include_args_in_description: bool = False,
    ) -> None:
        super().__init__()

        self.tool = tool
        self.llm_client = llm_client
        self._model = model

        self._name = tool.name
        self.description = tool.description

        self.llm_input_wrapper = ParseToolInput(tool, llm_client, model)

        if include_args_in_description:
            field_names = list(tool._input.model_fields.keys())
            self.description += " | Takes as arguments: " + ", ".join(field_names)

    def invoke(self, input: NLInput) -> NLToolResult:
        """Execute the full pipeline synchronously."""
        validated = self._validate_input(input)

        # Step 1: Parse natural language to tool input
        input_result = self.llm_input_wrapper.invoke(validated)
        if not input_result.success or input_result.error:
            return NLToolResult(
                success=False, error=input_result.error, stage="input_parsing"
            )

        # Get validated tool input
        tool_input = self.llm_input_wrapper.get_tool_input(input_result)
        if tool_input is None:
            return NLToolResult(
                success=False,
                error=LLMError(
                    error="Failed to parse tool input",
                    type_of_error="Parsing error",
                    content="Could not validate parsed input against tool schema",
                    suggested_fix="Check task description and context",
                ),
                stage="input_parsing",
            )

        # Step 2: Execute the tool
        try:
            raw_output = self.tool.invoke(tool_input)
        except Exception as e:
            return NLToolResult(
                success=False,
                error=LLMError(
                    error=str(e),
                    type_of_error="Tool execution error",
                    content=f"Tool {self.tool.name} failed during execution",
                    suggested_fix="Check tool input and try again",
                ),
                stage="tool_execution",
            )

        # Step 3: Format the output
        output_wrapper = FormatToolOutput(
            self.tool, input.task, self.llm_client, self._model
        )
        output_result = output_wrapper.invoke(raw_output)

        if not output_result.success or output_result.error:
            return NLToolResult(
                success=False, error=output_result.error, stage="output_formatting"
            )

        return NLToolResult(success=True, result=output_result.result, stage="complete")

    async def ainvoke(self, input: NLInput) -> NLToolResult:
        """Execute the full pipeline asynchronously."""
        validated = self._validate_input(input)

        # Step 1: Parse natural language to tool input
        input_result = await self.llm_input_wrapper.ainvoke(validated)
        if not input_result.success or input_result.error:
            return NLToolResult(
                success=False, error=input_result.error, stage="input_parsing"
            )

        # Get validated tool input
        tool_input = self.llm_input_wrapper.get_tool_input(input_result)
        if tool_input is None:
            return NLToolResult(
                success=False,
                error=LLMError(
                    error="Failed to parse tool input",
                    type_of_error="Parsing error",
                    content="Could not validate parsed input against tool schema",
                    suggested_fix="Check task description and context",
                ),
                stage="input_parsing",
            )

        # Step 2: Execute the tool
        try:
            raw_output = await self.tool.ainvoke(tool_input)
        except Exception as e:
            return NLToolResult(
                success=False,
                error=LLMError(
                    error=str(e),
                    type_of_error="Tool execution error",
                    content=f"Tool {self.tool.name} failed during execution",
                    suggested_fix="Check tool input and try again",
                ),
                stage="tool_execution",
            )

        # Step 3: Format the output
        output_wrapper = FormatToolOutput(
            self.tool, input.task, self.llm_client, self._model
        )
        output_result = await output_wrapper.ainvoke(raw_output)

        if not output_result.success or output_result.error:
            return NLToolResult(
                success=False, error=output_result.error, stage="output_formatting"
            )

        return NLToolResult(success=True, result=output_result.result, stage="complete")


class ToolWithFormatterResult(NLOutput):
    """Result from ToolWithFormatter - 2-step pipeline result.

    Extends NLOutput with stage tracking.
    """

    stage: str = Field(
        default="complete",
        description=(
            "Stage where error occurred: 'tool_execution', 'output_formatting', "
            "or 'complete'"
        ),
    )


class ToolWithFormatter(BaseTool[ToolInputT, ToolWithFormatterResult]):
    """
    Wraps a tool to format its output using LLM (input passed directly to tool).

    Unlike NLTool, this takes the tool's native input directly, not natural language.
    """

    _input = BaseModel  # Placeholder, overwritten in __init__ with tool's input type
    _output = ToolWithFormatterResult

    example_outputs = (
        ToolWithFormatterResult(success=True, result="...", stage="complete"),
        ToolWithFormatterResult(
            success=False,
            error=LLMError(
                error="...", type_of_error="...", content="...", suggested_fix="..."
            ),
            stage="tool_execution",
        ),
    )

    def __init__(
        self,
        tool: BaseTool[ToolInputT, ToolOutputT],
        task: str,
        llm_client: LLMClient,
        model: str | None = None,
    ) -> None:
        super().__init__()

        self.tool = tool
        self.task = task
        self.llm_client = llm_client
        self._model = model

        self._name = tool.name
        self._input = tool._input
        self.description = tool.description
        self.example_inputs = tool.example_inputs

    def invoke(self, input: ToolInputT) -> ToolWithFormatterResult:
        """Execute tool and format output synchronously."""
        validated = self._validate_input(input)

        # Step 1: Execute the tool
        try:
            raw_output = self.tool.invoke(validated)
        except Exception as e:
            return ToolWithFormatterResult(
                success=False,
                error=LLMError(
                    error=str(e),
                    type_of_error="Tool execution error",
                    content=f"Tool {self.tool.name} failed",
                    suggested_fix="Check tool input",
                ),
                stage="tool_execution",
            )

        # Step 2: Format the output
        output_wrapper = FormatToolOutput(
            self.tool, self.task, self.llm_client, self._model
        )
        output_result = output_wrapper.invoke(raw_output)

        if not output_result.success or output_result.error:
            return ToolWithFormatterResult(
                success=False,
                error=output_result.error,
                stage="output_formatting",
            )

        return ToolWithFormatterResult(
            success=True,
            result=output_result.result,
            stage="complete",
        )

    async def ainvoke(self, input: ToolInputT) -> ToolWithFormatterResult:
        """Execute tool and format output asynchronously."""
        validated = self._validate_input(input)

        # Step 1: Execute the tool
        try:
            raw_output = await self.tool.ainvoke(validated)
        except Exception as e:
            return ToolWithFormatterResult(
                success=False,
                error=LLMError(
                    error=str(e),
                    type_of_error="Tool execution error",
                    content=f"Tool {self.tool.name} failed",
                    suggested_fix="Check tool input",
                ),
                stage="tool_execution",
            )

        # Step 2: Format the output
        output_wrapper = FormatToolOutput(
            self.tool, self.task, self.llm_client, self._model
        )
        output_result = await output_wrapper.ainvoke(raw_output)

        if not output_result.success or output_result.error:
            return ToolWithFormatterResult(
                success=False,
                error=output_result.error,
                stage="output_formatting",
            )

        return ToolWithFormatterResult(
            success=True,
            result=output_result.result,
            stage="complete",
        )
