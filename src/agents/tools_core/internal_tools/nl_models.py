"""
Shared models for natural language tool wrappers.

Models:
- NLInput: Natural language task input (task, context)
- NLOutput: Base NL result (success, result, error)
- LLMError: Error details (error, type_of_error, content, suggested_fix)
"""

from pydantic import BaseModel, Field


class NLInput(BaseModel):
    """Input for LLM-wrapped tools that take natural language tasks."""

    task: str = Field(
        description="The objective to accomplish by calling the provided tool."
    )
    context: str | None = Field(default=None, description="Context to solve the task.")


class LLMError(BaseModel):
    """Error response when LLM cannot complete the task."""

    error: str = Field(
        description="The error description, such as missing information needed to complete the task."
    )
    type_of_error: str = Field(
        description="Type of error, such as missing information or issues with tool call."
    )
    content: str = Field(description="Summary of available information.")
    suggested_fix: str = Field(description="Suggestion to fix.")


class NLOutput(BaseModel):
    """Base output for LLM-wrapped tools that return natural language results."""

    success: bool = Field(description="Whether the operation succeeded")
    result: str | None = Field(
        default=None,
        description="Result of running the tool, expressed in natural language (if success=True).",
    )
    error: LLMError | None = Field(
        default=None, description="Error details (if success=False)"
    )
