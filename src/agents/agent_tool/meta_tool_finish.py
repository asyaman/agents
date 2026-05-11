"""
Finish - a built-in meta tool that signals task completion.

The model calls `finish` to terminate the agent loop with a final
result and a success flag. Like `planstate_update`, this is a META tool:
it does not perform real work, it carries a control signal.

Per the meta-tool emission policy (enforced in
`AgentTool._execute_tool_calls`), `finish` must be emitted ALONE in its
own iteration — never mixed with `planstate_update`, another meta tool,
or any action tool. Mixed turns are rejected and the model is asked to
retry with each meta tool in its own iteration.
"""

from pydantic import BaseModel, Field

from agents.tools_core.base_tool import BaseTool, create_fn_tool


class FinishInput(BaseModel):
    """Input for the finish tool."""

    result: str = Field(description="The final result/answer for the objective.")
    success: bool = Field(
        default=True, description="Whether the task was successful."
    )


class FinishOutput(BaseModel):
    """Output from the finish tool."""

    acknowledged: bool = Field(default=True)


def create_finish_tool() -> BaseTool[FinishInput, FinishOutput]:
    """Create the finish tool that signals task completion.

    The tool itself just acknowledges. The agent loop reads `finish` from
    the model's tool_calls and uses its arguments (result, success) as the
    final AgentTool output.
    """

    @create_fn_tool(
        name="finish",
        description="Call this when the task is complete to return the final result.",
    )
    def finish(result: str, success: bool = True) -> FinishOutput:
        return FinishOutput(acknowledged=True)

    return finish  # type: ignore
