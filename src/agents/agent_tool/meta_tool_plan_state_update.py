"""
PlanStateUpdate - a built-in tool that mutates the AgentTool's PlanState
by reference.

The tool is constructed at the start of each AgentTool.ainvoke() with a
closure over the run's PlanState instance. The model calls it to create or
revise the plan; the tool replaces the entire task list (the model provides
the full new list).

This is the planstate_update equivalent of Claude Code's TodoWrite, adapted
to this codebase's planner/executor architecture.
"""

import typing as t

from pydantic import BaseModel, Field

from agents.agent_tool.plan_state import PlanState, PlanStatus, TaskState
from agents.tools_core.base_tool import BaseTool


class PlanStateUpdateInput(BaseModel):
    tasks: list[TaskState] = Field(
        description=(
            "The FULL new task list. Provide all tasks (pending, in_progress, "
            "completed, etc.), not just changes. The previous list is replaced "
            "entirely. Mark exactly one task in_progress at a time."
        )
    )
    plan_status: PlanStatus | None = Field(
        default=None,
        description=(
            "Optional update to the overall plan status. Set to 'completed' "
            "when all required tasks are done, 'failed' when the objective "
            "cannot be met, or 'blocked' when external input is required. "
            "Setting to 'completed' or 'failed' will terminate the loop."
        ),
    )


class PlanStateUpdateOutput(BaseModel):
    accepted: bool
    revision: int
    plan_status: PlanStatus
    serialized_plan: str


class PlanStateUpdate(BaseTool[PlanStateUpdateInput, PlanStateUpdateOutput]):
    """Built-in tool that mutates the closured PlanState by reference."""

    _name = "planstate_update"
    description = (
        "Write or update the structured plan. Provide the FULL new task list - "
        "not a delta. Use this when:\n"
        "  (a) creating the initial plan after the first information-gathering step,\n"
        "  (b) splitting a task into sub-tasks (e.g., 'for each user, do X' -> "
        "N concrete tasks),\n"
        "  (c) marking a task in_progress before starting work on it,\n"
        "  (d) marking a task completed/failed/cancelled when its status changes "
        "(in most cases the framework auto-updates the in_progress task's status "
        "from the next tool result, so explicit calls are only needed for splits, "
        "retries, or overrides),\n"
        "  (e) re-planning after observations contradict the current plan,\n"
        "  (f) setting plan_status='completed' or 'failed' to terminate the run.\n"
        "Independent tasks (no depends_on link between them) MAY be in_progress "
        "together for a parallel batch. Tasks with depends_on belong to later "
        "batches.\n"
        "Do NOT call planstate_update in parallel with action tools in the same "
        "turn - update the plan first, then call action tools on the next turn "
        "(or call action tools first, then update on the next turn after "
        "observing results).\n\n"
        "IMPORTANT - each TaskState's `inputs` is a JSON OBJECT, not a bare "
        "scalar. Set it to the kwargs the downstream tool expects, with the "
        "tool's parameter names as keys.\n"
        '  CORRECT:  inputs={"search_query": "domain_name.ch company"}\n'
        '  WRONG:    inputs="search_query: domain_name.ch company"   (bare string)'
    )
    _input = PlanStateUpdateInput
    _output = PlanStateUpdateOutput

    example_inputs: t.ClassVar[t.Sequence[PlanStateUpdateInput]] = (
        PlanStateUpdateInput(
            tasks=[
                TaskState(
                    id=1,
                    objective="Research target company",
                    inputs={"search_query": "domain_name.ch company"},
                    status="in_progress",
                ),
                TaskState(
                    id=2,
                    objective="Draft outreach email",
                    inputs={
                        "email_to": "james@domain_name.ch",
                        "company_description": "<filled after task 1>",
                    },
                    status="pending",
                    depends_on=[1],
                ),
            ]
        ),
    )

    example_outputs: t.ClassVar[t.Sequence[PlanStateUpdateOutput]] = (
        PlanStateUpdateOutput(
            accepted=True,
            revision=1,
            plan_status="active",
            serialized_plan=(
                "Objective: Outreach to James\n"
                "Plan status: active\n\nPlan:\n"
                "  [1] IN_PROGRESS  Research target company\n"
                "      inputs={'search_query': 'domain_name.ch company'}\n"
                "  [2] PENDING      Draft outreach email [depends on: [1]]\n"
                "\n(plan revision 1)"
            ),
        ),
    )

    def __init__(self, plan_state: PlanState):
        super().__init__()
        self._plan_state = plan_state

    def invoke(self, input: PlanStateUpdateInput) -> PlanStateUpdateOutput:
        validated = self._validate_input(input)
        self._plan_state.tasks = list(validated.tasks)
        self._plan_state.revision_count += 1
        if validated.plan_status is not None:
            self._plan_state.status = validated.plan_status
        # Auto-transition draft -> active when any task becomes in_progress
        if self._plan_state.status == "draft" and any(
            t.status == "in_progress" for t in validated.tasks
        ):
            self._plan_state.status = "active"
        return PlanStateUpdateOutput(
            accepted=True,
            revision=self._plan_state.revision_count,
            plan_status=self._plan_state.status,
            serialized_plan=self._plan_state.serialize_for_prompt(),
        )
