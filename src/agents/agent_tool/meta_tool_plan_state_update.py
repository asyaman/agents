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
        "Write or update the structured plan. Provide the FULL new task list "
        "— not a delta. The framework does NOT mutate plan_state from tool "
        "results; every transition is driven by this tool. See the planner "
        "prompt for the full decision tree (Reconcile-and-plan / "
        "Reconcile-and-finish / Plan / Finish). Use this tool to:\n"
        "  (a) draft the initial plan on the first iteration (if the "
        "objective needs 2+ tool calls),\n"
        "  (b) reconcile after an action batch — record what actually "
        "happened: set each just-run task's status (completed / failed / "
        "cancelled), `inputs` (the exact call args read from the tool "
        "message), and `result` (the tool's output or error from the "
        "tool message),\n"
        "  (c) plan forward — mark next task(s) `in_progress`, fan out "
        "umbrella work into per-item sub-tasks, add retry tasks "
        "(`parent_attempt_id`), cancel obsolete branches,\n"
        "  (d) terminate via `plan_status='completed'` / `'failed'`.\n"
        "Reconcile-and-plan (b + c) is ONE planstate_update call — never "
        "split reconciliation and forward planning across two iterations.\n\n"
        "Independent tasks (no depends_on link between them) MAY be "
        "in_progress together for a parallel batch. Tasks with depends_on "
        "belong to later batches.\n\n"
        "Emission rules: planstate_update MUST appear alone in an iteration, "
        "EXCEPT when paired with `finish` (Reconcile-and-finish mode — both "
        "in the same iteration; the framework runs planstate_update first). "
        "Never mix planstate_update with action tools.\n\n"
        "`inputs` and `result` are RETROSPECTIVE audit fields: leave both "
        "None for `pending` / `in_progress` tasks; populate them during "
        "reconciliation with the exact data observed from the tool message. "
        "If set, `inputs` MUST be a JSON OBJECT (never a bare scalar).\n"
        '  CORRECT:  inputs={"search_query": "example.com"}\n'
        '  WRONG:    inputs="search_query: example.com"   (bare string)'
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
