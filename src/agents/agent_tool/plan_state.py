"""
PlanState - mutable structured plan owned by AgentTool, mutated by the
PlanStateUpdate tool.

Design notes:
- Created per `AgentTool.ainvoke()` call; not held by AgentTool across runs.
- Mutated only via the PlanStateUpdate tool (model-driven) and the
  auto-status-update convention in `_execute_tool_calls`. Strategies receive
  it by reference but must not mutate it directly.
- Passed explicitly to `Strategy.plan(plan_state=...)` as a parameter.
- Sub-agents (SubAgentTool) get their own isolated PlanState because each
  ainvoke creates a new one.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TaskStatus = Literal[
    "pending",      # not started
    "in_progress",  # currently executing (exactly one task should be in_progress)
    "completed",    # finished successfully
    "failed",       # tried, did not succeed; result contains error description
    "blocked",      # cannot proceed (e.g., missing data, dependency failed)
    "cancelled",    # superseded by re-plan; kept for audit
]

PlanStatus = Literal[
    "draft",       # plan exists but no task has started yet
    "active",      # at least one task has been in_progress
    "completed",   # plan finished successfully (all required tasks completed)
    "failed",      # plan terminated with failure
    "blocked",     # cannot proceed; needs external input
]


class TaskState(BaseModel):
    """A single task within a PlanState."""

    id: int = Field(description="Unique stable identifier within this plan")
    objective: str = Field(description="What this task is for (sub-objective)")
    inputs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Materialized kwargs for the tool call(s) this task implies. "
            "MUST be a JSON OBJECT whose keys are the downstream tool's "
            "argument names. Example: for a task that calls "
            "send_email(to=..., subject=...), set "
            "inputs={\"to\": \"a@b.com\", \"subject\": \"Hi\"}. "
            "Do NOT pass a bare scalar like the value of one argument."
        ),
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def _coerce_inputs_to_dict(cls, v: Any) -> Any:
        """Tolerate models that emit a bare scalar where a kwargs dict is expected.

        A bare value gets wrapped as {"value": <scalar>} so the run survives
        and the model gets a chance to correct itself on the next turn.
        """
        if v is None or isinstance(v, dict):
            return v
        return {"value": v}

    @field_validator("result", "objective", mode="before")
    @classmethod
    def _coerce_text_field(cls, v: Any) -> Any:
        """Tolerate models that emit a structured value (dict/list) where a
        free-text string is expected.

        LLMs often mistakenly paste a previous tool's parsed output back into
        a text field (e.g., setting `result` to `{"foo": "bar"}` instead of
        the JSON-stringified form Phase D's auto-update writes). Rather than
        failing validation and bouncing the run, we serialize the value to
        a string so the run survives. The model still gets clearer plain
        text to read on the next turn.
        """
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)
    status: TaskStatus = "pending"
    result: str | None = Field(
        default=None,
        description=(
            "Final output as text, or error description if status == 'failed'."
        ),
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description=(
            "IDs of tasks that must reach 'completed' before this task can start."
        ),
    )
    parent_attempt_id: int | None = Field(
        default=None,
        description="If this task is a retry, the ID of the prior failed attempt.",
    )


class PlanState(BaseModel):
    """Mutable plan held by AgentTool for the duration of one ainvoke call."""

    objective: str
    tasks: list[TaskState] = Field(default_factory=list)
    status: PlanStatus = "draft"
    revision_count: int = 0

    def next_task_id(self) -> int:
        if not self.tasks:
            return 1
        return max(t.id for t in self.tasks) + 1

    def in_progress_task(self) -> TaskState | None:
        """Return one in_progress task, or None if none are in_progress.

        Convenience for the common case where a single task is in flight.
        When multiple tasks are in_progress (a parallel batch fan-out),
        returns the FIRST one in insertion order. Use `in_progress_tasks()`
        when you need the full set (e.g., Phase D auto-pairing).
        """
        tasks = self.in_progress_tasks()
        return tasks[0] if tasks else None

    def in_progress_tasks(self) -> list[TaskState]:
        """Return all tasks currently `in_progress`, preserving insertion order.

        Multiple in_progress tasks are valid when they form a parallel batch
        (no `depends_on` edges between them). Auto-status-update for parallel
        batches relies on this list to pair tool results back to their tasks.
        """
        return [t for t in self.tasks if t.status == "in_progress"]

    def tasks_in_display_order(self) -> list[TaskState]:
        """Topologically order tasks by `depends_on`, ascending `id` as tiebreaker.

        Display/iteration helper — does NOT mutate `self.tasks`. The raw
        `tasks` list stays in insertion order so IDs remain stable across
        re-plans. Use this when you want the dependency-respecting view
        (e.g., for printing the plan, exporting an audit trail, or any
        UI that should show tasks in execution order).

        Falls back to id-order for any tasks involved in a cycle or with
        dangling references — the run survives, just less prettily sorted.
        """
        by_id = {t.id: t for t in self.tasks}
        placed: set[int] = set()
        ordered: list[TaskState] = []
        remaining = sorted(self.tasks, key=lambda t: t.id)
        while remaining:
            ready = [
                t
                for t in remaining
                if all(
                    dep in placed or dep not in by_id for dep in t.depends_on
                )
            ]
            if not ready:
                # Cycle or unreachable subgraph — emit leftovers in id order.
                ordered.extend(remaining)
                break
            for t in ready:
                ordered.append(t)
                placed.add(t.id)
                remaining.remove(t)
        return ordered

    def serialize_for_prompt(self) -> str:
        """Render as a structured block for the strategy's prompt context."""
        if not self.tasks:
            return (
                f"Objective: {self.objective}\n"
                f"Plan status: {self.status}\n\n"
                "Plan: (empty - call planstate_update to draft one)"
            )
        lines = [
            f"Objective: {self.objective}",
            f"Plan status: {self.status}",
            "",
            "Plan:",
        ]
        for t in self.tasks_in_display_order():
            deps = f" [depends on: {t.depends_on}]" if t.depends_on else ""
            inputs = f"\n      inputs={t.inputs}" if t.inputs else ""
            result = f"\n      result={t.result[:200]}" if t.result else ""
            lines.append(
                f"  [{t.id}] {t.status.upper():<12} {t.objective}{deps}{inputs}{result}"
            )
        lines.append(f"\n(plan revision {self.revision_count})")
        return "\n".join(lines)
