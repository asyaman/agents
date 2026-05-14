"""Tests for the PlanStateUpdate tool."""

import pytest

from agents.agent_tool.meta_tool_plan_state_update import (
    PlanStateUpdate,
    PlanStateUpdateInput,
    PlanStateUpdateOutput,
)
from agents.agent_tool.plan_state import PlanState, TaskState


class TestPlanStateUpdateTool:
    def test_name_is_normalized_uppercase(self):
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        assert tool.name == "PLANSTATE_UPDATE"
        assert tool.raw_name == "planstate_update"

    def test_invoke_replaces_tasks_wholesale(self):
        plan = PlanState(
            objective="x",
            tasks=[TaskState(id=99, objective="stale")],
        )
        tool = PlanStateUpdate(plan)

        new_tasks = [
            TaskState(id=1, objective="t1"),
            TaskState(id=2, objective="t2"),
        ]
        out = tool.invoke(PlanStateUpdateInput(tasks=new_tasks))

        assert isinstance(out, PlanStateUpdateOutput)
        assert out.accepted is True
        assert out.revision == 1
        assert plan.tasks == new_tasks  # closured plan was mutated
        assert all(t.id != 99 for t in plan.tasks)

    def test_invoke_increments_revision_count(self):
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        for expected_rev in (1, 2, 3):
            out = tool.invoke(PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t")]))
            assert out.revision == expected_rev
            assert plan.revision_count == expected_rev

    def test_invoke_serialized_plan_reflects_new_state(self):
        plan = PlanState(objective="goal")
        tool = PlanStateUpdate(plan)
        out = tool.invoke(
            PlanStateUpdateInput(
                tasks=[TaskState(id=1, objective="step one", status="in_progress")]
            )
        )
        assert "step one" in out.serialized_plan
        assert "IN_PROGRESS" in out.serialized_plan

    def test_draft_auto_transitions_to_active_on_in_progress(self):
        plan = PlanState(objective="x")
        assert plan.status == "draft"
        tool = PlanStateUpdate(plan)
        out = tool.invoke(
            PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t", status="in_progress")])
        )
        assert plan.status == "active"
        assert out.plan_status == "active"

    def test_draft_stays_draft_when_no_task_in_progress(self):
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        out = tool.invoke(
            PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t", status="pending")])
        )
        assert plan.status == "draft"
        assert out.plan_status == "draft"

    def test_explicit_plan_status_overrides_auto_transition(self):
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        out = tool.invoke(
            PlanStateUpdateInput(
                tasks=[TaskState(id=1, objective="t", status="completed")],
                plan_status="completed",
            )
        )
        assert plan.status == "completed"
        assert out.plan_status == "completed"

    def test_setting_plan_status_failed(self):
        plan = PlanState(objective="x", status="active")
        tool = PlanStateUpdate(plan)
        out = tool.invoke(
            PlanStateUpdateInput(
                tasks=[TaskState(id=1, objective="t", status="failed")],
                plan_status="failed",
            )
        )
        assert plan.status == "failed"
        assert out.plan_status == "failed"

    def test_invoke_accepts_dict_input(self):
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        out = tool({"tasks": [{"id": 1, "objective": "t", "status": "pending"}]})
        assert out.accepted
        assert plan.tasks[0].objective == "t"

    @pytest.mark.asyncio
    async def test_acall_works_via_default_async_wrapper(self):
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        out = await tool.acall(PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t")]))
        assert out.accepted is True
        assert plan.revision_count == 1

    def test_mutation_visible_to_external_observer(self):
        """The closured plan_state is mutated by reference."""
        plan = PlanState(objective="x")
        tool = PlanStateUpdate(plan)
        external_ref = plan

        tool.invoke(
            PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t", status="completed")])
        )

        assert external_ref is plan
        assert len(external_ref.tasks) == 1
        assert external_ref.tasks[0].status == "completed"
        assert external_ref.revision_count == 1
