"""Tests for PlanState, TaskState, and the in_progress invariant."""


from agents.agent_tool.plan_state import PlanState, TaskState


class TestTaskState:
    """Tests for TaskState model."""

    def test_minimal_task(self):
        task = TaskState(id=1, objective="Do thing")
        assert task.id == 1
        assert task.objective == "Do thing"
        assert task.status == "pending"
        assert task.inputs is None
        assert task.result is None
        assert task.depends_on == []
        assert task.parent_attempt_id is None

    def test_task_with_all_fields(self):
        task = TaskState(
            id=2,
            objective="Process item",
            inputs={"item_id": "abc"},
            status="completed",
            result="processed",
            depends_on=[1],
            parent_attempt_id=None,
        )
        assert task.status == "completed"
        assert task.inputs == {"item_id": "abc"}
        assert task.result == "processed"
        assert task.depends_on == [1]

    def test_pydantic_round_trip(self):
        task = TaskState(
            id=3,
            objective="Roundtrip",
            status="in_progress",
            depends_on=[1, 2],
        )
        restored = TaskState(**task.model_dump())
        assert restored == task

    def test_task_status_literal_values(self):
        for status in [
            "pending",
            "in_progress",
            "completed",
            "failed",
            "blocked",
            "cancelled",
        ]:
            t = TaskState(id=1, objective="x", status=status)
            assert t.status == status

    def test_result_coerces_dict_to_json_string(self):
        """LLMs sometimes set `result` to a parsed tool output dict.
        The validator JSON-stringifies it instead of failing validation."""
        t = TaskState(id=1, objective="x", result={"foo": "bar"})
        assert t.result == '{"foo": "bar"}'

    def test_result_coerces_list_to_json_string(self):
        t = TaskState(id=1, objective="x", result=[1, 2, 3])
        assert t.result == "[1, 2, 3]"

    def test_result_passes_through_string(self):
        t = TaskState(id=1, objective="x", result="already a string")
        assert t.result == "already a string"

    def test_result_passes_through_none(self):
        t = TaskState(id=1, objective="x", result=None)
        assert t.result is None

    def test_objective_coerces_dict_to_json_string(self):
        """Same coercion for `objective` — also a free-text string field."""
        t = TaskState(id=1, objective={"summary": "do the thing"})
        assert t.objective == '{"summary": "do the thing"}'


class TestPlanStateNextTaskId:
    def test_empty_returns_one(self):
        plan = PlanState(objective="goal")
        assert plan.next_task_id() == 1

    def test_returns_max_plus_one(self):
        plan = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="a"),
                TaskState(id=5, objective="b"),
                TaskState(id=3, objective="c"),
            ],
        )
        assert plan.next_task_id() == 6


class TestPlanStateInProgressTasks:
    def test_no_in_progress_returns_none(self):
        plan = PlanState(
            objective="g",
            tasks=[TaskState(id=1, objective="a", status="pending")],
        )
        assert plan.in_progress_task() is None
        assert plan.in_progress_tasks() == []

    def test_one_in_progress_returns_it(self):
        t = TaskState(id=2, objective="b", status="in_progress")
        plan = PlanState(
            objective="g",
            tasks=[TaskState(id=1, objective="a", status="completed"), t],
        )
        assert plan.in_progress_task() is t
        assert plan.in_progress_tasks() == [t]

    def test_multiple_in_progress_returns_all_in_insertion_order(self):
        """Parallel batch fan-out: multiple tasks may be in_progress together
        (no depends_on between them). The framework now allows this — the
        single-task helper returns the first; the list helper returns all."""
        t1 = TaskState(id=1, objective="a", status="in_progress")
        t2 = TaskState(id=2, objective="b", status="in_progress")
        plan = PlanState(objective="g", tasks=[t1, t2])
        assert plan.in_progress_task() is t1  # first in insertion order
        assert plan.in_progress_tasks() == [t1, t2]


class TestPlanStateSerializeForPrompt:
    def test_empty_renders_hint(self):
        plan = PlanState(objective="My goal")
        s = plan.serialize_for_prompt()
        assert "My goal" in s
        assert "(empty" in s
        assert "planstate_update" in s
        assert "draft" in s  # Plan status: draft

    def test_renders_each_task(self):
        plan = PlanState(
            objective="Process users",
            tasks=[
                TaskState(id=1, objective="Fetch users", status="completed",
                          result="found 3 users"),
                TaskState(id=2, objective="Process user 1", status="in_progress",
                          inputs={"user_id": 1}, depends_on=[1]),
                TaskState(id=3, objective="Process user 2", status="pending",
                          depends_on=[1]),
            ],
            status="active",
            revision_count=2,
        )
        s = plan.serialize_for_prompt()

        assert "Process users" in s
        assert "active" in s
        assert "[1]" in s
        assert "[2]" in s
        assert "[3]" in s
        assert "COMPLETED" in s
        assert "IN_PROGRESS" in s
        assert "PENDING" in s
        assert "depends on: [1]" in s
        assert "user_id" in s
        assert "found 3 users" in s
        assert "revision 2" in s

    def test_truncates_long_results(self):
        long_result = "x" * 500
        plan = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="t", status="completed", result=long_result),
            ],
        )
        s = plan.serialize_for_prompt()
        assert "x" * 200 in s
        assert "x" * 201 not in s

    def test_renders_in_topological_order_after_replan(self):
        """After re-planning (retries), task IDs are insertion-ordered, not
        execution-ordered. The display should sort by the dependency graph
        (id-ascending tiebreak) without renumbering."""
        # Mimics the retry pattern: original tasks 1,2,3 then retry pair 4,5
        # inserted "between" 2 and 3 (3 has been rewired to depend on 5).
        plan = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="produce", status="completed"),
                TaskState(
                    id=2, objective="verify", status="completed", depends_on=[1]
                ),
                TaskState(
                    id=3,
                    objective="use",
                    status="pending",
                    depends_on=[5],  # rewired from [2] to [5]
                ),
                TaskState(
                    id=4,
                    objective="produce v2",
                    status="completed",
                    parent_attempt_id=1,
                ),
                TaskState(
                    id=5,
                    objective="verify v2",
                    status="in_progress",
                    parent_attempt_id=2,
                    depends_on=[4],
                ),
            ],
        )
        ordered_ids = [t.id for t in plan.tasks_in_display_order()]
        # Layer 0 (no deps): 1, 4 → sorted → [1, 4]
        # Layer 1 (deps in layer 0): 2 (deps=[1]), 5 (deps=[4]) → [2, 5]
        # Layer 2: 3 (deps=[5])
        assert ordered_ids == [1, 4, 2, 5, 3]

        # The serialized output should reflect the same order.
        s = plan.serialize_for_prompt()
        positions = {tid: s.index(f"[{tid}]") for tid in ordered_ids}
        assert (
            positions[1]
            < positions[4]
            < positions[2]
            < positions[5]
            < positions[3]
        )

    def test_display_order_falls_back_to_id_on_cycle(self):
        """A circular depends_on shouldn't crash serialization."""
        plan = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="a", depends_on=[2]),
                TaskState(id=2, objective="b", depends_on=[1]),
                TaskState(id=3, objective="c"),  # not in the cycle
            ],
        )
        ordered_ids = [t.id for t in plan.tasks_in_display_order()]
        # Task 3 has no deps so it's placed first; the cycle (1, 2) lands
        # afterwards in id order via the fallback.
        assert ordered_ids == [3, 1, 2]

    def test_display_order_ignores_dangling_depends_on(self):
        """A depends_on referencing a non-existent task ID shouldn't block placement."""
        plan = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="a", depends_on=[99]),  # 99 doesn't exist
                TaskState(id=2, objective="b", depends_on=[1]),
            ],
        )
        ordered_ids = [t.id for t in plan.tasks_in_display_order()]
        assert ordered_ids == [1, 2]


class TestPlanStateRoundTrip:
    def test_full_round_trip(self):
        plan = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="t1", status="completed", result="r1"),
                TaskState(id=2, objective="t2", status="pending", depends_on=[1]),
            ],
            status="active",
            revision_count=4,
        )
        restored = PlanState(**plan.model_dump())
        assert restored == plan


class TestPlanStateDefaults:
    def test_default_status_is_draft(self):
        plan = PlanState(objective="x")
        assert plan.status == "draft"
        assert plan.revision_count == 0
        assert plan.tasks == []
