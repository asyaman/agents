"""Tests for PlanState integration into AgentTool: termination paths,
auto-status-update, and tool injection.
"""

import pytest

from agents.agent_tool.agent_tool import (
    AgentTool,
    AgentToolInput,
    FinishInput,
    _summarize_plan_result,
)
from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.agent_tool.meta_tool_plan_state_update import PlanStateUpdateInput
from agents.agent_tool.plan_state import PlanState, TaskState
from agents.agent_tool.tests.common_fixtures import SearchInput, SearchTool
from agents.llm_core.llm_client import ToolCall


def _tc(tool_name, arguments, parsed, call_id):
    return ToolCall(
        tool_name=tool_name, arguments=arguments, parsed=parsed, id=call_id
    )


class _ScriptedStrategy(PlanningStrategy):
    """Returns a fixed sequence of StrategyOutput objects."""

    def __init__(self, outputs):
        self._outputs = outputs
        self.calls = 0
        self.received_plan_states: list = []

    async def plan(
        self, messages, tools, parallel_tool_calls=True, plan_state=None
    ):
        self.received_plan_states.append(plan_state)
        out = self._outputs[self.calls]
        self.calls += 1
        return out


class TestAgentToolPlanStateOutput:
    @pytest.mark.asyncio
    async def test_plan_state_returned_in_output(self, search_tool: SearchTool):
        strategy = _ScriptedStrategy(
            [StrategyOutput(tool_calls=[], result="Done")]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)

        result = await agent.ainvoke(AgentToolInput(objective="My objective"))

        assert result.plan_state is not None
        assert result.plan_state.objective == "My objective"
        assert result.plan_state.tasks == []
        assert result.plan_state.revision_count == 0
        assert result.plan_state.status == "draft"


class TestAgentToolPlanStateInjection:
    @pytest.mark.asyncio
    async def test_planstate_update_tool_passed_to_strategy(
        self, search_tool: SearchTool
    ):
        captured: list = []

        class CapturingStrategy(PlanningStrategy):
            async def plan(
                self, messages, tools, parallel_tool_calls=True, plan_state=None
            ):
                captured.append([t.name for t in tools])
                return StrategyOutput(tool_calls=[], result="ok")

        agent = AgentTool(tools=[search_tool], strategy=CapturingStrategy())
        await agent.ainvoke(AgentToolInput(objective="x"))

        assert captured, "strategy.plan was not called"
        assert "PLANSTATE_UPDATE" in captured[0]
        assert "FINISH" in captured[0]
        assert "SEARCH" in captured[0]

    @pytest.mark.asyncio
    async def test_planstate_update_can_be_disabled(
        self, search_tool: SearchTool
    ):
        captured: list = []

        class CapturingStrategy(PlanningStrategy):
            async def plan(
                self, messages, tools, parallel_tool_calls=True, plan_state=None
            ):
                captured.append([t.name for t in tools])
                return StrategyOutput(tool_calls=[], result="ok")

        agent = AgentTool(
            tools=[search_tool],
            strategy=CapturingStrategy(),
            include_planstate_update_tool=False,
        )
        await agent.ainvoke(AgentToolInput(objective="x"))

        assert "PLANSTATE_UPDATE" not in captured[0]

    @pytest.mark.asyncio
    async def test_strategy_receives_plan_state_by_reference(
        self, search_tool: SearchTool
    ):
        strategy = _ScriptedStrategy(
            [StrategyOutput(tool_calls=[], result="ok")]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="x"))

        assert strategy.received_plan_states[0] is result.plan_state


class TestAgentToolTerminationPath2:
    """Termination path 2: plan_status set to terminal value via planstate_update."""

    @pytest.mark.asyncio
    async def test_completed_plan_status_terminates(self):
        # Iter 1: planstate_update with plan_status='completed'
        # Loop should terminate before iter 2.
        tasks = [TaskState(id=1, objective="t", status="completed")]
        update_input = PlanStateUpdateInput(
            tasks=tasks, plan_status="completed"
        )
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                # Should never be reached
                StrategyOutput(tool_calls=[], result="should not reach"),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        assert result.success is True
        assert result.iterations_used == 1
        assert result.plan_state.status == "completed"
        assert strategy.calls == 1  # second iteration never ran

    @pytest.mark.asyncio
    async def test_failed_plan_status_terminates_with_failure(self):
        tasks = [TaskState(id=1, objective="t", status="failed")]
        update_input = PlanStateUpdateInput(
            tasks=tasks, plan_status="failed"
        )
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="should not reach"),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        assert result.success is False
        assert result.plan_state.status == "failed"
        assert strategy.calls == 1


class TestAgentToolAutoStatusUpdate:
    """Auto-status-update from tool execution per the convention."""

    @pytest.mark.asyncio
    async def test_single_action_tool_marks_in_progress_completed(
        self, search_tool: SearchTool
    ):
        # 1. planstate_update sets task as in_progress
        # 2. search runs — auto-update should mark task completed
        # 3. finish
        tasks = [
            TaskState(
                id=1,
                objective="Search",
                inputs={"query": "x"},
                status="in_progress",
            )
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        assert result.success
        assert result.plan_state.tasks[0].status == "completed"
        assert result.plan_state.tasks[0].result is not None
        assert "Result for: x" in result.plan_state.tasks[0].result

    @pytest.mark.asyncio
    async def test_planstate_update_alone_does_not_trigger_autoupdate(
        self, search_tool: SearchTool
    ):
        """A turn that only calls planstate_update should not auto-update statuses
        (planstate_update is a meta tool)."""
        tasks = [
            TaskState(
                id=1, objective="t", inputs={"q": "x"}, status="in_progress"
            )
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # The single in_progress task should remain in_progress (not auto-completed)
        assert result.plan_state.tasks[0].status == "in_progress"

    @pytest.mark.asyncio
    async def test_parallel_action_tools_skip_autoupdate(
        self, search_tool: SearchTool, calculator_tool
    ):
        """Cardinality fallback: 2 tools but only 1 in_progress task (no
        fan-out) → strict bijective pairing can't apply → auto-update is
        skipped, task 1 stays in_progress. The model must call
        planstate_update on the next turn to record outcomes (legacy
        behavior preserved when pairing is impossible)."""
        from agents.agent_tool.tests.common_fixtures import CalculatorInput

        tasks = [
            TaskState(id=1, objective="t1", status="in_progress"),
            TaskState(id=2, objective="t2", status="pending"),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                # Two action tools in parallel
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        ),
                        _tc(
                            "calculator",
                            {"expression": "1+1"},
                            CalculatorInput(expression="1+1"),
                            "c1",
                        ),
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(
            tools=[search_tool, calculator_tool], strategy=strategy
        )
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Auto-update was skipped — task 1 stays in_progress
        assert result.plan_state.tasks[0].status == "in_progress"


class TestAutoPairingParallelBatch:
    """Phase D extends the single-tool auto-update to parallel batches via
    strict bijective pairing on `inputs`. When the model fans out N sub-tasks
    (each `in_progress` with concrete `inputs`) and dispatches N tools whose
    args match, each tool's outcome lands in its corresponding task — no
    follow-up planstate_update needed.

    Pairing is strict: same N, every tool finds a unique matching task.
    Anything else falls back to the legacy "skip" behavior (model records
    via planstate_update). This avoids silent mispairing into wrong tasks.
    """

    @staticmethod
    def _make_search_tcs(*queries: str) -> list:
        return [
            _tc(
                "search",
                {"query": q},
                SearchInput(query=q),
                f"s_{i}",
            )
            for i, q in enumerate(queries)
        ]

    @pytest.mark.asyncio
    async def test_bijective_pairing_marks_all_tasks_completed(
        self, search_tool: SearchTool
    ):
        """3 fanned-out sub-tasks (each with unique `inputs`) + 3 parallel
        tool calls with matching args → each pair auto-updates."""
        tasks = [
            TaskState(
                id=1, objective="t1", inputs={"query": "a"}, status="in_progress"
            ),
            TaskState(
                id=2, objective="t2", inputs={"query": "b"}, status="in_progress"
            ),
            TaskState(
                id=3, objective="t3", inputs={"query": "c"}, status="in_progress"
            ),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                # Three parallel search calls, args match the three tasks.
                StrategyOutput(
                    tool_calls=self._make_search_tcs("a", "b", "c")
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # All three sub-tasks auto-completed via bijective match.
        statuses = {t.id: t.status for t in result.plan_state.tasks}
        assert statuses == {1: "completed", 2: "completed", 3: "completed"}
        # Each task got the right tool's output (verify by query echo).
        results = {t.id: t.result for t in result.plan_state.tasks}
        assert "Result for: a" in results[1]
        assert "Result for: b" in results[2]
        assert "Result for: c" in results[3]

    @pytest.mark.asyncio
    async def test_pairing_handles_mixed_success_and_failure(
        self, search_tool: SearchTool
    ):
        """If one of the parallel tools raises a parse error, that task
        is marked `failed` and the others `completed` — the pairing
        applies per-tool independently."""
        tasks = [
            TaskState(
                id=1, objective="t1", inputs={"query": "ok"}, status="in_progress"
            ),
            TaskState(
                id=2, objective="t2", inputs={"query": "bad"}, status="in_progress"
            ),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "ok"},
                            SearchInput(query="ok"),
                            "s_ok",
                        ),
                        # Second call has parse_error — this turns into a
                        # tool result that's_error=True; pairing still works.
                        ToolCall(
                            id="s_bad",
                            tool_name="search",
                            arguments={"query": "bad"},
                            parsed=None,
                            parse_error="bogus",
                        ),
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        by_id = {t.id: t for t in result.plan_state.tasks}
        assert by_id[1].status == "completed"
        assert by_id[2].status == "failed"
        assert "bogus" in by_id[2].result

    @pytest.mark.asyncio
    async def test_ambiguous_inputs_falls_back_to_no_update(
        self, search_tool: SearchTool
    ):
        """Two in_progress tasks with IDENTICAL `inputs` and two parallel
        tool calls with the same args → can't disambiguate which result
        belongs to which task → skip the whole batch, leave statuses
        untouched."""
        tasks = [
            TaskState(
                id=1,
                objective="dupe-a",
                inputs={"query": "same"},
                status="in_progress",
            ),
            TaskState(
                id=2,
                objective="dupe-b",
                inputs={"query": "same"},
                status="in_progress",
            ),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=self._make_search_tcs("same", "same")),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Both stay in_progress (legacy fallback).
        for t in result.plan_state.tasks:
            assert t.status == "in_progress"

    @pytest.mark.asyncio
    async def test_partial_match_completes_some_leaves_others_in_progress(
        self, search_tool: SearchTool
    ):
        """Per-pair semantics: a tool whose args match a task uniquely
        DOES update that task; a tool with no match is skipped (its
        result lives in messages only); a task with no matching tool
        stays in_progress for retry next round.

        plan_state is the driver — only registered (matched) work is
        recorded; unregistered tool calls are ignored at the plan level.
        """
        tasks = [
            TaskState(
                id=1, objective="t1", inputs={"query": "a"}, status="in_progress"
            ),
            TaskState(
                id=2, objective="t2", inputs={"query": "b"}, status="in_progress"
            ),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                # Tool 1 matches task 1 (query="a"); tool 2 ("x") has no
                # matching task → skipped at the plan level.
                StrategyOutput(tool_calls=self._make_search_tcs("a", "x")),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        by_id = {t.id: t for t in result.plan_state.tasks}
        # Task 1 was matched → completed
        assert by_id[1].status == "completed"
        assert by_id[1].result is not None
        # Task 2 was not matched (no tool with query="b") → still in_progress
        assert by_id[2].status == "in_progress"
        assert by_id[2].result is None


class TestNoAutoAdvance:
    """The framework does NOT auto-promote the next task after an action
    completes. After auto-status-update marks the in_progress task
    completed/failed, plan_state has no in_progress task — the model must
    decide what comes next (call planstate_update to mark the next task,
    or call finish, or replan based on the result)."""

    @pytest.mark.asyncio
    async def test_no_auto_advance_after_action_completes_task(
        self, search_tool: SearchTool
    ):
        """After an action runs and auto-update marks task 1 completed, the
        next pending task must stay pending — auto-advance was removed."""
        tasks = [
            TaskState(
                id=1, objective="search", inputs={"query": "x"}, status="in_progress"
            ),
            TaskState(
                id=2, objective="next", status="pending", depends_on=[1]
            ),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Task 1 was auto-completed via Phase D.
        assert result.plan_state.tasks[0].status == "completed"
        # Task 2 stays pending — no auto-advance.
        assert result.plan_state.tasks[1].status == "pending"
        assert result.plan_state.in_progress_task() is None


class TestPhaseDHints:
    """Phase D emits `[plan_state hint]` messages for off-plan, duplicate,
    and partial/ambiguous tool dispatches. Hints are appended onto the
    corresponding tool-result message so the model sees them on the next
    iteration.

    Five cases (per-call classification against the original in_progress
    snapshot):
      5. clean exact (unique exact match, task still available)
         → mark completed; no hint
      2. duplicate (exact match but task already claimed this turn)
         → first call completes the task; second call gets duplicate hint
      3. multi-exact (≥2 distinct tasks exactly match)
         → revision hint
      4. partial overlap (no exact, ≥1 partial match)
         → revision hint
      1. no overlap at all → off-plan hint
    """

    @staticmethod
    def _hint_messages(messages: list[dict]) -> list[dict]:
        return [
            m
            for m in messages
            if m.get("role") == "tool"
            and "[plan_state hint]" in str(m.get("content", ""))
        ]

    @pytest.mark.asyncio
    async def test_case_5_clean_exact_match_no_hint(
        self, search_tool: SearchTool
    ):
        """Case 5 — unique exact match against an available task → task
        marked completed; no hint emitted."""
        tasks = [
            TaskState(
                id=1,
                objective="search",
                inputs={"query": "x"},
                status="in_progress",
            )
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        assert self._hint_messages(result.messages) == []
        assert result.plan_state.tasks[0].status == "completed"

    @pytest.mark.asyncio
    async def test_case_1_no_overlap_off_plan_hint(
        self, search_tool: SearchTool
    ):
        """Case 1 — call args share NOTHING with any in_progress task's
        inputs (different key) → off-plan hint, plan_state untouched."""
        tasks = [
            TaskState(
                id=1,
                objective="t",
                inputs={"unrelated_field": "foo"},
                status="in_progress",
            )
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "bar"},
                            SearchInput(query="bar"),
                            "s1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        hint_msgs = self._hint_messages(result.messages)
        assert len(hint_msgs) == 1
        content = str(hint_msgs[0]["content"])
        assert "Off-plan" in content
        assert "Result for: bar" in content  # tool actually ran
        assert result.plan_state.tasks[0].status == "in_progress"

    @pytest.mark.asyncio
    async def test_case_2_duplicate_dispatch(
        self, search_tool: SearchTool
    ):
        """Case 2 — two parallel tool calls share the same args; both
        exactly match a single in_progress task. The first call claims
        the task and marks it completed; the second call gets a
        duplicate hint."""
        tasks = [
            TaskState(
                id=1,
                objective="t",
                inputs={"query": "x"},
                status="in_progress",
            )
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                # Two parallel SEARCH(q="x") calls, both targeting T1.
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        ),
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s2",
                        ),
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # First call records into T1 (case 5); T1 is completed.
        assert result.plan_state.tasks[0].status == "completed"
        # Exactly one duplicate hint — on the second tool message.
        hint_msgs = self._hint_messages(result.messages)
        assert len(hint_msgs) == 1
        content = str(hint_msgs[0]["content"])
        assert "Duplicate dispatch" in content
        assert "task id=1" in content
        # The hinted message is the SECOND call (s2), not the first.
        assert hint_msgs[0]["tool_call_id"] == "s2"

    @pytest.mark.asyncio
    async def test_case_3_multi_exact_revision_hint(
        self, search_tool: SearchTool
    ):
        """Case 3 — one tool call exactly matches ≥2 in_progress tasks
        (plan has duplicate inputs) → revision hint listing exact ids,
        plan_state untouched."""
        tasks = [
            TaskState(
                id=1,
                objective="t1",
                inputs={"query": "same"},
                status="in_progress",
            ),
            TaskState(
                id=2,
                objective="t2",
                inputs={"query": "same"},
                status="in_progress",
            ),
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "same"},
                            SearchInput(query="same"),
                            "s1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        hint_msgs = self._hint_messages(result.messages)
        assert len(hint_msgs) == 1
        content = str(hint_msgs[0]["content"])
        assert "Plan needs revision" in content
        assert "exactly matches 2" in content
        assert "[1, 2]" in content
        for t in result.plan_state.tasks:
            assert t.status == "in_progress"

    @pytest.mark.asyncio
    async def test_case_4_partial_overlap_revision_hint(
        self, calculator_tool
    ):
        """Case 4 — call args partially overlap a task's inputs (some
        keys/values agree, others don't) → revision hint listing partial
        ids, plan_state untouched.

        Setup: task has inputs={"expression":"1+1","extra":"k"}; call
        passes only {"expression":"1+1"}. Shared (expression, "1+1") so
        they partially overlap, but inputs are not dict-equal.
        """
        from agents.agent_tool.tests.common_fixtures import CalculatorInput

        tasks = [
            TaskState(
                id=1,
                objective="compute",
                inputs={"expression": "1+1", "extra": "k"},
                status="in_progress",
            )
        ]
        update_input = PlanStateUpdateInput(tasks=tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "calculator",
                            {"expression": "1+1"},
                            CalculatorInput(expression="1+1"),
                            "c1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[calculator_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        hint_msgs = self._hint_messages(result.messages)
        assert len(hint_msgs) == 1
        content = str(hint_msgs[0]["content"])
        assert "Plan needs revision" in content
        assert "partially overlaps" in content
        assert "[1]" in content
        assert result.plan_state.tasks[0].status == "in_progress"

    @pytest.mark.asyncio
    async def test_single_step_no_plan_no_hint(
        self, search_tool: SearchTool
    ):
        """Single-step "no plan" mode (plan_state.tasks empty) → Phase D
        skips the contract; no hint."""
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        assert self._hint_messages(result.messages) == []
        assert result.plan_state.tasks == []


class TestSummarizePlanResult:
    def test_empty_plan(self):
        plan = PlanState(objective="x")
        s = _summarize_plan_result(plan)
        assert "No tasks" in s
        assert "draft" in s

    def test_mixed_statuses(self):
        plan = PlanState(
            objective="x",
            status="failed",
            tasks=[
                TaskState(id=1, objective="a", status="completed"),
                TaskState(id=2, objective="b", status="failed"),
                TaskState(id=3, objective="c", status="blocked"),
                TaskState(id=4, objective="d", status="completed"),
            ],
        )
        s = _summarize_plan_result(plan)
        assert "failed" in s
        assert "Completed: 2/4" in s
        assert "Failed: 1" in s
        assert "Blocked: 1" in s


class TestMetaToolPolicyGuard:
    """Tests the strict meta-tool emission policy in _execute_tool_calls.

    Each meta tool (planstate_update, finish) MUST be emitted alone in its
    own iteration. Mixing them with each other or with action tools is a
    policy violation: the framework refuses to execute any tool in that turn
    and returns a uniform error result for each call so the model can retry.

    This guarantees plan_state cannot diverge from reality (e.g., a planstate
    update failing while an action's side effect already happened).
    """

    @pytest.mark.asyncio
    async def test_planstate_update_alone_executes_normally(
        self, search_tool: SearchTool
    ):
        """A single planstate_update on its own turn runs and mutates plan_state."""
        new_tasks = [
            TaskState(id=1, objective="t", status="in_progress")
        ]
        update_input = PlanStateUpdateInput(tasks=new_tasks)
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert result.success
        assert result.plan_state.revision_count == 1
        assert result.plan_state.tasks[0].status == "in_progress"

    @pytest.mark.asyncio
    async def test_finish_alone_terminates(self):
        """A single finish on its own turn terminates the loop normally.
        AgentTool extracts result/success from finish.arguments."""
        finish_args = {"result": "done", "success": True}
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        ),
                    ],
                ),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert result.success
        assert result.iterations_used == 1
        assert result.result == "done"
        # FINISH must sync plan_state.status to match its outcome so the
        # returned PlanState reflects that the run terminated.
        assert result.plan_state.status == "completed"

    @pytest.mark.asyncio
    async def test_finish_with_failure_marks_plan_failed(
        self, search_tool: SearchTool
    ):
        """FINISH(success=False) must set plan_state.status='failed' so the
        returned PlanState reflects the run's terminal outcome. This run
        does the pre-FINISH housekeeping (marks tasks cancelled/failed)
        first, then calls FINISH — the framework now requires all tasks
        be terminal before FINISH is honored."""
        # Iter 1: initial plan with one in_progress + one pending.
        initial_tasks = [
            TaskState(id=1, objective="t", status="in_progress"),
            TaskState(id=2, objective="t2", status="pending"),
        ]
        initial_update = PlanStateUpdateInput(tasks=initial_tasks)
        # Iter 2: cleanup — mark both tasks cancelled (gave up before
        # they were attempted/completed).
        cleanup_tasks = [
            TaskState(id=1, objective="t", status="cancelled"),
            TaskState(id=2, objective="t2", status="cancelled"),
        ]
        cleanup_update = PlanStateUpdateInput(tasks=cleanup_tasks)
        finish_args = {
            "result": "stopped: retry budget exhausted",
            "success": False,
        }
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            initial_update.model_dump(),
                            initial_update,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            cleanup_update.model_dump(),
                            cleanup_update,
                            "u2",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        )
                    ]
                ),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert not result.success
        assert result.plan_state.status == "failed"
        # Tasks were explicitly cancelled by the cleanup update before FINISH.
        assert result.plan_state.tasks[0].status == "cancelled"
        assert result.plan_state.tasks[1].status == "cancelled"

    @pytest.mark.asyncio
    async def test_finish_rejected_when_non_terminal_tasks_remain(
        self, search_tool: SearchTool
    ):
        """If the model emits FINISH while plan_state has non-terminal
        tasks, the framework rejects FINISH by rewriting its tool result
        into an error message. The loop continues so the model can do
        the housekeeping via planstate_update on the next iteration."""
        # Iter 1: plan with one pending task (non-terminal at FINISH time).
        tasks = [TaskState(id=1, objective="t", status="pending")]
        update_input = PlanStateUpdateInput(tasks=tasks)
        # Iter 2: model tries to FINISH while task 1 is still pending.
        finish_args = {"result": "premature finish", "success": True}
        # Iter 3: cleanup + finish.
        cleanup_tasks = [
            TaskState(id=1, objective="t", status="cancelled"),
        ]
        cleanup_update = PlanStateUpdateInput(tasks=cleanup_tasks)
        finish_args_final = {
            "result": "done after cleanup",
            "success": True,
        }

        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            cleanup_update.model_dump(),
                            cleanup_update,
                            "u2",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args_final,
                            FinishInput(**finish_args_final),
                            "f2",
                        )
                    ]
                ),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Final outcome reflects the SECOND finish (after housekeeping).
        assert result.success
        assert result.result == "done after cleanup"
        assert result.iterations_used == 4
        # The rejected first finish should have left an error tool result
        # in the message history (under tool_call_id "f1").
        rejected_tool_msg = next(
            (
                m
                for m in result.messages
                if m.get("role") == "tool" and m.get("tool_call_id") == "f1"
            ),
            None,
        )
        assert rejected_tool_msg is not None
        assert "rejected" in rejected_tool_msg["content"]
        assert "non-terminal" in rejected_tool_msg["content"]
        # Task is cancelled after cleanup.
        assert result.plan_state.tasks[0].status == "cancelled"
        assert result.plan_state.status == "completed"

    @pytest.mark.asyncio
    async def test_finish_allowed_with_empty_tasks(self):
        """If plan_state has no tasks at all (e.g., single-step run
        without planstate_update), FINISH is allowed without housekeeping
        — there's nothing to clean up."""
        finish_args = {"result": "done", "success": True}
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        )
                    ]
                ),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert result.success
        assert result.iterations_used == 1
        assert result.plan_state.tasks == []
        assert result.plan_state.status == "completed"

    @pytest.mark.asyncio
    async def test_planstate_update_with_action_is_blocked(
        self, search_tool: SearchTool
    ):
        """Mixing planstate_update with an action tool: NEITHER runs.
        Both get a policy-violation error result. The model retries next iter."""
        update_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="t", status="in_progress")]
        )
        # Iter 1 emits the forbidden combination.
        # Iter 2 emits planstate_update alone (model corrects).
        # Iter 3 emits the action alone.
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        ),
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        ),
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u2",
                        )
                    ]
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s2",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        assert result.success
        # Iter 1: policy violation — both calls get error tool results, neither
        # actually executed → plan_state should NOT have been mutated.
        # The first two tool result messages should be policy-violation errors.
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_results) >= 2
        first_two = [tr["content"] for tr in tool_results[:2]]
        for content in first_two:
            assert "Tool emission policy violation" in content
            assert "must be emitted ALONE" in content
        # By the end, the model corrected itself and the run completed.

    @pytest.mark.asyncio
    async def test_planstate_update_with_finish_is_blocked(self):
        """Mixing planstate_update with finish: BOTH blocked, neither runs.
        AgentTool detects finish from EXECUTED tool_calls, so a rejected
        finish does NOT terminate the loop. The model gets a policy
        error and must retry with each meta tool alone."""
        update_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="t", status="completed")],
            plan_status="completed",
        )
        finish_args = {"result": "done", "success": True}
        # Iter 1: forbidden combo. Iter 2: finish alone (model corrects).
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        ),
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        ),
                    ],
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f2",
                        ),
                    ],
                ),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Iter 1 was a policy violation → no tool ran, no termination signal.
        # Iter 2 emitted finish alone → AgentTool terminated with finish args.
        assert result.success is True
        assert result.result == "done"
        assert result.iterations_used == 2

        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        # Iter 1 emitted 2 tool calls, both got policy-violation error results.
        first_two = [tr["content"] for tr in tool_results[:2]]
        for content in first_two:
            assert "Tool emission policy violation" in content
        # The rejected planstate_update from iter 1 did NOT mutate plan_state
        # (revision_count is the proof). The "completed" status comes from
        # iter 2's successful FINISH(success=True), which syncs plan_status
        # to its terminal outcome.
        assert result.plan_state.revision_count == 0
        assert result.plan_state.status == "completed"

    @pytest.mark.asyncio
    async def test_multiple_planstate_updates_in_one_turn_blocked(self):
        """Two planstate_updates in one turn: both blocked."""
        u1 = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="a", status="pending")]
        )
        u2 = PlanStateUpdateInput(
            tasks=[TaskState(id=2, objective="b", status="pending")]
        )
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc("planstate_update", u1.model_dump(), u1, "u1"),
                        _tc("planstate_update", u2.model_dump(), u2, "u2"),
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        # Neither planstate_update ran — plan_state unchanged.
        assert result.plan_state.revision_count == 0
        assert result.plan_state.tasks == []

    @pytest.mark.asyncio
    async def test_parallel_action_tools_alone_still_allowed(
        self, search_tool: SearchTool, calculator_tool
    ):
        """Multiple non-meta tools in the same turn are allowed (parallel batch)."""
        from agents.agent_tool.tests.common_fixtures import CalculatorInput

        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        ),
                        _tc(
                            "calculator",
                            {"expression": "1+1"},
                            CalculatorInput(expression="1+1"),
                            "c1",
                        ),
                    ]
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(
            tools=[search_tool, calculator_tool], strategy=strategy
        )
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert result.success
        # Both action tools ran (no policy violation)
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        contents = [tr["content"] for tr in tool_results]
        assert any("Result for: x" in c for c in contents)
        assert any('"result":2' in c or '"result":2.0' in c for c in contents)


class TestParseErrorRecovery:
    """Tests that invalid tool arguments don't crash the run — they surface as
    tool result errors so the model can correct itself on the next iteration.
    """

    @pytest.mark.asyncio
    async def test_parse_error_surfaces_as_tool_result_and_run_continues(
        self, search_tool: SearchTool
    ):
        """A ToolCall with parse_error returns a tool error result; the loop
        continues and the model can retry."""
        # Iter 1: emit a search call with parse_error (simulates model emitting
        # invalid arguments that validation rejected).
        broken_tc = ToolCall(
            id="b1",
            tool_name="search",
            arguments={"unknown_field": "x"},
            parsed=None,
            parse_error="1 validation error for SearchInput\nquery\n  Field required",
        )
        # Iter 2: emit a corrected search call.
        good_tc = _tc("search", {"query": "ok"}, SearchInput(query="ok"), "g1")

        strategy = _ScriptedStrategy(
            [
                StrategyOutput(tool_calls=[broken_tc]),
                StrategyOutput(tool_calls=[good_tc]),
                StrategyOutput(tool_calls=[], result="recovered"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Run survived to completion
        assert result.success
        assert result.iterations_used == 3

        # Iter 1's tool result should contain the parse error message
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_results) == 2
        first_result_payload = tool_results[0]["content"]
        assert "Invalid arguments" in first_result_payload
        assert "Field required" in first_result_payload

        # Iter 2's tool result should be the actual SearchTool output
        second_result_payload = tool_results[1]["content"]
        assert "Result for: ok" in second_result_payload

    @pytest.mark.asyncio
    async def test_parse_error_drives_auto_status_update_to_failed(
        self, search_tool: SearchTool
    ):
        """If the only non-meta tool call had parse_error, auto-update marks
        the in_progress task as failed (not completed) and stores the error.

        Per the meta-tool policy, planstate_update must be emitted alone, so
        we mark the task in_progress in iter 1 (planstate_update alone) and
        emit the broken action call in iter 2.
        """
        # Iter 1: planstate_update alone marks task 1 in_progress with
        # inputs that match what the model will emit (broken call below
        # has the same keys/values — semantically: model emitted args
        # that look right structurally but the parser rejected something
        # else, e.g., a value type mismatch). Strict pairing requires the
        # call's args dict-equal the task's inputs.
        new_tasks = [
            TaskState(id=1, objective="t", inputs={"query": "x"},
                      status="in_progress")
        ]
        update_input = PlanStateUpdateInput(tasks=new_tasks)

        # Iter 2: broken search call with parse_error. The arguments
        # match the task's inputs so Phase D's strict pairing finds the
        # task; the parse_error path then marks it `failed`.
        broken_tc = ToolCall(
            id="b1",
            tool_name="search",
            arguments={"query": "x"},
            parsed=None,
            parse_error="validation error",
        )

        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                ),
                StrategyOutput(tool_calls=[broken_tc]),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Auto-update fires on iter 2 (one non-meta tool ran, task 1 was
        # in_progress) and marks the task failed because the broken tool
        # returned an error result.
        target = result.plan_state.tasks[0]
        assert target.status == "failed"
        assert target.result is not None
        assert "Invalid arguments" in target.result
