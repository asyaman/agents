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
    return ToolCall(tool_name=tool_name, arguments=arguments, parsed=parsed, id=call_id)


class _ScriptedStrategy(PlanningStrategy):
    """Returns a fixed sequence of StrategyOutput objects."""

    def __init__(self, outputs):
        self._outputs = outputs
        self.calls = 0
        self.received_plan_states: list = []

    async def plan(self, messages, tools, parallel_tool_calls=True, plan_state=None):
        self.received_plan_states.append(plan_state)
        out = self._outputs[self.calls]
        self.calls += 1
        return out


class TestAgentToolPlanStateOutput:
    @pytest.mark.asyncio
    async def test_plan_state_returned_in_output(self, search_tool: SearchTool):
        strategy = _ScriptedStrategy([StrategyOutput(tool_calls=[], result="Done")])
        agent = AgentTool(tools=[search_tool], strategy=strategy)

        result = await agent.ainvoke(AgentToolInput(objective="My objective"))

        assert result.plan_state is not None
        assert result.plan_state.objective == "My objective"
        assert result.plan_state.tasks == []
        assert result.plan_state.revision_count == 0
        assert result.plan_state.status == "draft"


class TestAgentToolPlanStateInjection:
    @pytest.mark.asyncio
    async def test_planstate_update_tool_passed_to_strategy(self, search_tool: SearchTool):
        captured: list = []

        class CapturingStrategy(PlanningStrategy):
            async def plan(self, messages, tools, parallel_tool_calls=True, plan_state=None):
                captured.append([t.name for t in tools])
                return StrategyOutput(tool_calls=[], result="ok")

        agent = AgentTool(tools=[search_tool], strategy=CapturingStrategy())
        await agent.ainvoke(AgentToolInput(objective="x"))

        assert captured, "strategy.plan was not called"
        assert "PLANSTATE_UPDATE" in captured[0]
        assert "FINISH" in captured[0]
        assert "SEARCH" in captured[0]

    @pytest.mark.asyncio
    async def test_planstate_update_can_be_disabled(self, search_tool: SearchTool):
        captured: list = []

        class CapturingStrategy(PlanningStrategy):
            async def plan(self, messages, tools, parallel_tool_calls=True, plan_state=None):
                captured.append([t.name for t in tools])
                return StrategyOutput(tool_calls=[], result="ok")

        agent = AgentTool(
            tools=[search_tool],
            strategy=CapturingStrategy(),
            enable_plan_state=False,
        )
        await agent.ainvoke(AgentToolInput(objective="x"))

        assert "PLANSTATE_UPDATE" not in captured[0]

    @pytest.mark.asyncio
    async def test_strategy_receives_plan_state_by_reference(self, search_tool: SearchTool):
        strategy = _ScriptedStrategy([StrategyOutput(tool_calls=[], result="ok")])
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
        update_input = PlanStateUpdateInput(tasks=tasks, plan_status="completed")
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
        update_input = PlanStateUpdateInput(tasks=tasks, plan_status="failed")
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


class TestFrameworkDoesNotMutatePlanStateFromActions:
    """The framework dispatches action tools and captures results in
    messages but does NOT mutate plan_state. The model handles all
    plan_state transitions via planstate_update (reconciliation)."""

    @pytest.mark.asyncio
    async def test_single_action_tool_leaves_plan_state_unchanged(self, search_tool: SearchTool):
        """After an action tool runs, the in_progress task remains
        in_progress with result=None. Plan_state is the model's
        responsibility to reconcile on the next iteration."""
        tasks = [
            TaskState(
                id=1,
                objective="Search",
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
        # Framework did NOT auto-update — task stays in_progress.
        assert result.plan_state.tasks[0].status == "in_progress"
        assert result.plan_state.tasks[0].result is None
        # But the tool's output IS in the message history for the
        # model to reconcile from.
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert any("Result for: x" in str(m.get("content", "")) for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_planstate_update_alone_does_not_trigger_autoupdate(
        self, search_tool: SearchTool
    ):
        """A turn that only calls planstate_update should not auto-update statuses
        (planstate_update is a meta tool)."""
        tasks = [TaskState(id=1, objective="t", inputs={"q": "x"}, status="in_progress")]
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
        agent = AgentTool(tools=[search_tool, calculator_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Auto-update was skipped — task 1 stays in_progress
        assert result.plan_state.tasks[0].status == "in_progress"


class TestNoAutoAdvance:
    """The framework does NOT mutate plan_state from tool results and does
    NOT auto-advance to the next task. Both `in_progress` (post-action)
    and `pending` (next step) stay as-is until the model reconciles via
    planstate_update."""

    @pytest.mark.asyncio
    async def test_no_auto_advance_after_action_completes_task(self, search_tool: SearchTool):
        """After an action runs, plan_state is untouched: task 1 stays
        in_progress (no auto-status-update), task 2 stays pending (no
        auto-advance). The model is expected to reconcile next
        iteration."""
        tasks = [
            TaskState(id=1, objective="search", status="in_progress"),
            TaskState(id=2, objective="next", status="pending", depends_on=[1]),
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

        # Framework didn't auto-update: task 1 still in_progress.
        assert result.plan_state.tasks[0].status == "in_progress"
        # Task 2 stays pending — no auto-advance.
        assert result.plan_state.tasks[1].status == "pending"
        # in_progress_task() still returns task 1 (model hasn't reconciled).
        assert result.plan_state.in_progress_task() is result.plan_state.tasks[0]


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
    async def test_planstate_update_alone_executes_normally(self, search_tool: SearchTool):
        """A single planstate_update on its own turn runs and mutates plan_state."""
        new_tasks = [TaskState(id=1, objective="t", status="in_progress")]
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
    async def test_finish_with_failure_marks_plan_failed(self, search_tool: SearchTool):
        """FINISH(success=False) must set plan_state.status='failed' so the
        returned PlanState reflects the run's terminal outcome. This run
        does the pre-FINISH housekeeping (marks tasks cancelled/failed)
        together with FINISH in a Reconcile-and-finish bundle — the
        framework requires all tasks be terminal before FINISH is honored,
        and the back-to-back planstate_update rule forces this housekeeping
        to be co-emitted with FINISH rather than split into a separate
        planstate-only iteration."""
        # Iter 1: initial plan with one in_progress + one pending.
        initial_tasks = [
            TaskState(id=1, objective="t", status="in_progress"),
            TaskState(id=2, objective="t2", status="pending"),
        ]
        initial_update = PlanStateUpdateInput(tasks=initial_tasks)
        # Iter 2: Reconcile-and-finish bundle — cleanup + finish together.
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
                        ),
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        ),
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
    async def test_finish_rejected_when_non_terminal_tasks_remain(self, search_tool: SearchTool):
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
    async def test_planstate_update_with_action_is_blocked(self, search_tool: SearchTool):
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
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_results) >= 2
        first_two = [tr["content"] for tr in tool_results[:2]]
        for content in first_two:
            assert "Tool emission policy violation" in content
        # By the end, the model corrected itself and the run completed.

    @pytest.mark.asyncio
    async def test_planstate_update_with_finish_is_allowed_reconcile_and_finish(
        self,
    ):
        """planstate_update + finish co-emission is ALLOWED — this is the
        Reconcile-and-finish mode. The framework runs planstate_update
        FIRST so its mutations (marking remaining tasks terminal) are
        visible before finish's pre-termination housekeeping check runs.
        """
        update_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="t", status="completed")],
            plan_status="completed",
        )
        finish_args = {"result": "done", "success": True}
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        # Bundle: planstate_update + finish in one turn.
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
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Bundle accepted; finish terminated the run.
        assert result.success is True
        assert result.result == "done"
        assert result.iterations_used == 1
        # planstate_update ran (mutated plan_state) BEFORE finish — task 1
        # is completed and plan_status is the terminal value.
        assert result.plan_state.revision_count == 1
        assert result.plan_state.tasks[0].status == "completed"
        assert result.plan_state.status == "completed"
        # The two tool result messages should NOT be policy-violation errors.
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        for tr in tool_results:
            assert "Tool emission policy violation" not in tr["content"]

    @pytest.mark.asyncio
    async def test_planstate_update_finish_runs_planstate_first(self):
        """When planstate_update + finish are emitted in EITHER order,
        the framework reorders so planstate_update runs first. This
        ensures the pre-termination housekeeping check sees the mutated
        plan_state."""
        # Plan starts with a non-terminal task. The bundle's
        # planstate_update marks it terminal, which lets the housekeeping
        # check pass. If finish ran first, the check would reject.
        update_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="t", status="cancelled")],
        )
        finish_args = {"result": "done", "success": True}
        strategy = _ScriptedStrategy(
            [
                # Emit finish BEFORE planstate_update in the strategy
                # output. The framework should sort meta tools so
                # planstate_update runs first.
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "finish",
                            finish_args,
                            FinishInput(**finish_args),
                            "f1",
                        ),
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        ),
                    ],
                ),
            ]
        )
        # Pre-existing plan_state has a pending task (will become
        # cancelled via the bundled planstate_update).
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # The bundle ran cleanly — planstate_update first set the task
        # to `cancelled`, then finish saw all-terminal plan_state and
        # honored termination.
        assert result.success is True
        assert result.plan_state.tasks[0].status == "cancelled"

    @pytest.mark.asyncio
    async def test_multiple_planstate_updates_in_one_turn_blocked(self):
        """Two planstate_updates in one turn: both blocked."""
        u1 = PlanStateUpdateInput(tasks=[TaskState(id=1, objective="a", status="pending")])
        u2 = PlanStateUpdateInput(tasks=[TaskState(id=2, objective="b", status="pending")])
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
        agent = AgentTool(tools=[search_tool, calculator_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert result.success
        # Both action tools ran (no policy violation)
        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        contents = [tr["content"] for tr in tool_results]
        assert any("Result for: x" in c for c in contents)
        assert any('"result":2' in c or '"result":2.0' in c for c in contents)


class TestNoBackToBackPlanStateUpdates:
    """Cross-iteration rule: two consecutive `planstate_update`-only
    iterations are rejected by the framework BEFORE any tool runs. The
    second turn's tool calls all receive an auto-correct error so the
    model knows to switch to action / finish / planstate+finish.

    Previous planstate_update emissions that errored (parse_error,
    runtime error) do NOT trigger the rule — the model never actually
    reconciled, so retrying is legitimate.
    """

    @pytest.mark.asyncio
    async def test_back_to_back_planstate_only_is_rejected(self, search_tool: SearchTool):
        """Two successful planstate_update-only iterations in a row:
        the second one is rejected with an auto-correct hint, no
        mutation happens, and the model can recover on the next
        iteration by emitting an action tool."""
        u1 = PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t1", status="in_progress")])
        u2 = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="t1-renamed", status="in_progress")]
        )
        strategy = _ScriptedStrategy(
            [
                # Iter 1: planstate_update only (legitimate initial plan)
                StrategyOutput(
                    tool_calls=[
                        _tc("planstate_update", u1.model_dump(), u1, "u1"),
                    ],
                ),
                # Iter 2: planstate_update only AGAIN — back-to-back, REJECTED
                StrategyOutput(
                    tool_calls=[
                        _tc("planstate_update", u2.model_dump(), u2, "u2"),
                    ],
                ),
                # Iter 3: model recovers — emits action tool
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        ),
                    ],
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        # Iter 1's planstate_update ran (revision_count == 1).
        # Iter 2's planstate_update was rejected — revision_count stays at 1
        # and task objective is NOT updated to "t1-renamed".
        assert result.plan_state.revision_count == 1
        assert result.plan_state.tasks[0].objective == "t1"

        # The rejected u2 tool result has the auto-correct error payload.
        rejected_msg = next(
            (
                m
                for m in result.messages
                if m.get("role") == "tool" and m.get("tool_call_id") == "u2"
            ),
            None,
        )
        assert rejected_msg is not None
        assert "Back-to-back planstate_update" in rejected_msg["content"]
        # Hint mentions the three legal next moves
        assert "action tool" in rejected_msg["content"]
        assert "finish" in rejected_msg["content"]
        assert "Reconcile-and-finish" in rejected_msg["content"]

    @pytest.mark.asyncio
    async def test_planstate_then_action_is_allowed(self, search_tool: SearchTool):
        """planstate_update-only → action tool: not back-to-back, runs normally."""
        u1 = PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t1", status="in_progress")])
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc("planstate_update", u1.model_dump(), u1, "u1"),
                    ],
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "search",
                            {"query": "x"},
                            SearchInput(query="x"),
                            "s1",
                        ),
                    ],
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        assert result.success
        # Search ran cleanly (no policy violation)
        search_tool_msg = next(
            (
                m
                for m in result.messages
                if m.get("role") == "tool" and m.get("tool_call_id") == "s1"
            ),
            None,
        )
        assert search_tool_msg is not None
        assert "Back-to-back" not in search_tool_msg["content"]

    @pytest.mark.asyncio
    async def test_planstate_then_planstate_plus_finish_is_allowed(self):
        """planstate_update-only → planstate_update+finish bundle is NOT
        back-to-back (the second turn is not planstate-only; it's the
        Reconcile-and-finish bundle)."""
        u1 = PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t1", status="in_progress")])
        u2 = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="t1", status="completed")],
            plan_status="completed",
        )
        finish_args = {"result": "done", "success": True}
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(
                    tool_calls=[
                        _tc("planstate_update", u1.model_dump(), u1, "u1"),
                    ],
                ),
                StrategyOutput(
                    tool_calls=[
                        _tc("planstate_update", u2.model_dump(), u2, "u2"),
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
        assert result.iterations_used == 2
        # Bundle's planstate_update DID run (revision_count went 1 → 2).
        assert result.plan_state.revision_count == 2
        assert result.plan_state.tasks[0].status == "completed"
        # u2 tool result is NOT an auto-correct error
        u2_msg = next(
            (
                m
                for m in result.messages
                if m.get("role") == "tool" and m.get("tool_call_id") == "u2"
            ),
            None,
        )
        assert u2_msg is not None
        assert "Back-to-back" not in u2_msg["content"]

    @pytest.mark.asyncio
    async def test_planstate_errored_then_retry_planstate_is_allowed(self):
        """If the previous planstate_update emission errored (parse_error
        or runtime error), retrying planstate_update on the next
        iteration is NOT back-to-back — the model never actually
        reconciled, so the retry is legitimate."""
        # Build a planstate_update ToolCall with parse_error set; the
        # framework should surface it as a tool error.
        u_good = PlanStateUpdateInput(tasks=[TaskState(id=1, objective="t1", status="in_progress")])
        # Iter 1: planstate_update with parse_error (simulating bad args)
        bad_tc = ToolCall(
            tool_name="planstate_update",
            arguments={"oops": "bad"},
            parsed=None,
            id="u_bad",
            parse_error="Invalid args: missing required field 'tasks'",
        )
        strategy = _ScriptedStrategy(
            [
                StrategyOutput(tool_calls=[bad_tc]),
                # Iter 2: model retries with valid args — back-to-back
                # check must NOT fire because iter 1's emission errored.
                StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            u_good.model_dump(),
                            u_good,
                            "u_good",
                        ),
                    ],
                ),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))
        # Iter 2's planstate_update actually ran (revision_count=1)
        assert result.plan_state.revision_count == 1
        assert result.plan_state.tasks[0].objective == "t1"
        # Iter 2's tool result is NOT the back-to-back rejection
        u_good_msg = next(
            (
                m
                for m in result.messages
                if m.get("role") == "tool" and m.get("tool_call_id") == "u_good"
            ),
            None,
        )
        assert u_good_msg is not None
        assert "Back-to-back" not in u_good_msg["content"]


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
    async def test_consecutive_parse_errors_get_corrective_hint(self, search_tool: SearchTool):
        """When the SAME tool parse-errors two iterations in a row, the
        second error result includes an extra corrective hint so the
        model knows to trim large audit fields rather than retrying the
        same malformed shape."""
        first_broken = ToolCall(
            id="b1",
            tool_name="search",
            arguments={"unknown": "x"},
            parsed=None,
            parse_error="1 validation error for SearchInput\nquery\n  Field required",
        )
        second_broken = ToolCall(
            id="b2",
            tool_name="search",
            arguments={"oops": "y"},
            parsed=None,
            parse_error="1 validation error for SearchInput\nquery\n  Field required",
        )
        good_tc = _tc("search", {"query": "ok"}, SearchInput(query="ok"), "g1")

        strategy = _ScriptedStrategy(
            [
                StrategyOutput(tool_calls=[first_broken]),
                StrategyOutput(tool_calls=[second_broken]),
                StrategyOutput(tool_calls=[good_tc]),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        # First parse error has NO hint (no prior call yet)
        first_content = tool_results[0]["content"]
        assert "Invalid arguments" in first_content
        assert "Consecutive parse_error" not in first_content

        # Second parse error has the corrective hint
        second_content = tool_results[1]["content"]
        assert "Invalid arguments" in second_content
        assert "Consecutive parse_error" in second_content
        assert "task.result" in second_content
        assert "SHORT" in second_content

        # Third call succeeded, no hint
        third_content = tool_results[2]["content"]
        assert "Result for: ok" in third_content

    @pytest.mark.asyncio
    async def test_parse_error_then_different_tool_succeeds_no_hint(
        self, search_tool: SearchTool, calculator_tool
    ):
        """If the previous parse_error was for a DIFFERENT tool, the
        current tool's parse_error should NOT get the consecutive hint
        (the sequence is not consecutive for the same tool)."""
        from agents.agent_tool.tests.common_fixtures import CalculatorInput

        # Iter 1: search parse-errors
        search_broken = ToolCall(
            id="s1",
            tool_name="search",
            arguments={},
            parsed=None,
            parse_error="1 validation error for SearchInput\nquery\n  Field required",
        )
        # Iter 2: calculator parse-errors (different tool — not consecutive)
        calc_broken = ToolCall(
            id="c1",
            tool_name="calculator",
            arguments={},
            parsed=None,
            parse_error="1 validation error for CalculatorInput\nexpression\n  Field required",
        )
        good_tc = _tc(
            "calculator",
            {"expression": "1+1"},
            CalculatorInput(expression="1+1"),
            "c2",
        )

        strategy = _ScriptedStrategy(
            [
                StrategyOutput(tool_calls=[search_broken]),
                StrategyOutput(tool_calls=[calc_broken]),
                StrategyOutput(tool_calls=[good_tc]),
                StrategyOutput(tool_calls=[], result="done"),
            ]
        )
        agent = AgentTool(tools=[search_tool, calculator_tool], strategy=strategy)
        result = await agent.ainvoke(AgentToolInput(objective="goal"))

        tool_results = [m for m in result.messages if m.get("role") == "tool"]
        # Iter 2's calculator parse error should NOT have the hint —
        # the previous parse_error was for search, not calculator.
        second_content = tool_results[1]["content"]
        assert "Invalid arguments" in second_content
        assert "Consecutive parse_error" not in second_content

    @pytest.mark.asyncio
    async def test_parse_error_leaves_task_in_progress_for_model_reconciliation(
        self, search_tool: SearchTool
    ):
        """If a non-meta tool call had parse_error, the framework does
        NOT auto-mutate plan_state. The error surfaces in the tool
        message; the model is expected to reconcile (mark task `failed`)
        on the next iteration via planstate_update.
        """
        new_tasks = [TaskState(id=1, objective="t", status="in_progress")]
        update_input = PlanStateUpdateInput(tasks=new_tasks)

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

        # Framework did NOT auto-mutate plan_state. Task stays
        # in_progress; the model is expected to reconcile next iter.
        target = result.plan_state.tasks[0]
        assert target.status == "in_progress"
        assert target.result is None
        # The parse error IS in message history for the model to read.
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert any("Invalid arguments" in str(m.get("content", "")) for m in tool_msgs)
