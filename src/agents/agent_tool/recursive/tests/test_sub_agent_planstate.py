"""Sub-agent integration tests focused on PlanState isolation.

Each AgentTool.ainvoke() creates its own PlanState. SubAgentTool constructs
a fresh AgentTool per child invocation, so child agents must NEVER see the
parent's plan and vice-versa.
"""

import pytest

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.agent_tool.meta_tool_plan_state_update import PlanStateUpdateInput
from agents.agent_tool.plan_state import TaskState
from agents.agent_tool.recursive.context import initialize_recursion_context
from agents.agent_tool.recursive.sub_agent_tool import (
    SubAgentInput,
    SubAgentTool,
)
from agents.llm_core.llm_client import ToolCall


def _tc(tool_name, arguments, parsed, call_id):
    return ToolCall(
        tool_name=tool_name, arguments=arguments, parsed=parsed, id=call_id
    )


@pytest.mark.asyncio
async def test_two_agent_invocations_get_independent_plan_states():
    """Two ainvoke calls on the same AgentTool must get independent PlanState."""

    class WriteOnceStrategy(PlanningStrategy):
        def __init__(self, marker: str):
            self.marker = marker
            self.calls = 0

        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            self.calls += 1
            if self.calls == 1:
                tasks = [
                    TaskState(
                        id=1, objective=f"task for {self.marker}",
                        status="pending",
                    )
                ]
                update_input = PlanStateUpdateInput(tasks=tasks)
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "u1",
                        )
                    ]
                )
            return StrategyOutput(
                finished=True, success=True, result=f"done {self.marker}"
            )

    agent_a = AgentTool(tools=[], strategy=WriteOnceStrategy("A"))
    agent_b = AgentTool(tools=[], strategy=WriteOnceStrategy("B"))

    result_a = await agent_a.ainvoke(AgentToolInput(objective="goal A"))
    result_b = await agent_b.ainvoke(AgentToolInput(objective="goal B"))

    assert result_a.plan_state is not result_b.plan_state
    assert result_a.plan_state.objective == "goal A"
    assert result_b.plan_state.objective == "goal B"
    assert result_a.plan_state.tasks[0].objective == "task for A"
    assert result_b.plan_state.tasks[0].objective == "task for B"


@pytest.mark.asyncio
async def test_subagent_child_has_isolated_plan_state():
    """A SubAgentTool invocation must give the child its own PlanState.

    Parent's plan must contain parent tasks; child's plan must contain child
    tasks; neither leaks into the other.
    """
    initialize_recursion_context(max_depth=3)

    parent_marker_objective = "parent task only"
    child_marker_objective = "child task only"

    class ParentStrategy(PlanningStrategy):
        def __init__(self):
            self.calls = 0

        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            self.calls += 1
            if self.calls == 1:
                tasks = [
                    TaskState(
                        id=1, objective=parent_marker_objective,
                        status="in_progress",
                    )
                ]
                update_input = PlanStateUpdateInput(tasks=tasks)
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "wp1",
                        )
                    ]
                )
            if self.calls == 2:
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "delegate_subtask",
                            {"sub_objective": "child sub-objective"},
                            SubAgentInput(sub_objective="child sub-objective"),
                            "d1",
                        )
                    ]
                )
            return StrategyOutput(
                finished=True, success=True, result="parent done"
            )

    class ChildStrategy(PlanningStrategy):
        def __init__(self):
            self.calls = 0
            self.observed_plan_states: list = []

        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            self.observed_plan_states.append(plan_state)
            self.calls += 1
            if self.calls == 1:
                tasks = [
                    TaskState(
                        id=1, objective=child_marker_objective,
                        status="in_progress",
                    )
                ]
                update_input = PlanStateUpdateInput(tasks=tasks)
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "wc1",
                        )
                    ]
                )
            return StrategyOutput(
                finished=True, success=True, result="child done"
            )

    child_strategy = ChildStrategy()

    sub_agent = SubAgentTool(
        available_tools=[],
        strategy_factory=lambda: child_strategy,
        include_self_in_children=False,
    )

    parent_strategy = ParentStrategy()
    parent_agent = AgentTool(
        tools=[sub_agent],
        strategy=parent_strategy,
    )

    result = await parent_agent.ainvoke(
        AgentToolInput(objective="parent goal", max_iterations=10)
    )

    assert result.success
    assert result.plan_state is not None

    # Parent's plan_state has only the parent task.
    parent_objectives = {t.objective for t in result.plan_state.tasks}
    assert parent_objectives == {parent_marker_objective}
    assert child_marker_objective not in parent_objectives

    # Child saw a plan_state where the parent's task did NOT appear.
    assert child_strategy.observed_plan_states, "child strategy not invoked"
    for observed in child_strategy.observed_plan_states:
        assert observed is not None
        # Child's PlanState is distinct from parent's
        assert observed is not result.plan_state
        # Child's objective is the sub-objective, not the parent's
        assert observed.objective == "child sub-objective"
        observed_objectives = {t.objective for t in observed.tasks}
        assert parent_marker_objective not in observed_objectives


@pytest.mark.asyncio
async def test_subagent_child_plan_revision_starts_at_zero():
    """Child agents start with revision_count=0 even if the parent revised."""
    initialize_recursion_context(max_depth=3)

    captured_initial_revisions: list[int] = []

    class ParentStrategy(PlanningStrategy):
        def __init__(self):
            self.calls = 0

        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            self.calls += 1
            if self.calls == 1:
                tasks = [TaskState(id=1, objective="p1", status="pending")]
                update_input = PlanStateUpdateInput(tasks=tasks)
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "wp1",
                        )
                    ]
                )
            if self.calls == 2:
                # Revise the plan again — parent revision_count becomes 2
                tasks = [
                    TaskState(id=1, objective="p1", status="completed")
                ]
                update_input = PlanStateUpdateInput(tasks=tasks)
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "planstate_update",
                            update_input.model_dump(),
                            update_input,
                            "wp2",
                        )
                    ]
                )
            if self.calls == 3:
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "delegate_subtask",
                            {"sub_objective": "child"},
                            SubAgentInput(sub_objective="child"),
                            "d1",
                        )
                    ]
                )
            return StrategyOutput(
                finished=True, success=True, result="parent done"
            )

    class ChildStrategy(PlanningStrategy):
        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            if plan_state is not None:
                captured_initial_revisions.append(plan_state.revision_count)
            return StrategyOutput(
                finished=True, success=True, result="child done"
            )

    sub_agent = SubAgentTool(
        available_tools=[],
        strategy_factory=lambda: ChildStrategy(),
        include_self_in_children=False,
    )

    parent_agent = AgentTool(
        tools=[sub_agent],
        strategy=ParentStrategy(),
    )

    result = await parent_agent.ainvoke(
        AgentToolInput(objective="parent goal", max_iterations=10)
    )
    assert result.success
    # Parent did planstate_update twice ⇒ parent revision_count == 2
    assert result.plan_state.revision_count == 2

    # Child observed revision_count == 0 (fresh PlanState), proving isolation
    assert captured_initial_revisions == [0]


@pytest.mark.asyncio
async def test_subagent_with_plan_status_termination():
    """Child can terminate via plan_status='completed' independently of parent."""
    initialize_recursion_context(max_depth=3)

    class ParentStrategy(PlanningStrategy):
        def __init__(self):
            self.calls = 0

        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            self.calls += 1
            if self.calls == 1:
                return StrategyOutput(
                    tool_calls=[
                        _tc(
                            "delegate_subtask",
                            {"sub_objective": "child task"},
                            SubAgentInput(sub_objective="child task"),
                            "d1",
                        )
                    ]
                )
            return StrategyOutput(
                finished=True, success=True, result="parent done"
            )

    class ChildStrategy(PlanningStrategy):
        def __init__(self):
            self.calls = 0

        async def plan(
            self, messages, tools, parallel_tool_calls=True, plan_state=None
        ):
            self.calls += 1
            # Child terminates on first call by setting plan_status=completed
            tasks = [
                TaskState(id=1, objective="t", status="completed")
            ]
            update_input = PlanStateUpdateInput(
                tasks=tasks, plan_status="completed"
            )
            return StrategyOutput(
                tool_calls=[
                    _tc(
                        "planstate_update",
                        update_input.model_dump(),
                        update_input,
                        "u1",
                    )
                ]
            )

    sub_agent = SubAgentTool(
        available_tools=[],
        strategy_factory=lambda: ChildStrategy(),
        include_self_in_children=False,
    )

    parent_agent = AgentTool(
        tools=[sub_agent],
        strategy=ParentStrategy(),
    )

    result = await parent_agent.ainvoke(
        AgentToolInput(objective="parent goal", max_iterations=10)
    )

    assert result.success
    # Parent's plan was never updated — it started in draft and stayed there
    assert result.plan_state.status == "draft"
    assert result.plan_state.tasks == []
