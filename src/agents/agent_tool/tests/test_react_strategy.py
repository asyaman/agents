"""Tests for ReactStrategy."""

from unittest.mock import MagicMock

import pytest

from agents.agent_tool.meta_tool_plan_state_update import (
    PlanStateUpdate,
    PlanStateUpdateInput,
)
from agents.agent_tool.plan_state import PlanState, TaskState
from agents.agent_tool.react_strategy import ReactStrategy
from agents.agent_tool.tests.common_fixtures import SearchTool
from agents.llm_core.llm_client import TextResponse, ToolCall, ToolCallResponse


class TestReactStrategy:
    """Tests for ReactStrategy."""

    def test_init_defaults(self, mock_llm_client: MagicMock):
        strategy = ReactStrategy(action_client=mock_llm_client)
        assert strategy.reasoning_prompt is None
        assert strategy.action_prompt is None
        # reasoning_client defaults to action_client
        assert strategy.reasoning_client is mock_llm_client

    def test_init_custom_prompts(self, mock_llm_client: MagicMock):
        strategy = ReactStrategy(
            action_client=mock_llm_client,
            reasoning_prompt="Think carefully",
            action_prompt="Now act",
        )
        assert strategy.reasoning_prompt == "Think carefully"
        assert strategy.action_prompt == "Now act"

    @pytest.mark.asyncio
    async def test_plan_two_phase_execution(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Test that React does reasoning then action."""
        strategy = ReactStrategy(
            action_client=mock_llm_client,
            reasoning_prompt="Think step by step",
            action_prompt="Select a tool",
        )

        # First call: reasoning (text response)
        # Second call: action (tool call)
        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="I should search for information first."),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="test-id-1", tool_name="search", arguments={"query": "test"}
                    )
                ]
            ),
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Find information"}],
            tools=[search_tool],
        )

        # Verify two LLM calls were made
        assert mock_llm_client.agenerate.call_count == 2

        # Reasoning text is captured in messages
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "I should search for information first."

        # Tool calls passed through (loop continues since action tool, not finish)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_plan_with_finish_passes_through(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Strategy passes finish tool through; AgentTool detects + terminates.
        Reasoning text is still captured in output messages."""
        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="The task is complete."),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="test-id-1",
                        tool_name="finish",
                        arguments={"result": "Done!", "success": True},
                    )
                ]
            ),
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Complete task"}],
            tools=[search_tool],
        )

        # finish stays in tool_calls; strategy doesn't extract args.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "finish"
        assert result.tool_calls[0].arguments["result"] == "Done!"
        # Reasoning still captured.
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "The task is complete."

    @pytest.mark.asyncio
    async def test_plan_no_tool_calls_signals_unsuccessful_terminate(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """Empty tool_calls = strategy-internal terminate (success=False)."""
        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="I cannot proceed with this task."),
            ToolCallResponse(tool_calls=[]),
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "Do something"}],
            tools=[search_tool],
        )

        assert result.tool_calls == []
        assert result.success is False
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "I cannot proceed with this task."


class TestReactStrategyAutoTranslate:
    """Tests for the auto-translate-plan stage that runs an extra LLM call
    between reasoning and action to translate the free-text reasoning into
    a structured planstate_update."""

    @pytest.mark.asyncio
    async def test_translator_runs_when_enabled_and_tool_present(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """When auto_translate_plan=True AND planstate_update is among tools,
        the strategy makes 3 LLM calls (reason → translate → act) and the
        translator's planstate_update mutates plan_state."""
        plan_state = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="search step", status="pending"),
            ],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        # Translator's emitted planstate_update: marks task 1 in_progress
        translator_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="search step", status="in_progress")]
        )

        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            # Phase 1: reasoning text
            TextResponse(content="I'll search for the answer next."),
            # Phase 1.5: translator emits planstate_update
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="t1",
                        tool_name="planstate_update",
                        arguments=translator_input.model_dump(),
                        parsed=translator_input,
                    )
                ]
            ),
            # Phase 2: action
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="a1", tool_name="search", arguments={"query": "x"}
                    )
                ]
            ),
        ]

        result = await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # Three LLM calls: reasoning + translator + action
        assert mock_llm_client.agenerate.call_count == 3
        # plan_state was mutated by translator
        assert plan_state.tasks[0].status == "in_progress"
        assert plan_state.revision_count == 1
        # Action phase still produced its tool call
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_translator_skipped_when_disabled(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """auto_translate_plan=False → only 2 LLM calls, plan_state unchanged."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="pending")],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        strategy = ReactStrategy(
            action_client=mock_llm_client, auto_translate_plan=False
        )

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # Only 2 calls — no translator stage
        assert mock_llm_client.agenerate.call_count == 2
        # plan_state untouched
        assert plan_state.tasks[0].status == "pending"
        assert plan_state.revision_count == 0

    @pytest.mark.asyncio
    async def test_translator_skipped_when_planstate_update_tool_absent(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """If the planstate_update tool isn't in `tools`, the translator stage
        is silently skipped (e.g., AgentTool was constructed with
        include_planstate_update_tool=False)."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="pending")],
        )

        # auto_translate_plan defaults to True; tools list has no PlanStateUpdate
        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool],  # no PlanStateUpdate
            plan_state=plan_state,
        )

        assert mock_llm_client.agenerate.call_count == 2
        assert plan_state.tasks[0].status == "pending"
        assert plan_state.revision_count == 0

    @pytest.mark.asyncio
    async def test_translator_no_change_emitted_is_a_noop(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """When the translator emits no tool calls (because the plan is
        already correct), plan_state is unchanged and the action phase
        proceeds normally."""
        plan_state = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="x", status="in_progress"),
            ],
            status="active",
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="continuing the existing plan."),
            # Translator: no change needed
            ToolCallResponse(tool_calls=[]),
            # Action
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        assert mock_llm_client.agenerate.call_count == 3
        # No mutation
        assert plan_state.tasks[0].status == "in_progress"
        assert plan_state.revision_count == 0

    @pytest.mark.asyncio
    async def test_action_phase_excludes_planstate_update_when_translator_ran(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """When the translator is active, the action phase's tool list must
        NOT include planstate_update — otherwise the action LLM tends to emit
        redundant planstate_update calls instead of doing real work."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="pending")],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        translator_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="x", status="in_progress")]
        )

        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="t1",
                        tool_name="planstate_update",
                        arguments=translator_input.model_dump(),
                        parsed=translator_input,
                    )
                ]
            ),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # Inspect the action phase (3rd LLM call): its `tools` argument must
        # contain `search` but NOT `planstate_update`.
        action_call = mock_llm_client.agenerate.call_args_list[2]
        action_tool_names = {
            tool.name.upper() for tool in action_call.kwargs["tools"]
        }
        assert "SEARCH" in action_tool_names
        assert "PLANSTATE_UPDATE" not in action_tool_names

    @pytest.mark.asyncio
    async def test_action_phase_keeps_planstate_update_when_translator_disabled(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """With auto_translate_plan=False the model must still see
        planstate_update in its action tools (since nothing else manages
        plan structure)."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="pending")],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        strategy = ReactStrategy(
            action_client=mock_llm_client, auto_translate_plan=False
        )

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # Action call (2nd) — no translator step, so planstate_update stays
        # available so the model can manage plan_state itself.
        action_call = mock_llm_client.agenerate.call_args_list[1]
        action_tool_names = {
            tool.name.upper() for tool in action_call.kwargs["tools"]
        }
        assert "PLANSTATE_UPDATE" in action_tool_names

    @pytest.mark.asyncio
    async def test_translator_retries_on_parse_error_and_succeeds(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """When the translator's first call has a parse_error, the strategy
        feeds the error back and retries; the second attempt succeeds."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="pending")],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        good_input = PlanStateUpdateInput(
            tasks=[TaskState(id=1, objective="x", status="in_progress")]
        )

        strategy = ReactStrategy(
            action_client=mock_llm_client, plan_translator_max_retries=2
        )

        # Reasoning, translator-fail (parse_error), translator-retry-success, action
        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="t1",
                        tool_name="planstate_update",
                        arguments={"bad": "data"},
                        parsed=None,
                        parse_error="tasks.0.result: invalid",
                    )
                ]
            ),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="t2",
                        tool_name="planstate_update",
                        arguments=good_input.model_dump(),
                        parsed=good_input,
                    )
                ]
            ),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # 4 LLM calls: reasoning, translator-fail, translator-retry, action
        assert mock_llm_client.agenerate.call_count == 4
        # Retry succeeded → plan_state was mutated
        assert plan_state.tasks[0].status == "in_progress"
        assert plan_state.revision_count == 1

        # The second translator call must include the failed tool result
        # so the model can see and correct its error.
        retry_call = mock_llm_client.agenerate.call_args_list[2]
        retry_messages = retry_call.kwargs["messages"]
        tool_results = [m for m in retry_messages if m.get("role") == "tool"]
        assert tool_results, "Retry call missing the tool error message"
        assert "invalid" in tool_results[-1]["content"]

    @pytest.mark.asyncio
    async def test_translator_gives_up_after_max_retries(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """When every attempt fails (parse_error each time), the strategy
        bails after `plan_translator_max_retries` retries and proceeds to
        action without mutating plan_state."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="pending")],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        strategy = ReactStrategy(
            action_client=mock_llm_client, plan_translator_max_retries=1
        )

        bad = ToolCallResponse(
            tool_calls=[
                ToolCall(
                    id="t",
                    tool_name="planstate_update",
                    arguments={"bad": "data"},
                    parsed=None,
                    parse_error="bad args",
                )
            ]
        )

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            bad,  # first translator attempt fails
            bad,  # retry also fails (max_retries=1 → 2 total attempts)
            # action still runs even though translation failed
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # 4 calls: reasoning + 2 translator attempts + action
        assert mock_llm_client.agenerate.call_count == 4
        # Plan state untouched
        assert plan_state.tasks[0].status == "pending"
        assert plan_state.revision_count == 0

    @pytest.mark.asyncio
    async def test_translator_does_not_retry_on_no_change_emitted(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """An empty tool_calls response is a legitimate no-op signal — not
        a recoverable error. The strategy must NOT retry in that case."""
        plan_state = PlanState(
            objective="goal",
            tasks=[TaskState(id=1, objective="x", status="in_progress")],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        strategy = ReactStrategy(
            action_client=mock_llm_client, plan_translator_max_retries=3
        )

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(tool_calls=[]),  # translator: no change
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # Only 3 calls: reasoning + ONE translator (no retry on no-op) + action
        assert mock_llm_client.agenerate.call_count == 3

    @pytest.mark.asyncio
    async def test_action_phase_sees_translated_plan(
        self, mock_llm_client: MagicMock, search_tool: SearchTool
    ):
        """After the translator mutates plan_state, the action phase prompt
        must include the updated plan block (not the pre-translation one)."""
        plan_state = PlanState(
            objective="goal",
            tasks=[
                TaskState(id=1, objective="search step", status="pending"),
            ],
        )
        ps_tool = PlanStateUpdate(plan_state=plan_state)

        translator_input = PlanStateUpdateInput(
            tasks=[
                TaskState(id=1, objective="search step", status="in_progress"),
            ]
        )

        strategy = ReactStrategy(action_client=mock_llm_client)

        mock_llm_client.agenerate.side_effect = [
            TextResponse(content="reasoning"),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="t1",
                        tool_name="planstate_update",
                        arguments=translator_input.model_dump(),
                        parsed=translator_input,
                    )
                ]
            ),
            ToolCallResponse(
                tool_calls=[
                    ToolCall(id="a1", tool_name="search", arguments={"query": "x"})
                ]
            ),
        ]

        await strategy.plan(
            messages=[{"role": "user", "content": "go"}],
            tools=[search_tool, ps_tool],
            plan_state=plan_state,
        )

        # The 3rd call is the action phase. Inspect its messages for the
        # translated plan block (task 1 IN_PROGRESS, not PENDING).
        action_call_args = mock_llm_client.agenerate.call_args_list[2]
        action_messages = action_call_args.kwargs["messages"]
        plan_blocks = [
            m["content"]
            for m in action_messages
            if m.get("role") == "system"
            and "Current Plan State" in m.get("content", "")
        ]
        assert plan_blocks, "Action phase missing plan_state block"
        assert "IN_PROGRESS" in plan_blocks[-1]
        assert "PENDING" not in plan_blocks[-1].split("[1]")[1].split("\n")[0]
