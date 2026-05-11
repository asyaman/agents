"""
AgentTool - General-purpose agent executor with pluggable planning strategies.

AgentTool handles:
- Message history management (list[ChatCompletionMessageParam])
- Tool execution
- Iteration control

The planning strategy handles:
- LLM interaction
- Reasoning (if applicable)
- Tool selection
"""

import asyncio
import json
import typing as t

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.agent_tool.base_strategy import PlanningStrategy
from agents.agent_tool.meta_tool_finish import (
    FinishInput,
    FinishOutput,
    create_finish_tool,
)
from agents.agent_tool.meta_tool_plan_state_update import PlanStateUpdate
from agents.agent_tool.plan_state import PlanState, TaskState
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import ToolCall
from agents.tools_core.base_tool import BaseTool

# Re-exports for backwards-compatible imports. New code should import these
# from their dedicated modules; these aliases keep existing import sites
# (tests, sub-agent fixtures) working without churn.
__all__ = [
    "AgentTool",
    "AgentToolInput",
    "AgentToolOutput",
    "FinishInput",
    "FinishOutput",
    "create_finish_tool",
]

# Tool names treated by the framework as "meta" — they manage plan state or
# terminate the loop. The meta-emission policy in `_execute_tool_calls`
# requires each meta tool to be emitted ALONE in its own iteration. They are
# never co-emitted with each other or with action tools.
_FINISH_TOOL_NAME: str = "FINISH"
_PLANSTATE_UPDATE_TOOL_NAME: str = "PLANSTATE_UPDATE"
_META_TOOL_NAMES: frozenset[str] = frozenset(
    {_PLANSTATE_UPDATE_TOOL_NAME, _FINISH_TOOL_NAME}
)

# Load templates
_templates = get_agent_tool_template_module("agent_tool.jinja")


class AgentToolInput(BaseModel):
    """Input for the AgentTool."""

    objective: str = Field(description="The task/objective to accomplish.")
    context: str | None = Field(
        default=None,
        description="Additional context to help accomplish the objective.",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations before stopping.",
    )


class AgentToolOutput(BaseModel):
    """Output from the AgentTool."""

    result: str = Field(description="The result of the task.")
    success: bool = Field(description="Whether the task was completed successfully.")
    iterations_used: int = Field(description="Number of iterations used.")
    # Note: Using dict instead of ChatCompletionMessageParam to avoid Pydantic v2
    # serialization issues with OpenAI's complex Union types
    messages: list[dict[str, t.Any]] = Field(
        default_factory=list,
        description="Full conversation history (for debugging/inspection).",
    )
    plan_state: PlanState | None = Field(
        default=None,
        description=(
            "Final plan state after the run. Useful for sub-agent return paths "
            "and post-run inspection."
        ),
    )


class AgentTool(BaseTool[AgentToolInput, AgentToolOutput]):
    """
    General-purpose agent tool with pluggable planning strategy.

    The AgentTool provides an agentic loop where:
    1. Strategy decides what to do (reasoning + tool selection)
    2. AgentTool executes the tools
    3. Results are added to message history
    4. Repeat until task is complete or max iterations

    Usage:
        # Simple direct execution
        agent = AgentTool(
            tools=[search_tool, calculator_tool],
            llm_client=client,
            strategy=DirectStrategy(),
        )

        # With reasoning (React pattern)
        agent = AgentTool(
            tools=[search_tool, calculator_tool],
            llm_client=client,
            strategy=ReactStrategy(),
        )

        # Execute
        result = await agent.ainvoke(AgentToolInput(
            objective="Find the population of France and calculate 10% of it"
        ))
    """

    _name = "agent"
    description = "A general-purpose agent that can use tools to accomplish objectives."
    _input = AgentToolInput
    _output = AgentToolOutput

    def __init__(
        self,
        tools: list[BaseTool[t.Any, t.Any]],
        strategy: PlanningStrategy,
        system_prompt: str | None = None,
        include_planstate_update_tool: bool = True,
        parallel_tool_calls: bool = True,
        guidance_messages: list[str] | None = None,
    ) -> None:
        """
        Initialize AgentTool.

        The `finish` meta tool is always included — it's the universal
        termination signal that every agent needs. The `planstate_update`
        meta tool is optional via `include_planstate_update_tool`.

        Args:
            tools: List of tools available to the agent
            strategy: Planning strategy (owns its LLM client and model config)
            system_prompt: Custom system prompt (uses template if None)
            include_planstate_update_tool: Whether to auto-add the planstate_update
                tool that lets the model mutate the per-run PlanState (default
                True). The tool is constructed per ainvoke call so it closes
                over that run's PlanState. Set False for plan-state-agnostic
                strategies that don't want the model to see the meta tool.
            parallel_tool_calls: Allow parallel tool calls (LLM and execution)
            guidance_messages: Additional system messages to inject after the main
                system prompt (e.g., sub-agent usage guidance)
        """
        super().__init__()
        self.strategy = strategy
        self._system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls
        self.guidance_messages = guidance_messages or []
        self._include_planstate_update_tool = include_planstate_update_tool

        # `finish` is always added; `planstate_update` is added per-run in
        # ainvoke because it needs the run's PlanState as a closure.
        self.tools = list(tools)
        self.tools.append(create_finish_tool())

    def _get_system_prompt(self, tools: list[BaseTool[t.Any, t.Any]]) -> str:
        """Get the system prompt from template or custom."""
        if self._system_prompt:
            return self._system_prompt
        return _templates.system_prompt(tools=tools)

    def _get_task_prompt(self, objective: str, context: str | None) -> str:
        """Get the task prompt from template."""
        return _templates.task_prompt(objective=objective, context=context)

    def invoke(self, input: AgentToolInput) -> AgentToolOutput:
        """Sync execution - wraps async implementation."""
        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: AgentToolInput) -> AgentToolOutput:
        """Execute the agent task asynchronously."""
        validated = self._validate_input(input)

        # Per-run PlanState. Created here (not in __init__) so each ainvoke call
        # gets isolated state. Sub-agents that build a fresh AgentTool
        # automatically get their own plan.
        plan_state = PlanState(objective=validated.objective)

        # Per-run tool list. PlanStateUpdate is added here so it closes over
        # this run's PlanState.
        run_tools: list[BaseTool[t.Any, t.Any]] = list(self.tools)
        if self._include_planstate_update_tool:
            run_tools.append(PlanStateUpdate(plan_state))

        logger.info(
            "Starting agent task | objective={} | max_iterations={} | strategy={}",
            (
                validated.objective[:50] + "..."
                if len(validated.objective) > 50
                else validated.objective
            ),
            validated.max_iterations,
            type(self.strategy).__name__,
        )

        result = await self._agent_loop(
            objective=validated.objective,
            context=validated.context,
            max_iterations=validated.max_iterations,
            plan_state=plan_state,
            run_tools=run_tools,
        )

        if result.success:
            logger.success(
                "Task completed | iterations={}",
                result.iterations_used,
            )
        else:
            logger.warning(
                "Task failed | iterations={} | result={}",
                result.iterations_used,
                result.result[:100] if result.result else "No result",
            )

        return result

    async def _agent_loop(
        self,
        objective: str,
        context: str | None,
        max_iterations: int,
        plan_state: PlanState,
        run_tools: list[BaseTool[t.Any, t.Any]],
    ) -> AgentToolOutput:
        """Run the agent loop until task completion or max iterations.

        Three coordinated termination paths:
          1. Strategy-signaled finish (StrategyOutput.finished == True),
             checked BEFORE tool execution.
          2. Plan-status-driven finish (plan_state.status terminal), checked
             AFTER tool execution.
          3. Max iterations exhausted, returns success=False.
        """

        # Initialize messages with system prompt and task
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._get_system_prompt(run_tools)},
        ]

        # Inject guidance messages (e.g., sub-agent usage hints)
        for guidance in self.guidance_messages:
            messages.append({"role": "system", "content": guidance})

        messages.append(
            {"role": "user", "content": self._get_task_prompt(objective, context)}
        )

        # Use uppercase keys for case-insensitive lookup (tool.name is normalized
        # to uppercase)
        tool_map = {tool.name.upper(): tool for tool in run_tools}

        for iteration in range(max_iterations):
            logger.debug(
                "Agent iteration {}/{} | messages={} | plan_revision={} | "
                "plan_status={}",
                iteration + 1,
                max_iterations,
                len(messages),
                plan_state.revision_count,
                plan_state.status,
            )

            # Strategy receives plan_state by reference (read-only convention),
            # messages (read-only history), tools, and returns a per-turn delta.
            strategy_output = await self.strategy.plan(
                messages=messages,
                tools=run_tools,
                parallel_tool_calls=self.parallel_tool_calls,
                plan_state=plan_state,
            )

            # Apply the per-turn delta (e.g., reasoning text) to durable history
            messages.extend(strategy_output.messages)

            # Termination path 1a: strategy emitted no tool_calls
            # (strategy-internal terminate). AgentTool reads success/result
            # directly from StrategyOutput. This typically means the LLM
            # didn't call any tool and the strategy decided to stop.
            if not strategy_output.tool_calls:
                return AgentToolOutput(
                    result=strategy_output.result or _summarize_plan_result(plan_state),
                    success=strategy_output.success,
                    iterations_used=iteration + 1,
                    messages=t.cast(list[dict[str, t.Any]], messages),
                    plan_state=plan_state,
                )

            # Execute tool_calls. Each meta tool (planstate_update, finish)
            # must be emitted ALONE in its own iteration; the policy guard in
            # _execute_tool_calls rejects mixed turns and returns True to
            # signal that nothing actually executed (so we must NOT honor any
            # finish/plan_status signals derived from the rejected calls).
            policy_violated = await self._execute_tool_calls(
                tool_calls=strategy_output.tool_calls,
                tool_map=tool_map,
                messages=messages,
                plan_state=plan_state,
                parallel=self.parallel_tool_calls,
            )

            # Termination path 1b: model called the `finish` tool and it
            # actually ran (no policy violation). AgentTool extracts the
            # result/success from finish's arguments. We scan the EXECUTED
            # tool_calls — not strategy_output's intent — so a finish
            # rejected by the policy guard never terminates the loop.
            if not policy_violated:
                finish_tc = next(
                    (
                        tc for tc in strategy_output.tool_calls
                        if tc.tool_name.upper() == _FINISH_TOOL_NAME
                    ),
                    None,
                )
                if finish_tc is not None:
                    success_flag = finish_tc.arguments.get("success", True)
                    # Sync plan_state.status to mirror the FINISH outcome.
                    # The agent has terminated; leaving plan_status="active"
                    # in the returned PlanState would be a stale read of an
                    # in-flight run that no longer is. Task statuses are
                    # left untouched — the model owns those.
                    plan_state.status = "completed" if success_flag else "failed"
                    return AgentToolOutput(
                        result=finish_tc.arguments.get(
                            "result", "Task completed"
                        ),
                        success=success_flag,
                        iterations_used=iteration + 1,
                        messages=t.cast(list[dict[str, t.Any]], messages),
                        plan_state=plan_state,
                    )

            # Termination path 2: model set plan_status to a terminal value
            # via planstate_update.
            if plan_state.status in ("completed", "failed"):
                return AgentToolOutput(
                    result=_summarize_plan_result(plan_state),
                    success=plan_state.status == "completed",
                    iterations_used=iteration + 1,
                    messages=t.cast(list[dict[str, t.Any]], messages),
                    plan_state=plan_state,
                )

        # Termination path 3: max iterations reached.
        logger.warning(
            "Max iterations reached | max={}",
            max_iterations,
        )
        return AgentToolOutput(
            result="Max iterations reached without completing the task",
            success=False,
            iterations_used=max_iterations,
            messages=t.cast(list[dict[str, t.Any]], messages),
            plan_state=plan_state,
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        tool_map: dict[str, BaseTool[t.Any, t.Any]],
        messages: list[ChatCompletionMessageParam],
        plan_state: PlanState,
        parallel: bool = True,
    ) -> bool:
        """Execute tool calls in meta-first order, with auto-status-update.

        Returns:
            True if a policy violation was detected and tool execution was
            skipped (the model is expected to retry). False if all tools were
            allowed to run normally.

        Execution order within a single turn:
          Phase A — META TOOLS (planstate_update, finish): run first,
            sequentially in the order the model emitted them. They mutate
            plan_state immediately. Running them first lets the model emit
            `[planstate_update(mark task N in_progress), action_tool_for_N]`
            in the same turn — by the time the action runs, plan_state
            already reflects the new in_progress task.
          Phase B — SNAPSHOT: capture the in_progress task AFTER meta tools
            have mutated plan_state. This snapshot is what auto-status-update
            will target.
          Phase C — NON-META TOOLS: run in parallel (or sequentially if
            `parallel=False`). These are the actual work tools.
          Phase D — AUTO-STATUS-UPDATE: if exactly ONE non-meta tool ran AND
            the snapshot was non-empty, mark the snapshot's task
            completed/failed and store the tool's output as its result. For
            multi-action turns the auto-update is skipped — the model must
            record statuses via planstate_update on the following turn.

        Tool result messages are appended to `messages` in the same order the
        tools were emitted by the model (matching the assistant's tool_calls
        list), so the OpenAI API's tool_call_id pairing remains intact.

        Args:
            tool_calls: List of tool calls to execute
            tool_map: Mapping of tool names to tool instances
            messages: Message history to append to
            plan_state: PlanState to auto-update from tool results
            parallel: If True, execute non-meta tools in parallel; otherwise sequential
        """

        # Single assistant message with ALL tool calls (preserves model's order).
        # When `parsed` is None (the model emitted invalid arguments and
        # _parse_tool_calls flagged a parse_error), fall back to the raw
        # `arguments` dict so the assistant echo still goes back to the LLM.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": (
                                tc.parsed.model_dump_json()
                                if tc.parsed is not None
                                else json.dumps(tc.arguments)
                            ),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # POLICY GUARD: meta tools (planstate_update, finish) are atomic.
        # Each meta tool must be emitted ALONE in its own iteration to prevent
        # plan_state from diverging from reality (e.g., planstate_update fails
        # but action runs anyway → plan never recorded the new task but the
        # action's side effect already happened).
        # Forbidden combinations: multiple meta tools in one turn, OR any meta
        # tool mixed with non-meta tools. When violated, do NOT execute any of
        # the tools — return a uniform policy-violation error for each so the
        # model sees the failure and re-emits each meta tool in its own turn.
        meta_calls = [
            tc for tc in tool_calls
            if tc.tool_name.upper() in _META_TOOL_NAMES
        ]
        non_meta_calls = [
            tc for tc in tool_calls
            if tc.tool_name.upper() not in _META_TOOL_NAMES
        ]
        policy_violated = len(meta_calls) > 1 or (meta_calls and non_meta_calls)

        if policy_violated:
            meta_names = sorted({tc.tool_name for tc in meta_calls})
            other_names = sorted({tc.tool_name for tc in non_meta_calls})
            error_payload = json.dumps({
                "error": (
                    "Tool emission policy violation: meta tools "
                    f"({', '.join(sorted(_META_TOOL_NAMES)).lower()}) must be "
                    "emitted ALONE — one per turn, never mixed with each other "
                    "or with action tools. "
                    f"This turn emitted meta={meta_names} and "
                    f"non-meta={other_names}. "
                    "No tool was executed. Re-emit each meta tool in its own "
                    "iteration."
                )
            })
            for tc in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_payload,
                    }
                )
            logger.warning(
                "Policy violation: meta tools must be emitted alone | "
                "meta={} | non_meta={}",
                meta_names,
                other_names,
            )
            return True

        async def execute_single_tool(
            tool_call: ToolCall,
        ) -> tuple[str, str, bool]:
            """Execute a single tool. Returns (tool_call_id, result, is_error)."""
            tool_name = tool_call.tool_name

            # If _parse_tool_calls flagged a validation error on this call,
            # don't try to execute. Return the validation error as the tool
            # result so the model sees it on the next iteration and can fix
            # its arguments.
            if tool_call.parse_error is not None:
                tool_result = json.dumps({
                    "error": (
                        f"Invalid arguments for tool '{tool_name}': "
                        f"{tool_call.parse_error}"
                    )
                })
                logger.warning(
                    "Tool args validation failed | tool={} | error={}",
                    tool_name,
                    tool_call.parse_error[:200],
                )
                return (tool_call.id, tool_result, True)

            tool_input = tool_call.parsed

            logger.info(
                "Tool call | tool={} | args={}",
                tool_name,
                tool_input.model_dump_json()[:100],
            )

            tool = tool_map.get(tool_name.upper())
            if tool is None:
                tool_result = json.dumps({"error": f"Unknown tool '{tool_name}'"})
                logger.error("Unknown tool: {}", tool_name)
                return (tool_call.id, tool_result, True)
            try:
                result = await tool.acall(tool_input)
                tool_result = result.model_dump_json()
                logger.debug("Tool result: {}", tool_result[:200])
                return (tool_call.id, tool_result, False)
            except Exception as e:
                tool_result = json.dumps({"error": f"Error executing {tool_name}: {e}"})
                logger.error("Tool execution error: {}", e)
                return (tool_call.id, tool_result, True)

        # `meta_calls` and `non_meta_calls` were partitioned above by the
        # policy guard. After the guard, len(meta_calls) <= 1 and at most one
        # of (meta_calls, non_meta_calls) is non-empty.

        # PHASE A — meta tools, sequentially, in emission order.
        meta_results: list[tuple[str, str, bool]] = []
        for tc in meta_calls:
            meta_results.append(await execute_single_tool(tc))

        # PHASE B — snapshot AFTER meta tools have mutated plan_state.
        # Capture the FULL set of in_progress task ids so Phase D can pair
        # multiple parallel tool results to multiple in_progress tasks
        # (parallel-batch fan-out).
        snapshot_in_progress_ids: list[int] = [
            t.id for t in plan_state.in_progress_tasks()
        ]

        # PHASE C — non-meta tools (parallel or sequential)
        non_meta_results: list[tuple[str, str, bool]]
        if parallel:
            non_meta_results = list(
                await asyncio.gather(
                    *[execute_single_tool(tc) for tc in non_meta_calls]
                )
            )
        else:
            non_meta_results = [await execute_single_tool(tc) for tc in non_meta_calls]

        # Append tool result messages in the original emission order so
        # tool_call_id pairing stays intact for the OpenAI API.
        results_by_id = {r[0]: r for r in (*meta_results, *non_meta_results)}
        for tc in tool_calls:
            tool_call_id, tool_result, _is_error = results_by_id[tc.id]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result,
                }
            )

        # PHASE D — auto-status-update.
        # Single-tool turn: pair the one result to the one snapshot in_progress task.
        # Parallel-batch turn (N tools, N in_progress): strict bijective pairing
        # on (inputs) — only when cardinality matches AND every tool finds a
        # unique matching in_progress task. Otherwise skip (current legacy
        # behavior) so the model can record outcomes via planstate_update.
        if non_meta_results:
            _auto_status_update(
                plan_state=plan_state,
                non_meta_calls=non_meta_calls,
                non_meta_results=non_meta_results,
                snapshot_in_progress_ids=snapshot_in_progress_ids,
            )

        return False


def _truncate(s: str, n: int) -> str:
    """Truncate string to at most n chars, adding ellipsis marker if cut."""
    return s if len(s) <= n else s[:n] + "...[truncated]"


def _auto_status_update(
    plan_state: PlanState,
    non_meta_calls: list[ToolCall],
    non_meta_results: list[tuple[str, str, bool]],
    snapshot_in_progress_ids: list[int],
) -> None:
    """Phase D: write tool outcomes back into the matching plan_state tasks.

    Plan-state is the driver: only tool results that have a UNIQUELY
    matching in_progress task (by exact `inputs` equality) update plan_state.
    Anything else is left to the message history.

    Per-pair semantics (applied uniformly to single-tool and parallel turns):

    - For each tool result, look for in_progress snapshot tasks whose
      `inputs` exactly equal the call's args.
    - **Exactly one match** → mark that task `completed`/`failed` and copy
      the (truncated) tool result into `task.result`.
    - **Zero matches** → skip this tool (its result lives in `messages`
      only). The model can choose whether to register it later.
    - **More than one match** → skip this tool (ambiguous); avoid silent
      mispair. The model will see the duplicate-inputs plan and can
      resolve via planstate_update.
    - **Tasks without a matching tool** stay `in_progress` — the next
      iteration has another shot at dispatching them.

    Single-tool legacy fallback: if exactly one tool ran AND no unique
    inputs-match was found AND exactly one in_progress task exists, pair
    the tool to that task regardless of inputs (preserves the legacy
    behavior where the model didn't bother specifying `inputs` on tasks).

    Logs each pairing decision so skips are diagnosable.
    """
    if not non_meta_results:
        return

    snapshot_tasks_by_id = {
        t.id: t
        for t in plan_state.tasks
        if t.id in snapshot_in_progress_ids and t.status == "in_progress"
    }
    if not snapshot_tasks_by_id:
        return

    calls_by_id = {tc.id: tc for tc in non_meta_calls}
    available = dict(snapshot_tasks_by_id)  # task_id -> task; pop on match
    pairings: list[tuple[TaskState, str, bool]] = []
    is_single_tool = len(non_meta_results) == 1

    for tc_id, result_str, is_error in non_meta_results:
        tc = calls_by_id.get(tc_id)
        if tc is None:
            logger.debug("Auto-pair: missing tool call id={}; skipping", tc_id)
            continue

        # Resolve call args to a dict for inputs equality.
        call_args: dict[str, t.Any] | None
        if isinstance(tc.arguments, dict):
            call_args = tc.arguments
        else:
            try:
                call_args = json.loads(tc.arguments) if tc.arguments else None
            except (json.JSONDecodeError, TypeError):
                call_args = None

        # Try strict inputs match first.
        if isinstance(call_args, dict):
            matches = [
                task
                for task in available.values()
                if task.inputs is not None and task.inputs == call_args
            ]
            if len(matches) == 1:
                task = matches[0]
                del available[task.id]
                pairings.append((task, result_str, is_error))
                continue
            if len(matches) > 1:
                logger.debug(
                    "Auto-pair: ambiguous match (>1 in_progress task with"
                    " inputs={}) for tool {}; skipping this tool",
                    call_args,
                    tc.tool_name,
                )
                continue

        # No inputs match. Single-tool legacy fallback: if exactly one
        # in_progress task remains available, pair to it.
        if is_single_tool and len(available) == 1:
            task = next(iter(available.values()))
            del available[task.id]
            pairings.append((task, result_str, is_error))
            logger.debug(
                "Auto-update: single-tool fallback paired to task {} "
                "(no inputs match)",
                task.id,
            )
            continue

        logger.debug(
            "Auto-pair: no matching task for tool {} (args={}); skipping"
            " (result in messages only)",
            tc.tool_name,
            call_args,
        )

    # Apply matched updates. Unmatched tasks stay in_progress for retry.
    for task, result_str, is_error in pairings:
        new_status = "failed" if is_error else "completed"
        task.status = new_status
        task.result = _truncate(result_str, 1000)
        logger.debug(
            "Auto-update: task {} ({}) → {}",
            task.id,
            task.objective[:50],
            new_status,
        )

    if available:
        logger.debug(
            "Auto-pair: {} task(s) remain in_progress (unmatched, will be"
            " retried next iter): ids={}",
            len(available),
            list(available.keys()),
        )


def _summarize_plan_result(plan_state: PlanState) -> str:
    """Compose a human-readable result summary from the final plan state."""
    if not plan_state.tasks:
        return f"No tasks executed. Plan status: {plan_state.status}."
    completed = [t for t in plan_state.tasks if t.status == "completed"]
    failed = [t for t in plan_state.tasks if t.status == "failed"]
    blocked = [t for t in plan_state.tasks if t.status == "blocked"]
    parts = [
        f"Plan status: {plan_state.status}",
        f"Completed: {len(completed)}/{len(plan_state.tasks)}",
    ]
    if failed:
        parts.append(f"Failed: {len(failed)} ({[t.id for t in failed]})")
    if blocked:
        parts.append(f"Blocked: {len(blocked)} ({[t.id for t in blocked]})")
    return ". ".join(parts)
