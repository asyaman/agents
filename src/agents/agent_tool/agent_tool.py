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
from agents.agent_tool.plan_state import PlanState
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
_META_TOOL_NAMES: frozenset[str] = frozenset({_PLANSTATE_UPDATE_TOOL_NAME, _FINISH_TOOL_NAME})

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
        enable_plan_state: bool = True,
        parallel_tool_calls: bool = True,
        guidance_messages: list[str] | None = None,
    ) -> None:
        """
        Initialize AgentTool.

        The `finish` meta tool is always included — it's the universal
        termination signal that every agent needs. Plan-state semantics
        (the `planstate_update` meta tool) are toggled via
        `enable_plan_state`.

        Args:
            tools: List of tools available to the agent
            strategy: Planning strategy (owns its LLM client and model config)
            system_prompt: Custom system prompt (uses template if None)
            enable_plan_state: Whether to enable plan-state semantics
                (default True). When True, the framework auto-adds the
                `planstate_update` meta tool that lets the model mutate
                the per-run PlanState. The tool is constructed per
                ainvoke call so it closes over that run's PlanState.
                Set False for plan-state-agnostic strategies.
            parallel_tool_calls: Allow parallel tool calls (LLM and execution)
            guidance_messages: Additional system messages to inject after the main
                system prompt (e.g., sub-agent usage guidance)
        """
        super().__init__()
        self.strategy = strategy
        self._system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls
        self.guidance_messages = guidance_messages or []
        self._enable_plan_state = enable_plan_state

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
        if self._enable_plan_state:
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

        Four coordinated termination paths, checked in order each iteration:
          1a. Strategy-internal terminate (`StrategyOutput.tool_calls == []`),
              checked BEFORE tool execution. Typically a protocol violation
              since `finish` is always available — surfaced as
              `success=False` with the strategy's textual fallback as
              `result`.
          1b. `finish` tool was emitted and ran successfully, checked AFTER
              tool execution. BEFORE honoring it, the framework enforces a
              **pre-termination housekeeping check**: if `plan_state` has
              any non-terminal tasks (`pending` / `in_progress` /
              `blocked`), the FINISH is rejected — its tool-result message
              is rewritten to a tool-error payload listing the offending
              task ids, and the loop continues. The model then has a
              chance to call `planstate_update` to mark each task terminal
              and re-emit `finish` on a later iteration. When the check
              passes, `plan_state.status` is synced to `"completed"` /
              `"failed"` from `finish.arguments["success"]` and the loop
              exits.
          2.  Plan-status-driven finish (`plan_state.status` in
              `{"completed", "failed"}` after `planstate_update`), checked
              AFTER tool execution. Trusted as the model's explicit
              terminal signal — no housekeeping check is applied
              because the model has just authored the plan structure
              and is responsible for its consistency.
          3.  Max iterations exhausted — returns `success=False` with a
              hardcoded message; `plan_state.status` is left untouched.

        See the README's "Iteration anatomy → Termination matrix" for
        the per-path detail (when checked, result source, plan_status
        handling).
        """

        # Initialize messages with system prompt and task
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._get_system_prompt(run_tools)},
        ]

        # Inject guidance messages (e.g., sub-agent usage hints)
        for guidance in self.guidance_messages:
            messages.append({"role": "system", "content": guidance})

        messages.append({"role": "user", "content": self._get_task_prompt(objective, context)})

        # Use uppercase keys for case-insensitive lookup (tool.name is normalized
        # to uppercase)
        tool_map = {tool.name.upper(): tool for tool in run_tools}

        for iteration in range(max_iterations):
            logger.debug(
                "Agent iteration {}/{} | messages={} | plan_revision={} | plan_status={}",
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
                        tc
                        for tc in strategy_output.tool_calls
                        if tc.tool_name.upper() == _FINISH_TOOL_NAME
                    ),
                    None,
                )
                if finish_tc is not None:
                    # Pre-termination housekeeping check: plan_state must be
                    # internally consistent (every task in a terminal status)
                    # before we honor `finish`. If non-terminal tasks remain,
                    # reject this finish by rewriting its tool-result message
                    # to a tool-error explaining what needs cleaning up. The
                    # model sees the error on the next iteration and can do
                    # the housekeeping via planstate_update, then re-emit
                    # finish on the iteration after that. Same shape as the
                    # parse-error / unknown-tool surfacing pattern in
                    # `execute_single_tool`.
                    non_terminal_ids = [
                        task.id
                        for task in plan_state.tasks
                        if task.status in ("pending", "in_progress", "blocked")
                    ]
                    if non_terminal_ids:
                        error_msg = (
                            "finish was rejected: plan_state has "
                            f"{len(non_terminal_ids)} non-terminal task(s) "
                            f"(ids={non_terminal_ids}). Call "
                            "planstate_update first to mark each one "
                            "appropriately — `completed` (with result from "
                            "message history if the work was done), "
                            "`cancelled` (if no longer reachable), or "
                            "`failed` (if attempted and unrecoverable). "
                            "Then re-emit `finish` on the next iteration."
                        )
                        # Find the FINISH tool-result message in `messages`
                        # (just appended by Phase A) and replace its content
                        # with the error payload so the model reads it
                        # next turn.
                        for msg in reversed(messages):
                            if (
                                msg.get("role") == "tool"
                                and msg.get("tool_call_id") == finish_tc.id
                            ):
                                msg["content"] = json.dumps({"error": error_msg})
                                break
                        logger.warning(
                            "Pre-termination housekeeping rejected FINISH "
                            "| non_terminal_ids={} | continuing loop",
                            non_terminal_ids,
                        )
                        # Skip termination; fall through to next iteration.
                    else:
                        success_flag = finish_tc.arguments.get("success", True)
                        # Sync plan_state.status to mirror the FINISH
                        # outcome. The agent has terminated; leaving
                        # plan_status="active" in the returned PlanState
                        # would be a stale read of an in-flight run that no
                        # longer is. Task statuses are already terminal at
                        # this point (housekeeping check passed).
                        plan_state.status = "completed" if success_flag else "failed"
                        return AgentToolOutput(
                            result=finish_tc.arguments.get("result", "Task completed"),
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
        """Execute tool calls in meta-first order.

        Returns:
            True if a policy violation was detected and tool execution was
            skipped (the model is expected to retry). False if all tools
            were allowed to run normally.

        Allowed per-iteration combinations (enforced by the policy guard
        below; anything else is rejected, no tool runs):
          - one `planstate_update` alone
          - one `finish` alone
          - one or more non-meta (action) tools
          - one `planstate_update` PLUS one `finish` together
            (Reconcile-and-finish mode — `planstate_update` runs first
            so its mutations are visible before `finish`'s pre-termination
            housekeeping check)

        Cross-iteration rule (also enforced by the policy guard, BEFORE
        any tool runs, by inspecting `messages` history):
          - No back-to-back `planstate_update`-only iterations. If the
            previous turn's emission was a `planstate_update`-only call
            that actually executed (not parse-error / not policy-rejected)
            and this turn is also `planstate_update`-only, the framework
            rejects all tool calls in this turn. The model must instead
            emit an action tool, a `finish`, or a `planstate_update +
            finish` bundle. This prevents replan-only loops that burn
            iterations without making external progress.

        Execution order within a single turn:
          Phase A — META TOOLS run first, sequentially. When both
            `planstate_update` and `finish` are present, `planstate_update`
            is sorted first so it mutates `plan_state` before `finish` is
            evaluated.
          Phase B — NON-META (action) TOOLS run in parallel (or
            sequentially if `parallel=False`). Pydantic schema validation
            happens up-front (parse_error short-circuits execution);
            otherwise the tool's `acall` runs and its result (or error)
            is captured.

        The framework does NOT mutate `plan_state` from action-tool
        results. After every action batch `plan_state` is stale; the
        model reconciles it on the next iteration via `planstate_update`
        (Reconcile-and-plan or Reconcile-and-finish modes — see the
        decision tree in the planner prompt).

        Tool result messages are appended to `messages` in the same
        order the tools were emitted by the model (matching the
        assistant's tool_calls list), so the OpenAI API's tool_call_id
        pairing remains intact.

        Args:
            tool_calls: List of tool calls to execute
            tool_map: Mapping of tool names to tool instances
            messages: Message history to append to
            plan_state: PlanState (passed by reference; mutated only by
                `planstate_update`'s `acall`, never by this method)
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
        # Allowed combinations:
        # - single planstate_update alone
        # - single finish alone
        # - one or more non-meta (action) tools
        # - planstate_update + finish together (Reconcile-and-finish mode);
        #   planstate_update runs FIRST so it can mark remaining tasks
        #   terminal before finish performs the pre-termination housekeeping
        #   check. Exactly one of each, no non-meta tools alongside.
        # Forbidden: any meta + non-meta mixture; two planstate_updates;
        # two finishes; three or more meta tools.
        meta_calls = [tc for tc in tool_calls if tc.tool_name.upper() in _META_TOOL_NAMES]
        non_meta_calls = [tc for tc in tool_calls if tc.tool_name.upper() not in _META_TOOL_NAMES]
        planstate_count = sum(
            1 for tc in meta_calls if tc.tool_name.upper() == _PLANSTATE_UPDATE_TOOL_NAME
        )
        finish_count = sum(1 for tc in meta_calls if tc.tool_name.upper() == _FINISH_TOOL_NAME)
        policy_violated = (meta_calls and non_meta_calls) or planstate_count > 1 or finish_count > 1

        if policy_violated:
            meta_names = sorted({tc.tool_name for tc in meta_calls})
            other_names = sorted({tc.tool_name for tc in non_meta_calls})
            error_payload = json.dumps(
                {
                    "error": (
                        "Tool emission policy violation. Allowed combinations: "
                        "(a) one planstate_update alone, (b) one finish alone, "
                        "(c) one or more action tools, (d) one planstate_update "
                        "PLUS one finish (Reconcile-and-finish). No other "
                        "combinations are valid. "
                        f"This turn emitted meta={meta_names} and "
                        f"non-meta={other_names}. "
                        "No tool was executed."
                    )
                }
            )
            for tc in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_payload,
                    }
                )
            logger.warning(
                "Policy violation: meta tools must be emitted alone "
                "or as planstate_update+finish | meta={} | non_meta={}",
                meta_names,
                other_names,
            )
            return True

        # CROSS-ITERATION GUARD: back-to-back planstate_update-only is
        # rejected BEFORE any tool runs (replan-only loops burn
        # iterations without external progress). Read history to detect
        # whether the previous turn was a planstate_update-only emission
        # that actually executed (parse-error / policy-rejected previous
        # turns don't count — the model never reconciled, so retrying
        # planstate_update now is legitimate). Skip the assistant message
        # we just appended at the top of this method.
        is_planstate_only_now = planstate_count == 1 and finish_count == 0 and not non_meta_calls
        if is_planstate_only_now and _previous_emission_was_planstate_only_success(
            messages, skip_last_assistant=True
        ):
            auto_correct_payload = json.dumps(
                {
                    "error": (
                        "Back-to-back planstate_update-only emissions are not "
                        "allowed. The previous iteration already reconciled the "
                        "plan; this iteration must make external progress. "
                        "Choose ONE of: "
                        "(a) call an action tool to advance a pending/in-progress "
                        "task, "
                        "(b) call `finish` alone if the objective is already met, "
                        "or "
                        "(c) emit `planstate_update + finish` together "
                        "(Reconcile-and-finish) if you only need a final "
                        "housekeeping pass before terminating. "
                        "Do NOT emit another planstate_update by itself."
                    )
                }
            )
            for tc in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": auto_correct_payload,
                    }
                )
            logger.warning(
                "Back-to-back planstate_update-only rejected | "
                "previous iteration was also planstate_update-only"
            )
            return True

        # When planstate_update and finish co-emit, planstate_update MUST run
        # first so it can mark remaining tasks terminal before finish's
        # pre-termination housekeeping check runs.
        meta_calls.sort(
            key=lambda tc: 0 if tc.tool_name.upper() == _PLANSTATE_UPDATE_TOOL_NAME else 1
        )

        async def execute_single_tool(
            tool_call: ToolCall,
        ) -> tuple[str, str, bool]:
            """Execute a single tool. Returns (tool_call_id, result, is_error)."""
            tool_name = tool_call.tool_name

            # If _parse_tool_calls flagged a validation error on this call,
            # don't try to execute. Return the validation error as the tool
            # result so the model sees it on the next iteration and can fix
            # its arguments. If the previous emission for THIS SAME tool
            # also parse-errored, the model is stuck in a malformed-JSON
            # loop — append a strong corrective hint so it knows to trim
            # large audit fields rather than retrying the same shape.
            if tool_call.parse_error is not None:
                error_text = f"Invalid arguments for tool '{tool_name}': {tool_call.parse_error}"
                if _previous_call_to_tool_parse_errored(messages, tool_name):
                    error_text += (
                        "\n\n[Consecutive parse_error for this tool.] "
                        "Common cause: malformed JSON from embedded quotes, "
                        "newlines, or copied tool output in a text field. "
                        "If this is planstate_update, keep `task.result` "
                        "and `task.inputs` SHORT — one plain-prose sentence "
                        "with no embedded quotes and no copied tool output. "
                        "The full tool output is already in message history; "
                        "audit fields are just brief summary pointers."
                    )
                tool_result = json.dumps({"error": error_text})
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
        # policy guard. After the guard, exactly one of these holds:
        # - meta_calls is empty (one or more action tools)
        # - meta_calls has 1 element (planstate_update OR finish alone)
        # - meta_calls has 2 elements (planstate_update + finish, in that
        #   order after the sort above; no non_meta_calls in this case)

        # PHASE A — meta tools, sequentially. When both planstate_update
        # and finish are present, planstate_update runs first (sorted above)
        # so its mutations are visible before finish's pre-termination
        # housekeeping check.
        meta_results: list[tuple[str, str, bool]] = []
        for tc in meta_calls:
            meta_results.append(await execute_single_tool(tc))

        # PHASE B — non-meta (action) tools, parallel or sequential. The
        # framework does NOT mutate plan_state from their results; that's
        # the model's job via planstate_update on the next iteration
        # (Reconcile-and-plan / Reconcile-and-finish modes).
        non_meta_results: list[tuple[str, str, bool]]
        if parallel:
            non_meta_results = list(
                await asyncio.gather(*[execute_single_tool(tc) for tc in non_meta_calls])
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

        return False


def _previous_call_to_tool_parse_errored(
    messages: list[ChatCompletionMessageParam],
    tool_name: str,
) -> bool:
    """Inspect message history to determine whether the most recent prior
    call to `tool_name` resulted in a parse_error.

    Walks backwards through history, skipping the current turn's
    just-appended assistant message. Finds the most recent assistant
    message that emitted `tool_name`; returns True iff that call's tool
    result message indicates a parse_error (content contains the
    `"Invalid arguments for tool"` marker).

    Used to detect repeated malformed-JSON loops on the same tool so we
    can surface a stronger corrective hint to the model.
    """
    upper_name = tool_name.upper()
    # Skip the current turn's assistant message (most recent assistant
    # with tool_calls), then find the most recent PRIOR assistant message
    # that emitted `tool_name`.
    found_current = False
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant":
            continue
        tool_calls = list(msg.get("tool_calls") or [])
        if not tool_calls:
            continue
        if not found_current:
            found_current = True
            continue
        # This is a prior assistant message — check if it emitted our tool.
        matching = [
            t.cast(dict[str, t.Any], tc)
            for tc in tool_calls
            if t.cast(dict[str, t.Any], tc).get("function", {}).get("name", "").upper()
            == upper_name
        ]
        if not matching:
            # Most recent prior assistant did not emit our tool; sequence
            # is broken — not "consecutive."
            return False
        # Find the corresponding tool result message and check for parse error.
        for tc in matching:
            tc_id = tc.get("id")
            for j in range(idx + 1, len(messages)):
                m2 = messages[j]
                if m2.get("role") == "tool" and m2.get("tool_call_id") == tc_id:
                    content = m2.get("content", "")
                    if isinstance(content, str) and "Invalid arguments for tool" in content:
                        return True
                    return False
        return False
    return False


def _previous_emission_was_planstate_only_success(
    messages: list[ChatCompletionMessageParam],
    skip_last_assistant: bool,
) -> bool:
    """Inspect message history to determine whether the previous turn's
    emission was a successful `planstate_update`-only call.

    A "successful planstate_update-only" turn is one where the assistant
    message contained exactly one tool_call (planstate_update) AND the
    corresponding tool result was NOT an error payload (so the call
    actually executed and mutated plan_state).

    Parse-errored and policy-rejected previous turns do NOT count — the
    model never actually reconciled, so retrying planstate_update is
    legitimate.

    Args:
        messages: Full message history.
        skip_last_assistant: If True, ignore the most recent assistant
            message with tool_calls (use this when called from
            `_execute_tool_calls` AFTER the current turn's assistant
            message has already been appended).
    """
    # Find the target assistant message with tool_calls.
    found_count = 0
    target_idx = -1
    needed = 2 if skip_last_assistant else 1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            found_count += 1
            if found_count == needed:
                target_idx = idx
                break
    if target_idx < 0:
        return False

    target_tool_calls = list(messages[target_idx].get("tool_calls") or [])
    if len(target_tool_calls) != 1:
        return False
    target_tc = t.cast(dict[str, t.Any], target_tool_calls[0])
    tc_name = target_tc.get("function", {}).get("name", "")
    if tc_name.upper() != _PLANSTATE_UPDATE_TOOL_NAME:
        return False

    # Find the matching tool-result message and inspect its content for
    # an error marker. Tool results follow their assistant message in
    # history, so we scan forward from target_idx.
    target_tc_id = target_tc.get("id")
    for idx in range(target_idx + 1, len(messages)):
        msg = messages[idx]
        if msg.get("role") == "tool" and msg.get("tool_call_id") == target_tc_id:
            content = msg.get("content", "")
            if isinstance(content, str) and '"error"' in content:
                return False
            return True
    return False


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
