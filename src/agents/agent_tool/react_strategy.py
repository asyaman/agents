"""
React Strategy - Reason-Act-Observe pattern.

The ReAct pattern separates reasoning from action:
1. Reasoning phase: Analyze state, plan steps, identify IMMEDIATE next action
2. Action phase: Select tool(s) for ONLY the next step
3. (AgentTool executes and adds results to messages)
4. Next iteration: Re-reason with new information

Best for:
- Complex multi-step tasks
- Tasks requiring adaptation based on intermediate results
- When explicit reasoning improves accuracy

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

import json
import typing as t

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.agent_tool.meta_tool_plan_state_update import (
    PlanStateUpdate,
    PlanStateUpdateInput,
)
from agents.agent_tool.plan_state import PlanState
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("react_strategy.jinja")

# Tool name AgentTool uses for the planstate_update meta tool. Strategy
# extracts the tool by this name (case-insensitive) when running the
# auto-translate step.
_PLANSTATE_UPDATE_TOOL_NAME = "PLANSTATE_UPDATE"


class ReactStrategy(PlanningStrategy):
    """
    React strategy: Reason-Act-Observe pattern, optionally with an
    intermediate plan-translation stage that keeps the structured
    `PlanState` aligned with the model's free-text reasoning.

    Flow per iteration:
        1. Reasoning: free-text analysis of state and prior tool results;
           identifies the IMMEDIATE next action.
        2. Plan translation (OPTIONAL, default ON via
           `auto_translate_plan=True` when `planstate_update` is among the
           tools): a focused LLM call takes the reasoning + current
           `plan_state` and emits at most one `planstate_update` call that
           mirrors the reasoning's intent into structured plan changes
           (mark `in_progress`, add retries with `parent_attempt_id`, fan
           out, mark blocked/cancelled). The strategy invokes the
           `planstate_update` meta tool to mutate `plan_state` in place.
           The action phase then sees the rebuilt plan block.
        3. Action: dispatch the tool(s) for the next step.
        4. (AgentTool executes the tool calls and appends results to
           `messages`; Phase D auto-completes the `in_progress` task that
           the translator just set up.)
        5. Next iteration: re-reason with the new tool result.

    Best for:
        - Complex multi-step tasks where the plan branches on tool
          results (retries, conditional next steps, fan-outs).
        - Tasks where `plan_state.tasks` should remain a faithful audit
          trail of what actually happened (the translator step is what
          keeps it faithful — without it, the model has to call
          `planstate_update` itself between actions).
        - Settings where reasoning quality improves with explicit text.

    Key insight: the action phase focuses on the IMMEDIATE next step,
    not the entire plan. The translator stage is what bridges free-text
    reasoning and structured plan state — it does NOT do work; it only
    records the model's intent so AgentTool's auto-update has the right
    `in_progress` target when the action runs.

    Cost: with `auto_translate_plan=True` each iteration makes 3 LLM
    round-trips (reason → translate → act). Pass a smaller/cheaper
    `plan_translator_client`/`plan_translator_model` to keep the
    translator cheap. Set `auto_translate_plan=False` to skip the stage
    entirely (2 round-trips, but `plan_state.tasks` will drift behind
    reality unless the model emits `planstate_update` itself).

    Return behavior (after action phase):
        - No tool_calls → `tool_calls=[]`, `success=False`
          (strategy-internal terminate; AgentTool reads success/result).
        - Any tool_calls (including `finish`) → returned as-is. AgentTool
          executes them and detects `finish` to drive termination.

    Note: reasoning is always included in output messages, even on
    finish. The translator's internal LLM call is NOT added to the
    durable message history; only its plan_state side effect persists.

    Example (with translator):
        Iteration 1:
            Reasoning:    "Need: 1) get users, 2) update profiles. First, get users."
            Translation:  planstate_update([t1=get_users in_progress, t2=update pending])
            Action:       get_users(filter="active")

        Iteration 2:
            Reasoning:    "Got 5 users. Now update each profile in parallel."
            Translation:  planstate_update([t1=completed, t3..t7=update_profile
                          in_progress (one per user, no inter-deps), ...])
            Action:       [parallel] update_profile(user_id=1), ..., update_profile(user_id=5)
    """

    def __init__(
        self,
        action_client: LLMClient,
        action_model: str | None = None,
        reasoning_client: LLMClient | None = None,
        reasoning_model: str | None = None,
        reasoning_prompt: str | None = None,
        action_prompt: str | None = None,
        auto_translate_plan: bool = True,
        plan_translator_client: LLMClient | None = None,
        plan_translator_model: str | None = None,
        plan_translator_prompt: str | None = None,
        plan_translator_max_retries: int = 1,
    ):
        """
        Initialize ReactStrategy.

        Args:
            action_client: LLM client for action phase (tool selection)
            action_model: Model for action phase (uses client default if None)
            reasoning_client: LLM client for reasoning phase (uses action_client if None)
            reasoning_model: Model for reasoning phase (uses action_model if None)
            reasoning_prompt: Custom prompt for reasoning phase (uses template if None)
            action_prompt: Custom prompt for action phase (uses template if None)
            auto_translate_plan: When True (default) and `planstate_update` is
                among the available tools, an extra LLM call between reasoning
                and action translates the free-text reasoning into a structured
                planstate_update so the action phase sees a faithful plan.
                Set False to skip that step (e.g., to save a round-trip when
                you trust the model to call planstate_update itself).
            plan_translator_client: LLM client for the plan-translation step.
                Defaults to `reasoning_client` so a single small/cheap model
                can run both. Only used when auto_translate_plan is True.
            plan_translator_model: Model for the plan-translation step
                (defaults to `reasoning_model`).
            plan_translator_prompt: Custom prompt for the plan-translation
                step (defaults to template).
            plan_translator_max_retries: Number of additional translator
                attempts when the LLM emits invalid `planstate_update` args
                (parse error or `acall` failure). Each retry feeds the prior
                error back into the translator's context so the model can
                fix its output. Default is 1 (so up to 2 total attempts).
                Set to 0 to disable retries entirely.
        """
        self.action_client = action_client
        self.action_model = action_model
        # Default reasoning to use same client/model as action if not specified
        self.reasoning_client = reasoning_client or action_client
        self.reasoning_model = reasoning_model or action_model
        self.reasoning_prompt = reasoning_prompt
        self.action_prompt = action_prompt
        self.auto_translate_plan = auto_translate_plan
        # Default translator to use same client/model as reasoning
        self.plan_translator_client = plan_translator_client or self.reasoning_client
        self.plan_translator_model = plan_translator_model or self.reasoning_model
        self.plan_translator_prompt = plan_translator_prompt
        self.plan_translator_max_retries = plan_translator_max_retries

    def _get_reasoning_prompt(
        self,
        tools: list[BaseTool[t.Any, t.Any]],
        auto_translate: bool = False,
    ) -> str:
        """Get the reasoning prompt from template or custom.

        `auto_translate` switches the planstate-update guidance: when True,
        the prompt tells the model that an automated translator manages
        plan structure (so the model expresses plan changes in reasoning
        text rather than emitting `planstate_update` itself).
        """
        if self.reasoning_prompt:
            return self.reasoning_prompt
        tool_names = [tool.name for tool in tools]
        return _templates.reasoning_prompt(
            tool_names=tool_names, auto_translate=auto_translate
        )

    def _get_action_prompt(self, auto_translate: bool = False) -> str:
        """Get the action prompt from template or custom.

        When `auto_translate=True`, the prompt explicitly forbids
        `planstate_update` calls in the action phase (the translator
        already ran).
        """
        if self.action_prompt:
            return self.action_prompt
        return _templates.action_prompt(auto_translate=auto_translate)

    def _get_plan_translator_prompt(self) -> str:
        """Get the plan-translation prompt from template or custom."""
        if self.plan_translator_prompt:
            return self.plan_translator_prompt
        return _templates.plan_translation_prompt()

    @staticmethod
    def _find_planstate_update_tool(
        tools: list[BaseTool[t.Any, t.Any]],
    ) -> PlanStateUpdate | None:
        """Locate the planstate_update meta tool in the available tools list.

        Returns None if it isn't present (e.g., AgentTool was constructed with
        include_planstate_update_tool=False) — the caller should then skip
        the translation step entirely.
        """
        for tool in tools:
            if tool.name.upper() == _PLANSTATE_UPDATE_TOOL_NAME:
                if isinstance(tool, PlanStateUpdate):
                    return tool
        return None

    async def _translate_reasoning_to_plan(
        self,
        history: list[ChatCompletionMessageParam],
        plan_messages: list[ChatCompletionMessageParam],
        reasoning: str,
        planstate_update_tool: PlanStateUpdate,
    ) -> bool:
        """Run a focused LLM call to translate the reasoning into a plan
        update. Mutates the plan_state closured by `planstate_update_tool`
        when the translator emits an update.

        On a parse error or `acall` failure the translator's failed tool
        call and the resulting error are appended to the translator's
        local message history and the LLM is re-called — same pattern as
        execute_single_tool surfacing errors back to the model. Capped by
        `self.plan_translator_max_retries` to avoid runaway retry loops.

        Returns:
            True if the translator emitted a planstate_update and it ran
            successfully; False otherwise (no-op, wrong tool, or all
            retries exhausted).
        """
        translator_messages: list[ChatCompletionMessageParam] = (
            list(history)
            + plan_messages
            + [
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": self._get_plan_translator_prompt()},
            ]
        )

        # Cast through Any to bypass BaseTool's invariant generic parameters —
        # agenerate's signature accepts BaseTool[BaseModel, BaseModel] but
        # PlanStateUpdate has narrower I/O generics; the tool itself is
        # compatible at runtime.
        translator_tools = t.cast(
            "list[BaseTool[t.Any, t.Any]]", [planstate_update_tool]
        )

        max_attempts = self.plan_translator_max_retries + 1
        last_error: str | None = None

        for attempt in range(max_attempts):
            response = await self.plan_translator_client.agenerate(
                messages=translator_messages,
                model=self.plan_translator_model,
                mode="tool_calling",
                tools=translator_tools,
                parallel_tool_calls=False,
            )

            if not response.tool_calls:
                # Translator decided no change is needed. Don't retry — this
                # is a legitimate no-op signal, not an error.
                logger.debug("Plan translator: no change emitted")
                return False

            translator_call = next(
                (
                    tc
                    for tc in response.tool_calls
                    if tc.tool_name.upper() == _PLANSTATE_UPDATE_TOOL_NAME
                ),
                None,
            )
            if translator_call is None:
                # Wrong tool emitted — model error not recoverable through
                # arg-fixing retries; bail.
                logger.warning(
                    "Plan translator emitted non-planstate_update tool call;"
                    " ignoring | tools={}",
                    [tc.tool_name for tc in response.tool_calls],
                )
                return False

            # Try to validate + apply.
            error: str | None = None
            if translator_call.parse_error is not None:
                error = (
                    "Invalid arguments for planstate_update: "
                    f"{translator_call.parse_error}"
                )
            elif not isinstance(translator_call.parsed, PlanStateUpdateInput):
                error = (
                    "Parsed input has unexpected type: "
                    f"{type(translator_call.parsed).__name__}"
                )
            else:
                try:
                    output = await planstate_update_tool.acall(
                        translator_call.parsed
                    )
                    logger.debug(
                        "Plan translator: applied update | revision={} | "
                        "plan_status={}",
                        output.revision,
                        output.plan_status,
                    )
                    return True
                except Exception as e:
                    error = f"planstate_update execution raised: {e}"

            # We have an error. Decide whether to retry.
            last_error = error
            attempts_left = max_attempts - attempt - 1
            if attempts_left <= 0:
                logger.warning(
                    "Plan translator: out of retries; skipping mutation | "
                    "last_error={}",
                    error[:200],
                )
                return False

            logger.debug(
                "Plan translator: attempt {}/{} failed; retrying. error={}",
                attempt + 1,
                max_attempts,
                error[:200],
            )

            # Append the failed tool call + error result so the next attempt
            # sees its own mistake (mirrors execute_single_tool's "surface
            # error to model" pattern). The arguments are serialized as JSON
            # for the assistant message so the OpenAI API accepts them.
            args_str = (
                translator_call.arguments
                if isinstance(translator_call.arguments, str)
                else json.dumps(translator_call.arguments)
            )
            translator_messages.append(
                t.cast(
                    "ChatCompletionMessageParam",
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": translator_call.id,
                                "type": "function",
                                "function": {
                                    "name": translator_call.tool_name,
                                    "arguments": args_str,
                                },
                            }
                        ],
                    },
                )
            )
            translator_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": translator_call.id,
                    "content": json.dumps({"error": error}),
                }
            )

        # Defensive: the loop returns from inside on every path, but mypy
        # doesn't know that. Treat as exhausted retries.
        logger.warning(
            "Plan translator: exhausted attempts | last_error={}",
            (last_error or "unknown")[:200],
        )
        return True

    @staticmethod
    def _build_plan_messages(
        plan_state: PlanState | None,
    ) -> list[ChatCompletionMessageParam]:
        """Render the plan_state as a system-message block for prompt injection."""
        if plan_state is None:
            return []
        return [
            {
                "role": "system",
                "content": (
                    "## Current Plan State\n" + plan_state.serialize_for_prompt()
                ),
            }
        ]

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
        plan_state: PlanState | None = None,
    ) -> StrategyOutput:
        """
        Generate next actions using Reason-Act pattern.

        Args:
            messages: Current conversation history
            tools: Available tools (including finish tool)
            parallel_tool_calls: Allow LLM to return multiple tool calls
            plan_state: Optional durable plan state. If provided, the current
                serialized plan is injected as a system message into the
                reasoning phase so the model can read its own progress.
                With `auto_translate_plan=True` (default), an intermediate
                LLM call between reasoning and action translates the
                free-text reasoning into a structured planstate_update
                that mutates plan_state before the action phase runs.

        Returns:
            StrategyOutput with reasoning messages, tool_calls, and/or finished status
        """
        # Whether the plan translator will run this iteration. Computed
        # upfront so the reasoning and action prompts can reflect that the
        # translator is the one writing planstate_update (not the model).
        translator_active = (
            self.auto_translate_plan
            and plan_state is not None
            and self._find_planstate_update_tool(tools) is not None
        )

        # Phase 1: Reasoning (text response)
        # Inject the plan block so reasoning can see task progress.
        plan_messages = self._build_plan_messages(plan_state)

        reasoning_messages = (
            list(messages)
            + plan_messages
            + [
                {
                    "role": "user",
                    "content": self._get_reasoning_prompt(
                        tools, auto_translate=translator_active
                    ),
                }
            ]
        )

        reasoning_response = await self.reasoning_client.agenerate(
            messages=reasoning_messages,
            model=self.reasoning_model,
            mode="text",
        )
        reasoning = reasoning_response.content or ""

        # Phase 1.5: Plan translation (LLM call → planstate_update).
        # Translates the free-text reasoning into a structured plan delta so
        # plan_state.tasks stays a faithful projection of the model's intent
        # (parallel batches, retries, blocked/cancelled, new tasks). After
        # this runs we rebuild plan_messages so the action phase sees the
        # updated plan.
        if translator_active:
            planstate_update_tool = self._find_planstate_update_tool(tools)
            assert planstate_update_tool is not None  # narrowed by translator_active
            await self._translate_reasoning_to_plan(
                history=messages,
                plan_messages=plan_messages,
                reasoning=reasoning,
                planstate_update_tool=planstate_update_tool,
            )
            # Re-render the plan block from the (possibly mutated) plan_state
            # so the action phase prompt reflects the translation.
            plan_messages = self._build_plan_messages(plan_state)

        # Phase 2: Action selection (tool_calling)
        # When the translator ran, plan_state structure is now its job, not the
        # action LLM's — exclude planstate_update from the action tool list so
        # the action phase focuses on real work tools (+ finish). Without this,
        # the model would often emit a redundant planstate_update as its
        # "action", burning iterations and producing duplicate plan revisions.
        action_tools = tools
        if translator_active:
            action_tools = [
                tool
                for tool in tools
                if tool.name.upper() != _PLANSTATE_UPDATE_TOOL_NAME
            ]

        action_messages = (
            list(messages)
            + plan_messages
            + [
                {"role": "assistant", "content": reasoning},
                {
                    "role": "user",
                    "content": self._get_action_prompt(
                        auto_translate=translator_active
                    ),
                },
            ]
        )

        response = await self.action_client.agenerate(
            messages=action_messages,
            model=self.action_model,
            mode="tool_calling",
            tools=action_tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Include reasoning in output messages (becomes part of conversation)
        output_messages: list[ChatCompletionMessageParam] = [
            {"role": "assistant", "content": reasoning}
        ]

        # No tool calls → strategy-internal terminate. AgentTool reads
        # success/result directly from this StrategyOutput. (We treat the
        # absence of any tool — including `finish` — as an unsuccessful
        # termination, since the model didn't signal completion explicitly.)
        if not response.tool_calls:
            return StrategyOutput(
                messages=output_messages,
                tool_calls=[],
                success=False,
                result=response.finish_reason
                or reasoning + " No tool calls returned by LLM",
            )

        # Pass tool_calls through as-is (including any `finish`). AgentTool
        # detects `finish` after the policy guard runs and terminates with
        # the args from finish's arguments.
        return StrategyOutput(
            messages=output_messages,
            tool_calls=response.tool_calls,
        )
