"""
Direct Strategy - Single LLM call with tool_calling mode.

The simplest planning strategy: pass messages to LLM with tools,
let the LLM directly select and parameterize tools in one step.

When `plan_state` is provided, the current plan is injected as a system
message so the model can read its own progress, and any assistant-text
content the model emits alongside tool_calls is captured as a
free-form reasoning message in the conversation.

Best for:
- Simple tasks with clear tool mappings
- When reasoning overhead isn't needed
- Fast iteration cycles
- Plan-state-aware workflows (when the planstate_update tool is included)
"""

import typing as t

from openai.types.chat import ChatCompletionMessageParam

from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.agent_tool.plan_state import PlanState
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("direct_strategy.jinja")


def _extract_assistant_content(response: t.Any) -> str | None:
    """Pull assistant message content from a tool_calling response if present.

    OpenAI's tool_calling mode can return both assistant text and tool_calls
    in the same response. ToolCallResponse doesn't expose `.content`
    directly, so we read it from `raw_response.choices[0].message.content`.
    Returns None if no content is available.
    """
    raw = getattr(response, "raw_response", None)
    if raw is None:
        return None
    try:
        content = raw.choices[0].message.content
        return content if content else None
    except (AttributeError, IndexError):
        return None


class DirectStrategy(PlanningStrategy):
    """
    Direct strategy: Single LLM call with tool_calling mode.

    Flow:
        messages → LLM (tool_calling) → tool_calls → execute

    Best for:
        - Simple tasks with clear tool mappings
        - When reasoning overhead isn't needed
        - Fast iteration cycles

    The LLM directly selects and parameterizes tools in one step.

    Return behavior:
        - No tool_calls → tool_calls=[], success=False (strategy-internal
          terminate; AgentTool reads success/result here).
        - Any tool_calls (including `finish`) → returned as-is. AgentTool
          executes them and detects `finish` to drive termination.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        direct_prompt: str | None = None,
    ):
        """
        Initialize DirectStrategy.

        Args:
            llm_client: LLM client for generation
            model: Optional model override
            direct_prompt: Custom prompt for tool selection (uses template if None)
        """
        self.llm_client = llm_client
        self.model = model
        self.direct_prompt = direct_prompt

    def _get_direct_prompt(self, tools: list[BaseTool[t.Any, t.Any]]) -> str:
        """Get the action prompt from template or custom."""
        if self.direct_prompt:
            return self.direct_prompt
        tool_names = [tool.name for tool in tools]
        return _templates.direct_prompt(tool_names=tool_names)

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
        plan_state: PlanState | None = None,
    ) -> StrategyOutput:
        """
        Generate next actions via single LLM call.

        Args:
            messages: Current conversation history
            tools: Available tools (including finish tool)
            parallel_tool_calls: Allow LLM to return multiple tool calls
            plan_state: Optional durable plan state. If provided, the current
                serialized plan is injected as a system message so the model
                can read its own progress. DirectStrategy never mutates
                plan_state; mutation happens only via the model's
                planstate_update tool calls.

        Returns:
            StrategyOutput with tool_calls or finished status
        """
        plan_messages: list[ChatCompletionMessageParam] = []
        if plan_state is not None:
            plan_messages.append(
                {
                    "role": "system",
                    "content": (
                        "## Current Plan State\n" + plan_state.serialize_for_prompt()
                    ),
                }
            )

        call_messages = (
            list(messages)
            + plan_messages
            + [{"role": "user", "content": self._get_direct_prompt(tools)}]
        )
        response = await self.llm_client.agenerate(
            messages=call_messages,
            model=self.model,
            mode="tool_calling",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Capture any assistant text content emitted alongside tool_calls.
        # Returned as a separate text-only assistant message so the durable
        # history records the model's rationale; the assistant tool_calls
        # message is appended later by _execute_tool_calls.
        output_messages: list[ChatCompletionMessageParam] = []
        content = _extract_assistant_content(response)
        if content:
            output_messages.append({"role": "assistant", "content": content})

        # No tool calls → strategy-internal terminate. AgentTool reads
        # success/result directly from this StrategyOutput (we treat the
        # absence of any tool call — including `finish` — as an unsuccessful
        # termination, since the model didn't signal completion explicitly).
        if not response.tool_calls:
            return StrategyOutput(
                messages=output_messages,
                tool_calls=[],
                success=False,
                result=(
                    content
                    or response.finish_reason
                    or "No tool calls returned by LLM"
                ),
            )

        # Return tool_calls (possibly including `finish`) for AgentTool to
        # execute. AgentTool detects `finish` after the policy guard runs and
        # terminates with the args from the finish tool's arguments.
        return StrategyOutput(
            messages=output_messages,
            tool_calls=response.tool_calls,
        )
